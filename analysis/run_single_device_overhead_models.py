from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import DEFAULT_OUTPUT_DIR
from extract_marker_free_cycle_features import extract_profile


PROFILER_MODE = "nsys_gpu_metrics"
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
QUALITY_FEATURES = (
    "detector_confidence",
    "emission_horizon_seconds",
    "feature_cycle_count",
)


def load_current_marker_free_features(root: Path) -> pd.DataFrame:
    detection_path = (
        root
        / "model_experiments"
        / "pre6g_multiscale_detector_part_c2"
        / "yolo_deployment_detections.csv"
    )
    detections = pd.read_csv(detection_path)
    detections = detections[detections.accepted].copy()
    manifest = pd.read_csv(root / "marker_free_timeseries_manifest.csv")
    detections = detections.merge(
        manifest[["device_id", "workload_id", "cache_path"]],
        on=["device_id", "workload_id"],
        validate="one_to_one",
    )
    rows = [
        extract_profile(pd.Series(item._asdict()), Path(item.cache_path))
        for item in detections.itertuples(index=False)
    ]
    features = pd.DataFrame(rows).sort_values(["device_id", "workload_id"])
    if len(features) != 48:
        raise ValueError(f"Expected 48 accepted multiscale YOLO profiles; got {len(features)}.")
    return features


def build_samples(root: Path) -> pd.DataFrame:
    features = load_current_marker_free_features(root)
    overhead = pd.read_csv(root / "profiler_overhead.csv")
    overhead = overhead[overhead.profiler_mode == PROFILER_MODE].copy()
    overhead = overhead.rename(
        columns={
            "profile_runtime_ms": "actual_profile_runtime_ms",
            "baseline_ground_truth_ms": "actual_dry_runtime_ms",
            "overhead_ratio": "actual_overhead_ratio",
            "overhead_percent": "actual_overhead_percent",
        }
    )
    label_columns = [
        "device_id",
        "workload_id",
        "actual_profile_runtime_ms",
        "actual_dry_runtime_ms",
        "actual_overhead_ratio",
        "actual_overhead_percent",
    ]
    data = features.merge(
        overhead[label_columns],
        on=["device_id", "workload_id"],
        how="inner",
        validate="one_to_one",
    )
    data["log_detected_period_ms"] = np.log(data.detected_period_ms.astype(float))
    required = label_columns[2:] + ["detected_period_ms"]
    if data[required].isna().any().any():
        raise ValueError("Missing profiling-overhead labels or detected periods.")
    if (data.actual_overhead_ratio <= 0).any():
        raise ValueError("Overhead ratios must be positive before log transformation.")
    counts = data.groupby("device_id").workload_id.nunique()
    if len(counts) != 2 or not (counts == 24).all():
        raise ValueError(f"Expected 24 YOLO workloads on each of two devices; got {counts.to_dict()}.")
    return data.sort_values(["device_id", "workload_id"]).reset_index(drop=True)


def feature_sets(data: pd.DataFrame) -> dict[str, list[str]]:
    behavior = sorted(column for column in data if column.startswith("mf_"))
    compact_suffixes = (
        "_mean",
        "_p90",
        "_active_fraction",
        "_cycle_mean_cv",
        "_harmonic1_fraction",
        "_cycle_repeatability",
    )
    compact = sorted(
        column
        for column in behavior
        if column.startswith("mf_summary_")
        or column.startswith("mf_cross_corr_")
        or column.endswith(compact_suffixes)
    )
    quality = list(QUALITY_FEATURES)
    return {
        "period_only_ridge": ["log_detected_period_ms"],
        "period_quality_ridge": ["log_detected_period_ms"] + quality,
        "behavior_only_ridge": behavior,
        "period_compact_ridge": ["log_detected_period_ms"] + compact + quality,
        "period_full_ridge": ["log_detected_period_ms"] + behavior + quality,
    }


def make_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold()),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def target_values(frame: pd.DataFrame) -> np.ndarray:
    return np.log(frame.actual_overhead_ratio.to_numpy(float))


def predict_ratio(model: Pipeline, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    # Profiling is not expected to accelerate the workload. The one observed
    # ratio below one is retained as label noise, while deployment predictions
    # are constrained to a physically meaningful non-negative overhead.
    return np.maximum(1.0, np.exp(model.predict(frame[features])))


def overhead_mae_pp(actual_ratio: np.ndarray, predicted_ratio: np.ndarray) -> float:
    return float(np.mean(np.abs(predicted_ratio - actual_ratio)) * 100.0)


def choose_alpha(train: pd.DataFrame, features: list[str]) -> tuple[float, float]:
    splitter = GroupKFold(n_splits=min(5, train.workload_id.nunique()))
    trials = []
    for order, alpha in enumerate(ALPHAS):
        actual_parts = []
        predicted_parts = []
        for fit_index, validation_index in splitter.split(
            train, groups=train.workload_id
        ):
            fit = train.iloc[fit_index]
            validation = train.iloc[validation_index]
            model = make_pipeline(alpha)
            model.fit(fit[features], target_values(fit))
            actual_parts.extend(validation.actual_overhead_ratio.to_numpy(float))
            predicted_parts.extend(predict_ratio(model, validation, features))
        score = overhead_mae_pp(
            np.asarray(actual_parts), np.asarray(predicted_parts)
        )
        trials.append((score, order, alpha))
    best = min(trials, key=lambda item: (item[0], item[1]))
    return float(best[2]), float(best[0])


def prediction_record(
    row: pd.Series,
    model_name: str,
    predicted_ratio: float,
    feature_count: int,
    alpha: float | None = None,
    inner_score: float | None = None,
    selected_family: str = "",
) -> dict:
    actual_ratio = float(row.actual_overhead_ratio)
    detected_period = float(row.detected_period_ms)
    estimated_dry = detected_period / predicted_ratio
    actual_dry = float(row.actual_dry_runtime_ms)
    return {
        "device_id": row.device_id,
        "workload_id": row.workload_id,
        "profiler_mode": PROFILER_MODE,
        "model": model_name,
        "selected_feature_family": selected_family,
        "input_feature_count": feature_count,
        "actual_profile_runtime_ms": float(row.actual_profile_runtime_ms),
        "detected_period_ms": detected_period,
        "actual_dry_runtime_ms": actual_dry,
        "estimated_dry_runtime_ms": estimated_dry,
        "actual_overhead_ratio": actual_ratio,
        "predicted_overhead_ratio": predicted_ratio,
        "actual_overhead_percent": (actual_ratio - 1.0) * 100.0,
        "predicted_overhead_percent": (predicted_ratio - 1.0) * 100.0,
        "overhead_abs_error_pp": abs(predicted_ratio - actual_ratio) * 100.0,
        "dry_runtime_ape_percent": abs(estimated_dry - actual_dry) / actual_dry * 100.0,
        "selected_alpha": alpha,
        "inner_overhead_mae_pp": inner_score,
    }


def outer_predictions(data: pd.DataFrame) -> list[dict]:
    specs = feature_sets(data)
    records = []
    for device, frame in data.groupby("device_id", sort=True):
        for workload in sorted(frame.workload_id.unique()):
            train = frame[frame.workload_id != workload]
            test = frame[frame.workload_id == workload]
            row = test.iloc[0]

            median_ratio = float(train.actual_overhead_ratio.median())
            records.append(
                prediction_record(
                    row,
                    "constant_median_ratio",
                    median_ratio,
                    feature_count=0,
                )
            )

            median_added_ms = float(
                (train.actual_profile_runtime_ms - train.actual_dry_runtime_ms).median()
            )
            estimated_dry = max(float(row.detected_period_ms) - median_added_ms, 1e-6)
            additive_ratio = float(row.detected_period_ms) / estimated_dry
            records.append(
                prediction_record(
                    row,
                    "constant_median_added_ms",
                    additive_ratio,
                    feature_count=1,
                )
            )

            family_trials = []
            for family_order, (name, features) in enumerate(specs.items()):
                alpha, inner_score = choose_alpha(train, features)
                model = make_pipeline(alpha)
                model.fit(train[features], target_values(train))
                predicted_ratio = float(predict_ratio(model, test, features)[0])
                records.append(
                    prediction_record(
                        row,
                        name,
                        predicted_ratio,
                        feature_count=len(features),
                        alpha=alpha,
                        inner_score=inner_score,
                    )
                )
                family_trials.append(
                    (inner_score, family_order, name, features, alpha, predicted_ratio)
                )

            selected = min(family_trials, key=lambda item: (item[0], item[1]))
            inner_score, _, name, features, alpha, predicted_ratio = selected
            records.append(
                prediction_record(
                    row,
                    "nested_selected_ridge",
                    predicted_ratio,
                    feature_count=len(features),
                    alpha=alpha,
                    inner_score=inner_score,
                    selected_family=name,
                )
            )
    return records


def metric_record(device: str, model: str, group: pd.DataFrame) -> dict:
    actual_ratio = group.actual_overhead_ratio.to_numpy(float)
    predicted_ratio = group.predicted_overhead_ratio.to_numpy(float)
    errors_pp = np.abs(predicted_ratio - actual_ratio) * 100.0
    signed_pp = (predicted_ratio - actual_ratio) * 100.0
    dry_ape = group.dry_runtime_ape_percent.to_numpy(float)
    return {
        "device_id": device,
        "profiler_mode": PROFILER_MODE,
        "model": model,
        "n": len(group),
        "overhead_mae_pp": float(np.mean(errors_pp)),
        "overhead_rmse_pp": float(np.sqrt(np.mean(signed_pp**2))),
        "overhead_median_ae_pp": float(np.median(errors_pp)),
        "overhead_p90_ae_pp": float(np.quantile(errors_pp, 0.9)),
        "overhead_max_ae_pp": float(np.max(errors_pp)),
        "overhead_bias_pp": float(np.mean(signed_pp)),
        "ratio_mape_percent": float(
            np.mean(np.abs(predicted_ratio - actual_ratio) / actual_ratio) * 100.0
        ),
        "estimated_dry_runtime_mape_percent": float(np.mean(dry_ape)),
        "estimated_dry_runtime_median_ape_percent": float(np.median(dry_ape)),
        "estimated_dry_runtime_max_ape_percent": float(np.max(dry_ape)),
    }


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (device, model), group in predictions.groupby(
        ["device_id", "model"], sort=True
    ):
        rows.append(metric_record(device, model, group))
    for model, group in predictions.groupby("model", sort=True):
        rows.append(metric_record("both_devices_macro", model, group))
    return pd.DataFrame(rows).sort_values(
        ["device_id", "overhead_mae_pp", "model"]
    )


def fit_production_models(
    data: pd.DataFrame, specs: dict[str, list[str]], output: Path
) -> pd.DataFrame:
    model_dir = output / "models"
    example_dir = output / "examples"
    model_dir.mkdir(parents=True, exist_ok=True)
    example_dir.mkdir(parents=True, exist_ok=True)
    selections = []
    for device, frame in data.groupby("device_id", sort=True):
        trials = []
        for family_order, (name, features) in enumerate(specs.items()):
            alpha, score = choose_alpha(frame, features)
            trials.append((score, family_order, name, features, alpha))
        score, _, name, features, alpha = min(
            trials, key=lambda item: (item[0], item[1])
        )
        model = make_pipeline(alpha)
        model.fit(frame[features], target_values(frame))
        model_raw_fields = [
            "detected_period_ms" if feature == "log_detected_period_ms" else feature
            for feature in features
        ]
        # detected_period_ms is always required to turn the predicted overhead
        # ratio into an estimated dry runtime, even for behavior-only models.
        raw_required_fields = list(
            dict.fromkeys(["detected_period_ms"] + model_raw_fields)
        )
        model_path = model_dir / f"{device}_{PROFILER_MODE}_overhead_ridge.joblib"
        example_path = example_dir / f"{device}_{PROFILER_MODE}_example.csv"
        joblib.dump(
            {
                "pipeline": model,
                "features": features,
                "raw_required_fields": raw_required_fields,
                "device_id": device,
                "profiler_mode": PROFILER_MODE,
                "workload_family": "yolo26",
                "predicts": "log(profile runtime / unprofiled dry runtime)",
                "prediction_constraint": "predicted overhead ratio is clipped to at least 1.0",
                "overhead_percent_formula": "(max(1, exp(model_output)) - 1) * 100",
                "estimated_dry_runtime_formula": "detected_period_ms / max(1, exp(model_output))",
                "validated_scope": "unseen YOLO26 workload on this known device",
            },
            model_path,
        )
        example_columns = ["device_id", "workload_id"] + raw_required_fields
        frame.iloc[[0]][example_columns].to_csv(example_path, index=False)
        selections.append(
            {
                "device_id": device,
                "profiler_mode": PROFILER_MODE,
                "selected_feature_family": name,
                "selected_alpha": alpha,
                "grouped_cv_overhead_mae_pp": score,
                "training_workloads": frame.workload_id.nunique(),
                "feature_count": len(features),
                "model_path": str(model_path.relative_to(output)),
                "example_input_path": str(example_path.relative_to(output)),
            }
        )
    return pd.DataFrame(selections)


def write_report(
    output: Path,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    selections: pd.DataFrame,
) -> None:
    lines = [
        "# YOLO single-device profiling-overhead prediction",
        "",
        "This experiment predicts Nsight GPU Metrics overhead independently for each known device. Inputs are marker-free profiling period, detector quality, and GPU metric cycle behavior. The dry-run baseline and profiler steady-window runtime are labels only.",
        "",
        "Outer evaluation is leave-one-workload-out within each device. Ridge alpha and the primary nested feature family are selected using only inner grouped folds.",
        "",
        "Overhead error is reported in percentage points (pp), because percentage error of the overhead itself is unstable near zero. Estimated dry runtime uses `detected_period_ms / predicted_overhead_ratio`.",
        "",
        "## Outer leave-one-workload-out results",
        "",
        "| Device | Model | N | Overhead MAE | Median AE | P90 AE | Max AE | Bias | Estimated dry-runtime MAPE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics[metrics.device_id != "both_devices_macro"].itertuples():
        lines.append(
            f"| {row.device_id} | {row.model} | {row.n} | {row.overhead_mae_pp:.2f} pp | {row.overhead_median_ae_pp:.2f} pp | {row.overhead_p90_ae_pp:.2f} pp | {row.overhead_max_ae_pp:.2f} pp | {row.overhead_bias_pp:+.2f} pp | {row.estimated_dry_runtime_mape_percent:.2f}% |"
        )
    lines += [
        "",
        "## Production fits",
        "",
        "| Device | Selected family | Alpha | Grouped-CV overhead MAE | Features |",
        "|---|---|---:|---:|---:|",
    ]
    for row in selections.itertuples():
        lines.append(
            f"| {row.device_id} | {row.selected_feature_family} | {row.selected_alpha:g} | {row.grouped_cv_overhead_mae_pp:.2f} pp | {row.feature_count} |"
        )
    primary = predictions[predictions.model == "nested_selected_ridge"].nlargest(
        10, "overhead_abs_error_pp"
    )
    lines += [
        "",
        "## Largest primary out-of-fold errors",
        "",
        "| Device | Workload | Actual overhead | Predicted overhead | Absolute error | Actual dry ms | Estimated dry ms |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in primary.itertuples():
        lines.append(
            f"| {row.device_id} | {row.workload_id} | {row.actual_overhead_percent:.2f}% | {row.predicted_overhead_percent:.2f}% | {row.overhead_abs_error_pp:.2f} pp | {row.actual_dry_runtime_ms:.3f} | {row.estimated_dry_runtime_ms:.3f} |"
        )
    lines += [
        "",
        "## Interpretation and limits",
        "",
        "- The validated scope is a held-out YOLO configuration on RTX4090 or RTX5090, not an unseen model family or unseen GPU.",
        "- This model applies only to `nsys_gpu_metrics`; profiler modes have materially different overhead distributions.",
        "- Each profiler/workload label currently comes from one profiling run, while dry ground truth is the median of three baselines. Small or negative observed overheads can therefore contain run-to-run noise.",
        "- The model estimates total observed steady-window slowdown; it does not separately identify counter sampling, CUDA interception, host scheduling, or detector error.",
        "- Workload metadata, iteration markers, callbacks, and dry runtime are excluded from deployment inputs.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate per-device YOLO Nsight GPU Metrics overhead models."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    output = root / "model_experiments" / "yolo_single_device_overhead"
    output.mkdir(parents=True, exist_ok=True)

    data = build_samples(root)
    specs = feature_sets(data)
    data.to_csv(output / "samples.csv", index=False)
    predictions = pd.DataFrame(outer_predictions(data))
    predictions.to_csv(output / "predictions.csv", index=False)
    metrics = summarize(predictions)
    metrics.to_csv(output / "metrics.csv", index=False)
    selections = fit_production_models(data, specs, output)
    selections.to_csv(output / "production_selection.csv", index=False)

    manifest = []
    for model, features in specs.items():
        for feature in features:
            manifest.append(
                {
                    "model": model,
                    "feature": feature,
                    "family": (
                        "marker_free_period"
                        if feature == "log_detected_period_ms"
                        else "detector_quality"
                        if feature in QUALITY_FEATURES
                        else "marker_free_cycle_behavior"
                    ),
                    "deployment_valid": True,
                }
            )
    pd.DataFrame(manifest).to_csv(output / "feature_manifest.csv", index=False)
    contract = {
        "workload_family": "yolo26",
        "profiler_mode": PROFILER_MODE,
        "devices": data.groupby("device_id").workload_id.nunique().to_dict(),
        "deployment_inputs": [
            "multiscale marker-free detected profiling period",
            "marker-free GPU metric cycle behavior",
            "detector quality fields",
        ],
        "label_only": [
            "profiler steady-window runtime",
            "median of three unprofiled dry-run steady-window runtimes",
        ],
        "excluded": [
            "iteration markers",
            "callbacks",
            "batch size",
            "image size",
            "AMP flag",
            "dry-run runtime at inference",
        ],
        "target": "log(profile runtime / unprofiled dry runtime)",
        "prediction_constraint": "overhead ratio >= 1.0",
        "outer_split": "leave one YOLO workload out within each device",
        "inner_split": "five-fold grouped by workload",
        "primary_metric": "overhead absolute error in percentage points",
    }
    (output / "experiment_contract.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )
    write_report(output, predictions, metrics, selections)
    print(
        f"Wrote {len(predictions)} out-of-fold predictions, {len(metrics)} metrics, "
        f"and {len(selections)} production models to {output}."
    )


if __name__ == "__main__":
    main()
