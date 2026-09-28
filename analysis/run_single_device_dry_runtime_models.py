from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from common import DEFAULT_OUTPUT_DIR
from run_single_device_overhead_models import (
    ALPHAS,
    PROFILER_MODE,
    QUALITY_FEATURES,
    build_samples,
    feature_sets,
    make_pipeline,
)


def target_values(frame: pd.DataFrame) -> np.ndarray:
    return np.log(frame.actual_dry_runtime_ms.to_numpy(float))


def predict_dry_runtime(model, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    prediction = np.exp(model.predict(frame[features]))
    # Profiling is not expected to improve runtime. Keep the deployment output
    # physically consistent while retaining the one negative observed label as
    # measurement noise during training.
    return np.minimum(prediction, frame.detected_period_ms.to_numpy(float))


def dry_runtime_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(predicted - actual) / actual) * 100.0)


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
            actual_parts.extend(validation.actual_dry_runtime_ms.to_numpy(float))
            predicted_parts.extend(predict_dry_runtime(model, validation, features))
        score = dry_runtime_mape(
            np.asarray(actual_parts), np.asarray(predicted_parts)
        )
        trials.append((score, order, alpha))
    best = min(trials, key=lambda item: (item[0], item[1]))
    return float(best[2]), float(best[0])


def prediction_record(
    row: pd.Series,
    model_name: str,
    predicted_dry_ms: float,
    feature_count: int,
    alpha: float | None = None,
    inner_score: float | None = None,
    selected_family: str = "",
) -> dict:
    actual_dry = float(row.actual_dry_runtime_ms)
    detected_period = float(row.detected_period_ms)
    predicted_ratio = detected_period / predicted_dry_ms
    actual_ratio = float(row.actual_overhead_ratio)
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
        "predicted_dry_runtime_ms": predicted_dry_ms,
        "dry_runtime_error_ms": predicted_dry_ms - actual_dry,
        "dry_runtime_ape_percent": abs(predicted_dry_ms - actual_dry)
        / actual_dry
        * 100.0,
        "actual_overhead_ratio": actual_ratio,
        "predicted_overhead_ratio": predicted_ratio,
        "actual_overhead_percent": (actual_ratio - 1.0) * 100.0,
        "predicted_overhead_percent": (predicted_ratio - 1.0) * 100.0,
        "overhead_abs_error_pp": abs(predicted_ratio - actual_ratio) * 100.0,
        "selected_alpha": alpha,
        "inner_dry_runtime_mape_percent": inner_score,
    }


def outer_predictions(data: pd.DataFrame) -> list[dict]:
    specs = feature_sets(data)
    records = []
    for _, frame in data.groupby("device_id", sort=True):
        for workload in sorted(frame.workload_id.unique()):
            train = frame[frame.workload_id != workload]
            test = frame[frame.workload_id == workload]
            row = test.iloc[0]
            detected_period = float(row.detected_period_ms)

            records.append(
                prediction_record(
                    row,
                    "uncorrected_detected_period",
                    detected_period,
                    feature_count=1,
                )
            )

            ratio = float((train.actual_dry_runtime_ms / train.detected_period_ms).median())
            ratio_prediction = min(detected_period, detected_period * ratio)
            records.append(
                prediction_record(
                    row,
                    "constant_median_ratio_correction",
                    max(ratio_prediction, 1e-6),
                    feature_count=1,
                )
            )

            added_ms = float(
                (train.actual_profile_runtime_ms - train.actual_dry_runtime_ms).median()
            )
            additive_prediction = min(detected_period, detected_period - added_ms)
            records.append(
                prediction_record(
                    row,
                    "constant_median_added_ms_correction",
                    max(additive_prediction, 1e-6),
                    feature_count=1,
                )
            )

            family_trials = []
            for family_order, (name, features) in enumerate(specs.items()):
                alpha, inner_score = choose_alpha(train, features)
                model = make_pipeline(alpha)
                model.fit(train[features], target_values(train))
                predicted_dry = float(predict_dry_runtime(model, test, features)[0])
                records.append(
                    prediction_record(
                        row,
                        name,
                        predicted_dry,
                        feature_count=len(features),
                        alpha=alpha,
                        inner_score=inner_score,
                    )
                )
                family_trials.append(
                    (inner_score, family_order, name, features, alpha, predicted_dry)
                )

            selected = min(family_trials, key=lambda item: (item[0], item[1]))
            inner_score, _, name, features, alpha, predicted_dry = selected
            records.append(
                prediction_record(
                    row,
                    "nested_selected_ridge",
                    predicted_dry,
                    feature_count=len(features),
                    alpha=alpha,
                    inner_score=inner_score,
                    selected_family=name,
                )
            )
    return records


def metric_record(device: str, model: str, group: pd.DataFrame) -> dict:
    actual = group.actual_dry_runtime_ms.to_numpy(float)
    predicted = group.predicted_dry_runtime_ms.to_numpy(float)
    errors = predicted - actual
    ape = np.abs(errors) / actual * 100.0
    overhead_error = group.overhead_abs_error_pp.to_numpy(float)
    return {
        "device_id": device,
        "profiler_mode": PROFILER_MODE,
        "model": model,
        "n": len(group),
        "dry_runtime_mae_ms": float(np.mean(np.abs(errors))),
        "dry_runtime_rmse_ms": float(np.sqrt(np.mean(errors**2))),
        "dry_runtime_mape_percent": float(np.mean(ape)),
        "dry_runtime_median_ape_percent": float(np.median(ape)),
        "dry_runtime_p90_ape_percent": float(np.quantile(ape, 0.9)),
        "dry_runtime_max_ape_percent": float(np.max(ape)),
        "dry_runtime_bias_ms": float(np.mean(errors)),
        "dry_runtime_r2": float(r2_score(actual, predicted)),
        "derived_overhead_mae_pp": float(np.mean(overhead_error)),
        "derived_overhead_median_ae_pp": float(np.median(overhead_error)),
        "derived_overhead_max_ae_pp": float(np.max(overhead_error)),
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
        ["device_id", "dry_runtime_mape_percent", "model"]
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
        raw_required_fields = list(
            dict.fromkeys(["detected_period_ms"] + model_raw_fields)
        )
        model_path = model_dir / f"{device}_{PROFILER_MODE}_dry_runtime_ridge.joblib"
        example_path = example_dir / f"{device}_{PROFILER_MODE}_example.csv"
        joblib.dump(
            {
                "pipeline": model,
                "features": features,
                "raw_required_fields": raw_required_fields,
                "device_id": device,
                "profiler_mode": PROFILER_MODE,
                "workload_family": "yolo26",
                "predicts": "log unprofiled dry-run iteration runtime in milliseconds",
                "prediction_constraint": "0 < predicted dry runtime <= detected profiling period",
                "predicted_overhead_formula": "detected_period_ms / predicted_dry_runtime_ms - 1",
                "validated_scope": "unseen YOLO26 workload on this known device",
            },
            model_path,
        )
        frame.iloc[[0]][
            ["device_id", "workload_id"] + raw_required_fields
        ].to_csv(example_path, index=False)
        selections.append(
            {
                "device_id": device,
                "profiler_mode": PROFILER_MODE,
                "selected_feature_family": name,
                "selected_alpha": alpha,
                "grouped_cv_dry_runtime_mape_percent": score,
                "training_workloads": frame.workload_id.nunique(),
                "feature_count": len(features),
                "model_path": str(model_path.relative_to(output)),
                "example_input_path": str(example_path.relative_to(output)),
            }
        )
    return pd.DataFrame(selections)


def build_target_comparison(root: Path, metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    direct = metrics[
        (metrics.device_id != "both_devices_macro")
        & (metrics.model == "nested_selected_ridge")
    ]
    for row in direct.itertuples():
        rows.append(
            {
                "device_id": row.device_id,
                "training_target": "direct_log_dry_runtime",
                "primary_nested_model": "nested_selected_ridge",
                "dry_runtime_mape_percent": row.dry_runtime_mape_percent,
                "overhead_mae_pp": row.derived_overhead_mae_pp,
            }
        )
    prior_path = (
        root
        / "model_experiments"
        / "yolo_single_device_overhead"
        / "metrics.csv"
    )
    if prior_path.exists():
        prior = pd.read_csv(prior_path)
        prior = prior[
            (prior.device_id != "both_devices_macro")
            & (prior.model == "nested_selected_ridge")
        ]
        for row in prior.itertuples():
            rows.append(
                {
                    "device_id": row.device_id,
                    "training_target": "direct_log_overhead_ratio",
                    "primary_nested_model": "nested_selected_ridge",
                    "dry_runtime_mape_percent": row.estimated_dry_runtime_mape_percent,
                    "overhead_mae_pp": row.overhead_mae_pp,
                }
            )
    return pd.DataFrame(rows).sort_values(["device_id", "training_target"])


def write_report(
    output: Path,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    selections: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    lines = [
        "# YOLO single-device direct dry-runtime prediction",
        "",
        "This experiment directly predicts the unprofiled dry-run iteration runtime from marker-free Nsight GPU Metrics profiling features on the same known device. Derived overhead is calculated from the detected profiling period and predicted dry runtime.",
        "",
        "Outer evaluation leaves one YOLO workload out within each device. Ridge alpha and the primary nested feature family are selected using only inner grouped folds.",
        "",
        "## Outer leave-one-workload-out results",
        "",
        "| Device | Model | N | Dry MAPE | Median APE | P90 APE | Max APE | R2 | Derived overhead MAE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics[metrics.device_id != "both_devices_macro"].itertuples():
        lines.append(
            f"| {row.device_id} | {row.model} | {row.n} | {row.dry_runtime_mape_percent:.2f}% | {row.dry_runtime_median_ape_percent:.2f}% | {row.dry_runtime_p90_ape_percent:.2f}% | {row.dry_runtime_max_ape_percent:.2f}% | {row.dry_runtime_r2:.4f} | {row.derived_overhead_mae_pp:.2f} pp |"
        )
    lines += [
        "",
        "## Production fits",
        "",
        "| Device | Selected family | Alpha | Grouped-CV dry MAPE | Features |",
        "|---|---|---:|---:|---:|",
    ]
    for row in selections.itertuples():
        lines.append(
            f"| {row.device_id} | {row.selected_feature_family} | {row.selected_alpha:g} | {row.grouped_cv_dry_runtime_mape_percent:.2f}% | {row.feature_count} |"
        )
    if not comparison.empty:
        lines += [
            "",
            "## Nested target comparison",
            "",
            "| Device | Training target | Dry-runtime MAPE | Overhead MAE |",
            "|---|---|---:|---:|",
        ]
        for row in comparison.itertuples():
            lines.append(
                f"| {row.device_id} | {row.training_target} | {row.dry_runtime_mape_percent:.2f}% | {row.overhead_mae_pp:.2f} pp |"
            )
    primary = predictions[predictions.model == "nested_selected_ridge"].nlargest(
        10, "dry_runtime_ape_percent"
    )
    lines += [
        "",
        "## Largest primary out-of-fold errors",
        "",
        "| Device | Workload | Actual dry ms | Predicted dry ms | APE | Actual overhead | Predicted overhead |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in primary.itertuples():
        lines.append(
            f"| {row.device_id} | {row.workload_id} | {row.actual_dry_runtime_ms:.3f} | {row.predicted_dry_runtime_ms:.3f} | {row.dry_runtime_ape_percent:.2f}% | {row.actual_overhead_percent:.2f}% | {row.predicted_overhead_percent:.2f}% |"
        )
    lines += [
        "",
        "## Interpretation and limits",
        "",
        "- The primary target is dry runtime, not overhead ratio. This tests whether the cleaner main runtime signal is easier to learn than a noisy ratio residual.",
        "- Derived overhead uses the marker-free detected period. Its error therefore includes both dry-runtime prediction error and the gap between detected period and profiler summary runtime.",
        "- The validated scope is a held-out YOLO configuration on RTX4090 or RTX5090, not an unseen workload family, profiler mode, or GPU.",
        "- Each profiling label currently comes from one run; repeated profiler collections are still needed to separate model error from run-to-run noise.",
        "- Workload metadata, markers, callbacks, profiler summary runtime, and dry runtime are excluded from deployment inputs.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train direct per-device YOLO dry-runtime models from profiling features."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    output = root / "model_experiments" / "yolo_single_device_dry_runtime"
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
    comparison = build_target_comparison(root, metrics)
    comparison.to_csv(output / "target_comparison.csv", index=False)

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
            "median of three unprofiled dry-run steady-window runtimes"
        ],
        "audit_only": ["profiler steady-window runtime", "observed overhead ratio"],
        "excluded": [
            "iteration markers",
            "callbacks",
            "batch size",
            "image size",
            "AMP flag",
            "dry-run runtime at inference",
            "profiler summary runtime at inference",
        ],
        "target": "log unprofiled dry-run iteration runtime in milliseconds",
        "prediction_constraint": "0 < predicted dry runtime <= detected profiling period",
        "outer_split": "leave one YOLO workload out within each device",
        "inner_split": "five-fold grouped by workload",
        "primary_metric": "dry-runtime MAPE",
    }
    (output / "experiment_contract.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )
    write_report(output, predictions, metrics, selections, comparison)
    print(
        f"Wrote {len(predictions)} out-of-fold predictions, {len(metrics)} metrics, "
        f"and {len(selections)} production models to {output}."
    )


if __name__ == "__main__":
    main()
