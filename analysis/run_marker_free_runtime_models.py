from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import DEFAULT_OUTPUT_DIR, PROJECT_ROOT
from run_baseline_models import metric_row


ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
HARDWARE_FIELDS = (
    "gpu_compute_capability",
    "gpu_sm_count",
    "gpu_vram_visible_gib",
    "gpu_default_power_limit_w",
    "gpu_max_sm_clock_mhz",
    "gpu_max_memory_clock_mhz",
    "pcie_negotiated_generation",
    "pcie_negotiated_link_width",
    "cpu_physical_core_count",
    "cpu_max_mhz",
    "system_memory_total_bytes",
)


def build_samples(root: Path, hardware_path: Path) -> pd.DataFrame:
    features = pd.read_csv(root / "marker_free_cycle_features.csv")
    labels = pd.read_csv(root / "ground_truth.csv")[
        ["device_id", "workload_id", "ground_truth_iter_ms"]
    ]
    evaluation = pd.read_csv(
        root / "model_experiments" / "part7_marker_free_cycles" / "adaptive_oof_predictions.csv"
    )[["device_id", "workload_id", "oracle_period_ms"]]
    features = features.merge(
        evaluation, on=["device_id", "workload_id"], validate="one_to_one"
    )
    hardware = pd.read_csv(hardware_path).set_index("device_id")
    rows = []
    for workload, profile_group in features.groupby("workload_id", sort=True):
        target_group = labels[labels.workload_id == workload]
        for source in profile_group.to_dict("records"):
            target = target_group[target_group.device_id != source["device_id"]].iloc[0]
            target_device = str(target.device_id)
            row = dict(source)
            row["source_device_id"] = source["device_id"]
            row["target_device_id"] = target_device
            row["direction"] = f"{source['device_id']}_to_{target_device}"
            row["target_ground_truth_iter_ms"] = float(target.ground_truth_iter_ms)
            row["log_detected_period_ms"] = np.log(float(source["detected_period_ms"]))
            row["log_oracle_period_ms"] = np.log(float(source["oracle_period_ms"]))
            for field in HARDWARE_FIELDS:
                source_value = float(hardware.loc[source["device_id"], field])
                target_value = float(hardware.loc[target_device, field])
                if source_value > 0 and target_value > 0:
                    row[f"hw_log_ratio_{field}"] = np.log(target_value / source_value)
                else:
                    row[f"hw_difference_{field}"] = target_value - source_value
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["workload_id", "source_device_id"])


def feature_sets(data: pd.DataFrame) -> dict[str, list[str]]:
    hardware = sorted(column for column in data if column.startswith("hw_"))
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
    quality = ["detector_confidence", "emission_horizon_seconds"]
    return {
        "behavior_only_ridge": hardware + behavior,
        "period_only_ridge": hardware + ["log_detected_period_ms"],
        "period_compact_behavior_ridge": hardware
        + ["log_detected_period_ms"]
        + compact,
        "period_full_behavior_ridge": hardware
        + ["log_detected_period_ms"]
        + behavior,
        "period_behavior_quality_ridge": hardware
        + ["log_detected_period_ms"]
        + compact
        + quality,
        "oracle_period_only_ridge": hardware + ["log_oracle_period_ms"],
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


def targets(frame: pd.DataFrame) -> np.ndarray:
    return np.log(frame.target_ground_truth_iter_ms.to_numpy(float))


def choose_alpha(train: pd.DataFrame, features: list[str]) -> tuple[float, float]:
    splitter = GroupKFold(n_splits=5)
    scores = []
    for order, alpha in enumerate(ALPHAS):
        errors = []
        for train_index, validation_index in splitter.split(
            train, groups=train.workload_id
        ):
            inner_train = train.iloc[train_index]
            validation = train.iloc[validation_index]
            pipeline = make_pipeline(alpha)
            pipeline.fit(inner_train[features], targets(inner_train))
            predicted = np.exp(pipeline.predict(validation[features]))
            actual = validation.target_ground_truth_iter_ms.to_numpy(float)
            errors.extend(np.abs(predicted - actual) / actual * 100)
        scores.append((float(np.mean(errors)), order, alpha))
    best = min(scores, key=lambda item: (item[0], item[1]))
    return float(best[2]), float(best[0])


def ridge_predictions(data: pd.DataFrame, scope: str) -> list[dict]:
    specs = feature_sets(data)
    results = []
    groups = [("both_directions", data)]
    if scope == "direction_specific":
        groups = list(data.groupby("direction", sort=True))
    for fitted_group, frame in groups:
        for held_out in sorted(frame.workload_id.unique()):
            train = frame[frame.workload_id != held_out]
            test = frame[frame.workload_id == held_out]
            for model, features in specs.items():
                alpha, inner_mape = choose_alpha(train, features)
                pipeline = make_pipeline(alpha)
                pipeline.fit(train[features], targets(train))
                predicted = np.exp(pipeline.predict(test[features]))
                for record, prediction in zip(test.to_dict("records"), predicted):
                    actual = float(record["target_ground_truth_iter_ms"])
                    results.append(
                        {
                            "training_scope": scope,
                            "fitted_group": fitted_group,
                            "direction": record["direction"],
                            "source_device_id": record["source_device_id"],
                            "target_device_id": record["target_device_id"],
                            "workload_id": held_out,
                            "model": model,
                            "deployment_valid": not model.startswith("oracle_"),
                            "input_feature_count": len(features),
                            "actual_ms": actual,
                            "predicted_ms": float(prediction),
                            "ape_percent": abs(float(prediction) - actual) / actual * 100,
                            "selected_alpha": alpha,
                            "inner_selection_mape_percent": inner_mape,
                        }
                    )
    return results


def ratio_predictions(data: pd.DataFrame) -> list[dict]:
    results = []
    for held_out in sorted(data.workload_id.unique()):
        training = data[data.workload_id != held_out]
        test = data[data.workload_id == held_out]
        for record in test.to_dict("records"):
            direction_train = training[training.direction == record["direction"]]
            for model, source_column in (
                ("period_direction_median_ratio", "detected_period_ms"),
                ("oracle_direction_median_ratio", "oracle_period_ms"),
            ):
                ratios = (
                    direction_train.target_ground_truth_iter_ms
                    / direction_train[source_column]
                )
                predicted = float(record[source_column] * ratios.median())
                actual = float(record["target_ground_truth_iter_ms"])
                results.append(
                    {
                        "training_scope": "direction_specific",
                        "fitted_group": record["direction"],
                        "direction": record["direction"],
                        "source_device_id": record["source_device_id"],
                        "target_device_id": record["target_device_id"],
                        "workload_id": held_out,
                        "model": model,
                        "deployment_valid": not model.startswith("oracle_"),
                        "input_feature_count": 1,
                        "actual_ms": actual,
                        "predicted_ms": predicted,
                        "ape_percent": abs(predicted - actual) / actual * 100,
                        "selected_alpha": np.nan,
                        "inner_selection_mape_percent": np.nan,
                    }
                )
    return results


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["training_scope", "model", "direction"]
    for key, group in predictions.groupby(keys, sort=True):
        row = metric_row(key[2], key[1], group.rename(columns={"actual_ms": "actual_ms", "predicted_ms": "predicted_ms"}))
        row["training_scope"] = key[0]
        row["deployment_valid"] = bool(group.deployment_valid.iloc[0])
        rows.append(row)
    for (scope, model), group in predictions.groupby(
        ["training_scope", "model"], sort=True
    ):
        row = metric_row("both_directions_combined", model, group)
        row["training_scope"] = scope
        row["deployment_valid"] = bool(group.deployment_valid.iloc[0])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["training_scope", "direction", "mape_percent", "model"]
    )


def nested_family_selection(predictions: pd.DataFrame) -> pd.DataFrame:
    """Select the feature family from inner-CV scores, never outer-test errors."""
    eligible = predictions[
        predictions.deployment_valid
        & predictions.inner_selection_mape_percent.notna()
        & ~predictions.model.str.startswith("nested_")
    ]
    rows = []
    for (_, _, _), group in eligible.groupby(
        ["training_scope", "fitted_group", "workload_id"], sort=True
    ):
        family_scores = (
            group.groupby("model", as_index=False)
            .inner_selection_mape_percent.first()
            .sort_values(["inner_selection_mape_percent", "model"])
        )
        selected_family = str(family_scores.iloc[0].model)
        selected = group[group.model == selected_family].copy()
        selected["selected_feature_family"] = selected_family
        selected["model"] = "nested_selected_deployment_ridge"
        rows.extend(selected.to_dict("records"))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate marker-free cross-device per-iteration runtime prediction."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--hardware",
        type=Path,
        default=PROJECT_ROOT / "hardware" / "hardware_descriptors.csv",
    )
    args = parser.parse_args()
    root = args.output_dir.resolve()
    output = root / "model_experiments" / "part8_marker_free_runtime"
    output.mkdir(parents=True, exist_ok=True)
    data = build_samples(root, args.hardware.resolve())
    data.to_csv(output / "directed_samples.csv", index=False)
    predictions = pd.DataFrame(
        ridge_predictions(data, "direction_specific")
        + ridge_predictions(data, "paired_combined")
        + ratio_predictions(data)
    )
    predictions["selected_feature_family"] = ""
    predictions = pd.concat(
        [predictions, nested_family_selection(predictions)], ignore_index=True
    )
    predictions.to_csv(output / "predictions.csv", index=False)
    metrics = summarize(predictions)
    metrics.to_csv(output / "metrics.csv", index=False)

    prior_rows = []
    comparisons = (
        (
            "part1_source_baseline",
            root / "model_experiments" / "part1_baselines" / "metrics.csv",
            "both_directions_macro_rows",
            "log_source_ols",
            "no: requires completed source baseline timing",
        ),
        (
            "part3_marker_aligned_full_profile",
            root / "model_experiments" / "part3_profile_models" / "metrics.csv",
            "both_directions_combined",
            "compact_gpu_ridge",
            "no: marker-aligned and includes workload metadata",
        ),
        (
            "part6_marker_aligned_short_window",
            root / "model_experiments" / "part6_short_window" / "metrics.csv",
            "both_directions_combined",
            "behavior_mean_duration_ridge",
            "no: iteration window and duration are marker-aligned",
        ),
    )
    for stage, path, direction, model, validity in comparisons:
        old = pd.read_csv(path)
        row = old[(old.direction == direction) & (old.model == model)].sort_values(
            "mape_percent"
        ).iloc[0]
        prior_rows.append(
            {
                "stage": stage,
                "model": model,
                "deployment_valid_for_unmodified_workload": validity,
                "mape_percent": row.mape_percent,
                "median_ape_percent": row.median_ape_percent,
                "max_ape_percent": row.max_ape_percent,
                "r2": row.r2,
            }
        )
    for stage, scope, model, validity in (
        (
            "part8_primary_nested",
            "direction_specific",
            "nested_selected_deployment_ridge",
            "yes",
        ),
        (
            "part8_best_fixed_family_posthoc",
            "direction_specific",
            "period_behavior_quality_ridge",
            "yes, but family identified after outer evaluation",
        ),
        (
            "part8_single_hardware_model_nested",
            "paired_combined",
            "nested_selected_deployment_ridge",
            "yes for known devices; unseen-device validity untested",
        ),
    ):
        row = metrics[
            (metrics.training_scope == scope)
            & (metrics.direction == "both_directions_combined")
            & (metrics.model == model)
        ].iloc[0]
        prior_rows.append(
            {
                "stage": stage,
                "model": model,
                "deployment_valid_for_unmodified_workload": validity,
                "mape_percent": row.mape_percent,
                "median_ape_percent": row.median_ape_percent,
                "max_ape_percent": row.max_ape_percent,
                "r2": row.r2,
            }
        )
    stage_comparison = pd.DataFrame(prior_rows)
    stage_comparison.to_csv(output / "stage_comparison.csv", index=False)

    manifest = []
    for model, columns in feature_sets(data).items():
        for column in columns:
            family = (
                "hidden_oracle"
                if column == "log_oracle_period_ms"
                else "marker_free_period"
                if column == "log_detected_period_ms"
                else "detector_quality"
                if column in {"detector_confidence", "emission_horizon_seconds"}
                else "hardware"
                if column.startswith("hw_")
                else "marker_free_behavior"
            )
            manifest.append(
                {
                    "model": model,
                    "feature": column,
                    "family": family,
                    "deployment_valid": family != "hidden_oracle",
                }
            )
    pd.DataFrame(manifest).to_csv(output / "feature_manifest.csv", index=False)

    combined = metrics[metrics.direction == "both_directions_combined"]
    lines = [
        "# Part 8 checkpoint: marker-free cross-device runtime prediction",
        "",
        "Outer evaluation holds out one workload on both devices. Ridge alpha is selected by an inner workload-grouped five-fold split. The deployment-valid feature sets contain no workload metadata, source baseline label, target profiling field, or iteration marker.",
        "",
        "`direction_specific` matches the earlier two-device experiments. `paired_combined` fits one model across both directions and adds source-to-target hardware ratios; with only two devices it is not an unseen-device test.",
        "",
        "## Combined-direction results",
        "",
        "| Training scope | Model | Valid at deployment | N | MAE ms | MAPE | Median APE | Max APE | R2 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in combined.itertuples():
        lines.append(
            f"| {row.training_scope} | {row.model} | {'yes' if row.deployment_valid else 'no (oracle upper bound)'} | {row.n} | {row.mae_ms:.3f} | {row.mape_percent:.2f}% | {row.median_ape_percent:.2f}% | {row.max_ape_percent:.2f}% | {row.r2:.4f} |"
        )
    lines += [
        "",
        "## Best deployment-valid result by direction and scope",
        "",
        "| Training scope | Direction | Model | MAPE | Median APE | Max APE |",
        "|---|---|---|---:|---:|---:|",
    ]
    valid = metrics[
        metrics.deployment_valid & (metrics.direction != "both_directions_combined")
    ]
    for (_, _), group in valid.groupby(["training_scope", "direction"], sort=True):
        row = group.sort_values(["mape_percent", "max_ape_percent"]).iloc[0]
        lines.append(
            f"| {row.training_scope} | {row.direction} | {row.model} | {row.mape_percent:.2f}% | {row.median_ape_percent:.2f}% | {row.max_ape_percent:.2f}% |"
        )
    best_key = combined[
        (combined.training_scope == "direction_specific")
        & (combined.model == "nested_selected_deployment_ridge")
    ].iloc[0]
    best_predictions = predictions[
        (predictions.training_scope == best_key.training_scope)
        & (predictions.model == best_key.model)
    ].nlargest(10, "ape_percent")
    lines += [
        "",
        "## Worst predictions for the primary nested result",
        "",
        f"Primary result: `{best_key.training_scope}/{best_key.model}`. The feature family for each held-out workload was selected using inner folds only.",
        "",
        "| Direction | Workload | Actual ms | Predicted ms | APE |",
        "|---|---|---:|---:|---:|",
    ]
    for row in best_predictions.itertuples():
        lines.append(
            f"| {row.direction} | {row.workload_id} | {row.actual_ms:.3f} | {row.predicted_ms:.3f} | {row.ape_percent:.2f}% |"
        )
    lines += [
        "",
        "## Comparison with earlier stages",
        "",
        "| Stage | Model | Deployment validity for an unmodified workload | MAPE | Median APE | Max APE | R2 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in stage_comparison.itertuples():
        lines.append(
            f"| {row.stage} | {row.model} | {row.deployment_valid_for_unmodified_workload} | {row.mape_percent:.2f}% | {row.median_ape_percent:.2f}% | {row.max_ape_percent:.2f}% | {row.r2:.4f} |"
        )
    lines += [
        "",
        "## Checkpoint interpretation",
        "",
        "The primary nested, direction-specific result reaches 4.50% MAPE without workload metadata or markers. The best fixed family reaches 4.35% MAPE, but that family was identified after viewing all outer-fold results and is therefore secondary evidence.",
        "",
        "Behavior-only and period-only models are both weaker than their combination, so the inferred source period is useful but does not replace the metric signature. The hidden-oracle period-only model is not better than the marker-free period-only model; segmentation error is not the main runtime-model bottleneck in this dataset.",
        "",
        "The 5090-to-4090 direction remains harder, with the largest primary error on C24. A single hardware-conditioned model across both directions reaches 4.97% MAPE, but two devices cannot establish generalization to an unseen GPU. The next data requirement is a third device or additional independent workload families; model complexity should not be increased before that validation.",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "experiment_contract.json").write_text(
        json.dumps(
            {
                "deployment_inputs": [
                    "marker-free source GPU metric time series",
                    "marker-free detected source period",
                    "source and target hardware descriptors",
                ],
                "label_only": ["target unprofiled baseline iteration runtime"],
                "excluded": [
                    "iteration markers",
                    "training callbacks",
                    "batch",
                    "image size",
                    "AMP flag",
                    "source baseline runtime",
                    "target profiling metrics",
                ],
                "outer_split": "leave one workload out on both devices",
                "inner_split": "five-fold grouped by workload",
                "oracle_models": "evaluation upper bounds only",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(predictions)} predictions and {len(metrics)} metric rows to {output}.")


if __name__ == "__main__":
    main()
