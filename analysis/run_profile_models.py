from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import DEFAULT_OUTPUT_DIR
from run_baseline_models import metric_row


STATIC_FEATURES = ["log_batch", "log_pixels", "amp_numeric"]
COMPACT_GPU_FEATURES = [
    "log_profile_sample_count",
    "source_gpu_gpc_clock_frequency_mhz_mean",
    "source_gpu_sys_clock_frequency_mhz_mean",
    "source_gpu_sm_issue_throughput_mean",
    "source_gpu_sms_active_throughput_mean",
    "source_gpu_tensor_active_throughput_mean",
    "source_gpu_gr_active_throughput_mean",
    "source_gpu_dram_read_bandwidth_throughput_mean",
    "source_gpu_dram_write_bandwidth_throughput_mean",
    "source_gpu_compute_warps_in_flight_throughput_mean",
    "source_gpu_compute_warps_in_flight_avg_mean",
    "source_gpu_compute_warps_in_flight_avg_warps_per_cycle_mean",
]


def prepare(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["log_batch"] = np.log(result.batch.astype(float))
    result["log_pixels"] = np.log(result.imgsz.astype(float) ** 2)
    result["amp_numeric"] = result.amp.astype(str).str.lower().map(
        {"true": 1.0, "false": 0.0}
    )
    result["log_profile_sample_count"] = np.log(
        result.source_gpu_sample_count.astype(float)
    )
    if result[STATIC_FEATURES + ["log_profile_sample_count"]].isna().any().any():
        raise ValueError("Required deployment-time features contain missing values.")
    return result


def common_gpu_summary_features(data: pd.DataFrame) -> list[str]:
    suffixes = ("_mean", "_cv", "_p90")
    candidates = [
        column
        for column in data.columns
        if column.startswith("source_gpu_")
        and column.endswith(suffixes)
        and not column.startswith("source_gpu_idle_")
    ]
    complete = [column for column in candidates if data[column].notna().all()]
    if not complete:
        raise ValueError("No common, complete GPU metric summary features found.")
    return sorted(complete)


def model_specs(summary_features: list[str]) -> dict[str, dict]:
    alpha_grid = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    pca_grid = [
        {"pca__n_components": components, "model__alpha": alpha}
        for components in (2, 4, 8, 12, 16)
        for alpha in (0.1, 1.0, 10.0, 100.0)
    ]
    profile_features = ["log_profile_sample_count"] + summary_features
    combined_features = STATIC_FEATURES + profile_features
    return {
        "static_only_ridge": {
            "features": STATIC_FEATURES,
            "use_pca": False,
            "grid": [{"model__alpha": alpha} for alpha in alpha_grid],
        },
        "profile_duration_proxy_ridge": {
            "features": STATIC_FEATURES + ["log_profile_sample_count"],
            "use_pca": False,
            "grid": [{"model__alpha": alpha} for alpha in alpha_grid],
        },
        "compact_gpu_ridge": {
            "features": STATIC_FEATURES + COMPACT_GPU_FEATURES,
            "use_pca": False,
            "grid": [{"model__alpha": alpha} for alpha in alpha_grid],
        },
        "gpu_summary_pca_ridge": {
            "features": profile_features,
            "use_pca": True,
            "grid": pca_grid,
        },
        "static_gpu_summary_pca_ridge": {
            "features": combined_features,
            "use_pca": True,
            "grid": pca_grid,
        },
        "static_gpu_summary_ridge": {
            "features": combined_features,
            "use_pca": False,
            "grid": [{"model__alpha": alpha} for alpha in alpha_grid],
        },
    }


def make_pipeline(use_pca: bool) -> Pipeline:
    steps = [
        ("variance", VarianceThreshold()),
        ("scale", StandardScaler()),
    ]
    if use_pca:
        steps.append(("pca", PCA()))
    steps.append(("model", Ridge()))
    return Pipeline(steps)


def target_values(frame: pd.DataFrame) -> np.ndarray:
    return np.log(frame.target_ground_truth_iter_ms.to_numpy(float))


def choose_parameters(train: pd.DataFrame, spec: dict) -> tuple[dict, float]:
    scores = []
    for order, parameters in enumerate(spec["grid"]):
        ape = []
        for workload in sorted(train.workload_id.unique()):
            inner_train = train[train.workload_id != workload]
            inner_test = train[train.workload_id == workload]
            pipeline = make_pipeline(spec["use_pca"])
            pipeline.set_params(**parameters)
            pipeline.fit(inner_train[spec["features"]], target_values(inner_train))
            predicted = float(
                np.exp(pipeline.predict(inner_test[spec["features"]])[0])
            )
            actual = float(inner_test.iloc[0].target_ground_truth_iter_ms)
            ape.append(abs(predicted - actual) / actual * 100)
        scores.append((float(np.mean(ape)), order, parameters))
    best = min(scores, key=lambda item: (item[0], item[1]))
    return best[2], best[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Part 3: deployment-aligned source Nsight GPU profiling models."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    output = root / "model_experiments" / "part3_profile_models"
    output.mkdir(parents=True, exist_ok=True)

    data = prepare(pd.read_csv(root / "cross_device_samples.csv"))
    summary_features = common_gpu_summary_features(data)
    missing_compact = [column for column in COMPACT_GPU_FEATURES if column not in data]
    if missing_compact:
        raise ValueError(f"Missing compact GPU features: {missing_compact}")
    specs = model_specs(summary_features)
    predictions = []

    for (source_device, target_device), direction_data in data.groupby(
        ["source_device_id", "target_device_id"], sort=True
    ):
        direction = f"{source_device}_to_{target_device}"
        for workload in sorted(direction_data.workload_id.unique()):
            train = direction_data[direction_data.workload_id != workload]
            test = direction_data[direction_data.workload_id == workload].iloc[0]
            for name, spec in specs.items():
                parameters, inner_mape = choose_parameters(train, spec)
                pipeline = make_pipeline(spec["use_pca"])
                pipeline.set_params(**parameters)
                pipeline.fit(train[spec["features"]], target_values(train))
                predicted = float(
                    np.exp(pipeline.predict(test.to_frame().T[spec["features"]])[0])
                )
                actual = float(test.target_ground_truth_iter_ms)
                predictions.append(
                    {
                        "direction": direction,
                        "source_device_id": source_device,
                        "target_device_id": target_device,
                        "workload_id": workload,
                        "model": name,
                        "actual_ms": actual,
                        "predicted_ms": predicted,
                        "error_ms": predicted - actual,
                        "absolute_error_ms": abs(predicted - actual),
                        "ape_percent": abs(predicted - actual) / actual * 100,
                        "selected_parameters": json.dumps(parameters, sort_keys=True),
                        "inner_selection_mape_percent": inner_mape,
                        "input_feature_count": len(spec["features"]),
                    }
                )

    prediction_frame = pd.DataFrame(predictions)
    metrics = []
    for (direction, model), group in prediction_frame.groupby(
        ["direction", "model"], sort=True
    ):
        metrics.append(metric_row(direction, model, group))
    for model, group in prediction_frame.groupby("model", sort=True):
        metrics.append(metric_row("both_directions_combined", model, group))
    metric_frame = pd.DataFrame(metrics).sort_values(
        ["direction", "mape_percent", "model"]
    )
    prediction_frame.to_csv(output / "predictions.csv", index=False)
    metric_frame.to_csv(output / "metrics.csv", index=False)

    feature_rows = []
    for model, spec in specs.items():
        for feature in spec["features"]:
            feature_rows.append(
                {
                    "model": model,
                    "feature": feature,
                    "family": "static"
                    if feature in STATIC_FEATURES
                    else "profile_duration"
                    if feature == "log_profile_sample_count"
                    else "gpu_metric",
                }
            )
    pd.DataFrame(feature_rows).to_csv(output / "feature_manifest.csv", index=False)

    part1 = pd.read_csv(root / "model_experiments" / "part1_baselines" / "metrics.csv")
    part2 = pd.read_csv(root / "model_experiments" / "part2_workload_models" / "metrics.csv")
    lines = [
        "# Part 3: deployment-aligned Nsight GPU profiling models",
        "",
        "The target-device unprofiled baseline runtime is used only as the label. Model inputs contain only workload fields and source-device Nsight GPU profiling fields available before prediction. `source_ground_truth_iter_ms` is excluded from every Part 3 model.",
        "",
        f"Common complete GPU summary inputs: {len(summary_features)} columns (mean, CV, and P90 summaries).",
        "",
        "Outer evaluation is leave-one-workload-out. Variance filtering, scaling, PCA, and hyperparameter selection are fitted only inside each outer training fold.",
        "",
        "| Direction | Model | Inputs | MAE ms | RMSE ms | MAPE | Median APE | Max APE | R2 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    input_counts = prediction_frame.groupby("model").input_feature_count.first()
    for row in metric_frame.itertuples():
        lines.append(
            f"| {row.direction} | {row.model} | {input_counts[row.model]} | {row.mae_ms:.3f} | {row.rmse_ms:.3f} | {row.mape_percent:.2f}% | {row.median_ape_percent:.2f}% | {row.max_ape_percent:.2f}% | {row.r2:.4f} |"
        )

    lines += [
        "",
        "## Comparison with earlier idealized baselines",
        "",
        "Part 1 and Part 2 use the source device's unprofiled ground-truth runtime, which is a stronger input than the short-profiling deployment scenario. They are reported as reference upper baselines, not deployment-equivalent competitors.",
        "",
        "| Direction | Part 1 best | Part 1 MAPE | Part 2 best | Part 2 MAPE | Part 3 best | Part 3 MAPE |",
        "|---|---|---:|---|---:|---|---:|",
    ]
    for direction in sorted(prediction_frame.direction.unique()):
        old1 = part1[part1.direction == direction].sort_values("mape_percent").iloc[0]
        old2 = part2[part2.direction == direction].sort_values("mape_percent").iloc[0]
        new = metric_frame[metric_frame.direction == direction].sort_values("mape_percent").iloc[0]
        lines.append(
            f"| {direction} | {old1.model} | {old1.mape_percent:.2f}% | {old2.model} | {old2.mape_percent:.2f}% | {new.model} | {new.mape_percent:.2f}% |"
        )

    lines += ["", "## Hyperparameter selection frequency", ""]
    for (direction, model), group in prediction_frame.groupby(
        ["direction", "model"], sort=True
    ):
        counts = Counter(group.selected_parameters)
        formatted = "; ".join(
            f"`{key}`: {value}" for key, value in counts.most_common()
        )
        lines.append(f"- {direction} / {model}: {formatted}")

    lines += ["", "## Worst workloads for the best Part 3 model", ""]
    for direction in sorted(prediction_frame.direction.unique()):
        best = (
            metric_frame[metric_frame.direction == direction]
            .sort_values("mape_percent")
            .iloc[0]
            .model
        )
        worst = prediction_frame[
            (prediction_frame.direction == direction)
            & (prediction_frame.model == best)
        ].nlargest(5, "ape_percent")
        lines += [
            f"### {direction}: `{best}`",
            "",
            "| Workload | Actual ms | Predicted ms | APE |",
            "|---|---:|---:|---:|",
        ]
        for row in worst.itertuples():
            lines.append(
                f"| {row.workload_id} | {row.actual_ms:.3f} | {row.predicted_ms:.3f} | {row.ape_percent:.2f}% |"
            )
        lines.append("")

    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        f"Wrote {len(prediction_frame)} predictions, {len(metric_frame)} metric rows, "
        f"and {len(summary_features)} common GPU summary features to {output}."
    )


if __name__ == "__main__":
    main()
