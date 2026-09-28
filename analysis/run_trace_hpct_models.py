from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import DEFAULT_OUTPUT_DIR
from run_baseline_models import metric_row
from run_profile_models import COMPACT_GPU_FEATURES, STATIC_FEATURES


TRACE_SOURCE_COLUMNS = [
    "source_kernel_count",
    "source_total_kernel_time_ms",
    "source_kernel_launch_interval_mean_us",
    "source_gpu_idle_gap_total_ms",
    "source_gpu_idle_gap_mean_us",
    "source_memcpy_count",
    "source_memcpy_bytes",
    "source_memcpy_time_ms",
    "source_memcpy_effective_bandwidth_gbps",
    "source_memset_count",
    "source_memset_bytes",
    "source_memset_time_ms",
    "source_sync_count",
    "source_sync_time_ms",
    "source_cuda_api_count",
    "source_cuda_api_time_ms",
]

HPCT_SOURCE_COLUMNS = [
    "source_hpct_meta_db_bytes",
    "source_hpct_profile_db_bytes",
    "source_hpct_cct_db_bytes",
    "source_hpct_hpcstruct_gpu_begin_count",
    "source_hpct_hpcrun_missing_correlation_count",
    "source_hpct_accelerator_trace_records",
]

TRACE_FEATURES = [f"log_trace_{column.removeprefix('source_')}" for column in TRACE_SOURCE_COLUMNS]
HPCT_FEATURES = [f"log_hpct_{column.removeprefix('source_hpct_')}" for column in HPCT_SOURCE_COLUMNS]


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
    for source, feature in zip(TRACE_SOURCE_COLUMNS, TRACE_FEATURES):
        result[feature] = np.log1p(result[source].astype(float))
    for source, feature in zip(HPCT_SOURCE_COLUMNS, HPCT_FEATURES):
        result[feature] = np.log1p(result[source].astype(float))
    required = STATIC_FEATURES + COMPACT_GPU_FEATURES + TRACE_FEATURES + HPCT_FEATURES
    if result[required].isna().any().any():
        missing = result[required].isna().sum()
        raise ValueError(f"Required features contain missing values: {missing[missing > 0].to_dict()}")
    return result


def model_specs() -> dict[str, list[str]]:
    gpu = COMPACT_GPU_FEATURES
    return {
        "static_only_ridge": STATIC_FEATURES,
        "trace_only_ridge": TRACE_FEATURES,
        "static_trace_ridge": STATIC_FEATURES + TRACE_FEATURES,
        "static_compact_gpu_ridge": STATIC_FEATURES + gpu,
        "static_hpct_proxy_ridge": STATIC_FEATURES + HPCT_FEATURES,
        "static_trace_hpct_proxy_ridge": STATIC_FEATURES + TRACE_FEATURES + HPCT_FEATURES,
        "static_compact_gpu_hpct_proxy_ridge": STATIC_FEATURES + gpu + HPCT_FEATURES,
        "static_compact_gpu_trace_ridge": STATIC_FEATURES + gpu + TRACE_FEATURES,
        "full_profile_fusion_ridge": STATIC_FEATURES + gpu + TRACE_FEATURES + HPCT_FEATURES,
    }


def make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("variance", VarianceThreshold()),
            ("scale", StandardScaler()),
            ("model", Ridge()),
        ]
    )


def target_values(frame: pd.DataFrame) -> np.ndarray:
    return np.log(frame.target_ground_truth_iter_ms.to_numpy(float))


def choose_alpha(train: pd.DataFrame, features: list[str]) -> tuple[float, float]:
    scores = []
    for order, alpha in enumerate((0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)):
        ape = []
        for workload in sorted(train.workload_id.unique()):
            inner_train = train[train.workload_id != workload]
            inner_test = train[train.workload_id == workload]
            pipeline = make_pipeline()
            pipeline.set_params(model__alpha=alpha)
            pipeline.fit(inner_train[features], target_values(inner_train))
            predicted = float(np.exp(pipeline.predict(inner_test[features])[0]))
            actual = float(inner_test.iloc[0].target_ground_truth_iter_ms)
            ape.append(abs(predicted - actual) / actual * 100)
        scores.append((float(np.mean(ape)), order, alpha))
    best = min(scores, key=lambda item: (item[0], item[1]))
    return best[2], best[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Part 4: source Nsight trace and HPCToolkit proxy ablations."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    output = root / "model_experiments" / "part4_trace_hpct_models"
    output.mkdir(parents=True, exist_ok=True)

    data = prepare(pd.read_csv(root / "cross_device_samples.csv"))
    specs = model_specs()
    predictions, coefficients = [], []

    for (source_device, target_device), direction_data in data.groupby(
        ["source_device_id", "target_device_id"], sort=True
    ):
        direction = f"{source_device}_to_{target_device}"
        for workload in sorted(direction_data.workload_id.unique()):
            train = direction_data[direction_data.workload_id != workload]
            test = direction_data[direction_data.workload_id == workload].iloc[0]
            for name, features in specs.items():
                alpha, inner_mape = choose_alpha(train, features)
                pipeline = make_pipeline()
                pipeline.set_params(model__alpha=alpha)
                pipeline.fit(train[features], target_values(train))
                predicted = float(
                    np.exp(pipeline.predict(test.to_frame().T[features])[0])
                )
                actual = float(test.target_ground_truth_iter_ms)
                parameter_text = json.dumps({"model__alpha": alpha}, sort_keys=True)
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
                        "selected_parameters": parameter_text,
                        "inner_selection_mape_percent": inner_mape,
                        "input_feature_count": len(features),
                    }
                )
                selected = np.asarray(features)[
                    pipeline.named_steps["variance"].get_support()
                ]
                for feature, coefficient in zip(
                    selected, pipeline.named_steps["model"].coef_
                ):
                    coefficients.append(
                        {
                            "direction": direction,
                            "held_out_workload": workload,
                            "model": name,
                            "feature": feature,
                            "standardized_coefficient": float(coefficient),
                            "selected_parameters": parameter_text,
                        }
                    )

    prediction_frame = pd.DataFrame(predictions)
    coefficient_frame = pd.DataFrame(coefficients)
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
    coefficient_frame.to_csv(output / "coefficients.csv", index=False)
    metric_frame.to_csv(output / "metrics.csv", index=False)

    feature_rows = []
    for model, features in specs.items():
        for feature in features:
            family = (
                "static"
                if feature in STATIC_FEATURES
                else "trace"
                if feature in TRACE_FEATURES
                else "hpct_proxy"
                if feature in HPCT_FEATURES
                else "profile_duration"
                if feature == "log_profile_sample_count"
                else "gpu_metric"
            )
            feature_rows.append({"model": model, "feature": feature, "family": family})
    pd.DataFrame(feature_rows).to_csv(output / "feature_manifest.csv", index=False)

    part3 = pd.read_csv(root / "model_experiments" / "part3_profile_models" / "metrics.csv")
    lines = [
        "# Part 4: Nsight trace and HPCToolkit proxy ablations",
        "",
        "All model inputs are observable on the source device during profiling. The target-device unprofiled baseline runtime is label-only, and `source_ground_truth_iter_ms` is excluded.",
        "",
        "Nsight trace inputs contain kernel, idle-gap, memory-operation, synchronization, and CUDA API counts/times. Current HPCToolkit inputs are artifact-structure proxies (database sizes, structure/correlation counts, and accelerator trace records), not extracted numeric CCT performance metrics.",
        "",
        "Outer evaluation is leave-one-workload-out. Scaling, variance filtering, and Ridge alpha selection occur only inside each outer training fold.",
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
        "## Comparison with Part 3",
        "",
        "| Direction | Part 3 best | Part 3 MAPE | Part 4 best | Part 4 MAPE | Relative change |",
        "|---|---|---:|---|---:|---:|",
    ]
    for direction in sorted(prediction_frame.direction.unique()):
        old = part3[part3.direction == direction].sort_values("mape_percent").iloc[0]
        new = metric_frame[metric_frame.direction == direction].sort_values("mape_percent").iloc[0]
        change = (new.mape_percent / old.mape_percent - 1) * 100
        lines.append(
            f"| {direction} | {old.model} | {old.mape_percent:.2f}% | {new.model} | {new.mape_percent:.2f}% | {change:+.1f}% |"
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

    lines += ["", "## Worst workloads for the best Part 4 model", ""]
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
        f"and {len(coefficient_frame)} coefficient rows to {output}."
    )


if __name__ == "__main__":
    main()
