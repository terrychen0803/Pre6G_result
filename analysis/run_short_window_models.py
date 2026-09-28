from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import DEFAULT_OUTPUT_DIR
from run_baseline_models import metric_row


STATIC_FEATURES = ["log_batch", "log_pixels", "amp_numeric"]
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


def prepare(short: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    metadata = ground_truth[
        ["device_id", "workload_id", "batch", "imgsz", "amp", "ground_truth_iter_ms"]
    ]
    device_rows = short.merge(
        metadata, on=["device_id", "workload_id"], validate="many_to_one"
    )
    rows = []
    for (workload, window), group in device_rows.groupby(
        ["workload_id", "window_iterations"], sort=True
    ):
        records = list(group.to_dict("records"))
        for source in records:
            for target in records:
                if source["device_id"] == target["device_id"]:
                    continue
                row = source.copy()
                row["source_device_id"] = source["device_id"]
                row["target_device_id"] = target["device_id"]
                row["target_ground_truth_iter_ms"] = target["ground_truth_iter_ms"]
                rows.append(row)
    result = pd.DataFrame(rows)
    result["log_batch"] = np.log(result.batch.astype(float))
    result["log_pixels"] = np.log(result.imgsz.astype(float) ** 2)
    result["amp_numeric"] = result.amp.astype(str).str.lower().map(
        {"true": 1.0, "false": 0.0}
    )
    result["log_profile_window_span_per_iter_ms"] = np.log(
        result.profile_window_span_per_iter_ms.astype(float)
    )
    iteration_cv = [column for column in result if column.endswith("_iteration_cv")]
    iteration_trend = [column for column in result if column.endswith("_iteration_trend")]
    phase_amplitude = [column for column in result if column.endswith("_phase_amplitude")]
    phase_peak = [column for column in result if column.endswith("_phase_peak")]
    phase_repeatability = [column for column in result if column.endswith("_phase_repeatability")]
    result["cycle_iteration_cv_mean"] = result[iteration_cv].mean(axis=1)
    result["cycle_iteration_trend_abs_mean"] = result[iteration_trend].abs().mean(axis=1)
    result["cycle_phase_amplitude_mean"] = result[phase_amplitude].mean(axis=1)
    result["cycle_phase_peak_std"] = result[phase_peak].std(axis=1, ddof=0)
    result["cycle_phase_repeatability_mean"] = result[phase_repeatability].mean(axis=1)
    return result.sort_values(
        ["source_device_id", "target_device_id", "window_iterations", "workload_id"]
    )


def feature_sets(data: pd.DataFrame) -> dict[str, list[str]]:
    metric_columns = [column for column in data if column.startswith("metric_")]
    means = sorted(
        column
        for column in metric_columns
        if column.endswith("_mean") and "_iteration_" not in column
    )
    distributions = sorted(
        column
        for column in metric_columns
        if "_iteration_" not in column and "_phase_" not in column
    )
    temporal = sorted(
        column
        for column in metric_columns
        if "_iteration_" in column or "_phase_" in column
    )
    compact_cycle = [
        "cycle_iteration_cv_mean",
        "cycle_iteration_trend_abs_mean",
        "cycle_phase_amplitude_mean",
        "cycle_phase_peak_std",
        "cycle_phase_repeatability_mean",
    ]
    duration = ["log_profile_window_span_per_iter_ms"]
    return {
        "behavior_mean_ridge": STATIC_FEATURES + means,
        "behavior_distribution_ridge": STATIC_FEATURES + distributions,
        "behavior_cycle_ridge": STATIC_FEATURES + means + temporal,
        "behavior_compact_cycle_ridge": STATIC_FEATURES + means + compact_cycle,
        "duration_only_ridge": STATIC_FEATURES + duration,
        "behavior_mean_duration_ridge": STATIC_FEATURES + means + duration,
    }


def make_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("variance", VarianceThreshold()),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def target_values(frame: pd.DataFrame) -> np.ndarray:
    return np.log(frame.target_ground_truth_iter_ms.to_numpy(float))


def choose_alpha(train: pd.DataFrame, features: list[str], seed: int) -> tuple[float, float]:
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    scores = []
    for order, alpha in enumerate(ALPHAS):
        ape = []
        for train_index, validation_index in splitter.split(train):
            inner_train = train.iloc[train_index]
            validation = train.iloc[validation_index]
            pipeline = make_pipeline(alpha)
            pipeline.fit(inner_train[features], target_values(inner_train))
            predicted = np.exp(pipeline.predict(validation[features]))
            actual = validation.target_ground_truth_iter_ms.to_numpy(float)
            ape.extend(np.abs(predicted - actual) / actual * 100)
        scores.append((float(np.mean(ape)), order, alpha))
    best = min(scores, key=lambda item: (item[0], item[1]))
    return best[2], best[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate causal short-prefix profiling windows."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    output = root / "model_experiments" / "part6_short_window"
    output.mkdir(parents=True, exist_ok=True)

    data = prepare(
        pd.read_csv(root / "short_window_profile_features.csv"),
        pd.read_csv(root / "ground_truth.csv"),
    )
    specs = feature_sets(data)
    predictions = []
    for (source, target, window), group in data.groupby(
        ["source_device_id", "target_device_id", "window_iterations"], sort=True
    ):
        direction = f"{source}_to_{target}"
        for workload in sorted(group.workload_id.unique()):
            train = group[group.workload_id != workload]
            test = group[group.workload_id == workload]
            for model, features in specs.items():
                seed = 1000 + int(window) * 10 + sum(ord(char) for char in workload + model)
                alpha, inner_mape = choose_alpha(train, features, seed)
                pipeline = make_pipeline(alpha)
                pipeline.fit(train[features], target_values(train))
                predicted = float(np.exp(pipeline.predict(test[features])[0]))
                actual = float(test.iloc[0].target_ground_truth_iter_ms)
                predictions.append(
                    {
                        "direction": direction,
                        "source_device_id": source,
                        "target_device_id": target,
                        "window_iterations": int(window),
                        "workload_id": workload,
                        "model": model,
                        "input_feature_count": len(features),
                        "duration_feature_used": "duration" in model,
                        "actual_ms": actual,
                        "predicted_ms": predicted,
                        "error_ms": predicted - actual,
                        "absolute_error_ms": abs(predicted - actual),
                        "ape_percent": abs(predicted - actual) / actual * 100,
                        "selected_parameters": json.dumps({"model__alpha": alpha}),
                        "inner_selection_mape_percent": inner_mape,
                    }
                )
    prediction_frame = pd.DataFrame(predictions)
    metric_rows = []
    for (direction, window, model), group in prediction_frame.groupby(
        ["direction", "window_iterations", "model"], sort=True
    ):
        row = metric_row(direction, model, group)
        row["window_iterations"] = window
        metric_rows.append(row)
    for (window, model), group in prediction_frame.groupby(
        ["window_iterations", "model"], sort=True
    ):
        row = metric_row("both_directions_combined", model, group)
        row["window_iterations"] = window
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["direction", "window_iterations", "mape_percent", "model"]
    )
    prediction_frame.to_csv(output / "predictions.csv", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)

    manifest_rows = []
    for model, features in specs.items():
        for feature in features:
            family = (
                "static"
                if feature in STATIC_FEATURES
                else "duration"
                if feature == "log_profile_window_span_per_iter_ms"
                else "cycle"
                if feature.startswith("cycle_") or "_iteration_" in feature or "_phase_" in feature
                else "behavior_distribution"
                if feature.endswith("_cv") or feature.endswith("_p90")
                else "behavior_mean"
            )
            manifest_rows.append({"model": model, "feature": feature, "family": family})
    pd.DataFrame(manifest_rows).to_csv(output / "feature_manifest.csv", index=False)

    lines = [
        "# Part 6: causal short-window profiling experiment",
        "",
        "Each K-iteration feature row uses only Nsight samples within TRAIN_ITER_0021 through TRAIN_ITER_(0020+K). No later iteration contributes to that row.",
        "",
        "Behavior-only models exclude GPU sample count, NVTX duration, timestamps, window span, source ground-truth runtime, and every target runtime field. Duration-assisted models are reported separately.",
        "",
        "Outer evaluation is leave-one-workload-out. Ridge alpha is selected inside the 23-workload outer training set using shuffled 5-fold validation; scaling and variance filtering are inner-fold local.",
        "",
        "| Direction | K | Model | Inputs | MAE ms | MAPE | Median APE | Max APE | R2 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    input_counts = prediction_frame.groupby("model").input_feature_count.first()
    for row in metrics.itertuples():
        lines.append(
            f"| {row.direction} | {row.window_iterations} | {row.model} | {input_counts[row.model]} | {row.mae_ms:.3f} | {row.mape_percent:.2f}% | {row.median_ape_percent:.2f}% | {row.max_ape_percent:.2f}% | {row.r2:.4f} |"
        )

    lines += ["", "## Best strict behavior-only model by window", "", "| Direction | K | Model | MAPE |", "|---|---:|---|---:|"]
    strict = metrics[
        (metrics.direction != "both_directions_combined")
        & (~metrics.model.str.contains("duration"))
    ]
    for (direction, window), group in strict.groupby(
        ["direction", "window_iterations"], sort=True
    ):
        best = group.sort_values("mape_percent").iloc[0]
        lines.append(f"| {direction} | {window} | {best.model} | {best.mape_percent:.2f}% |")

    lines += ["", "## Hyperparameter selection frequency", ""]
    for (direction, window, model), group in prediction_frame.groupby(
        ["direction", "window_iterations", "model"], sort=True
    ):
        counts = Counter(group.selected_parameters)
        formatted = "; ".join(f"`{key}`: {value}" for key, value in counts.most_common())
        lines.append(f"- {direction} / K={window} / {model}: {formatted}")
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        f"Wrote {len(prediction_frame)} predictions and {len(metrics)} metric rows to {output}."
    )


if __name__ == "__main__":
    main()
