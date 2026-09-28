from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sqlite3
import statistics
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("PRE6G_DATA_ROOT", PROJECT_ROOT / "data" / "raw")).expanduser()
DEFAULT_DETECTIONS = (
    PROJECT_ROOT / "analysis" / "high_load_01" / "trace_event_period" / "adaptive_deployment.csv"
)
DEFAULT_CACHE_ROOT = (
    PROJECT_ROOT / "analysis" / "high_load_01" / "cache" / "nsys_trace_events" / "RTX5090"
)
DEFAULT_RUNS_ROOT = (
    DATA_ROOT
    / "high_load_01_RTX5090_20260923"
    / "runs_loaded"
    / "high_load_01"
    / "RTX5090"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis" / "high_load_01" / "trace_runtime_prediction"
RIDGE_ALPHAS = (0.1, 1.0, 10.0)
FEATURE_NAMES = (
    "log_detected_period_ms",
    "log_kernel_event_rate_hz",
    "log_unique_kernel_count",
    "log_median_kernel_duration_us",
    "log_mean_kernel_duration_us",
    "log_kernel_busy_fraction",
    "anchor_robust_cv",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    fraction = position - low
    return float(ordered[low] * (1 - fraction) + ordered[high] * fraction)


def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [list(matrix[index]) + [vector[index]] for index in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        if abs(divisor) < 1e-12:
            raise RuntimeError("singular ridge system")
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                augmented[row][index] - factor * augmented[column][index]
                for index in range(size + 1)
            ]
    return [augmented[index][-1] for index in range(size)]


def ridge_predict(
    train_features: list[list[float]],
    train_targets: list[float],
    test_features: list[float],
    alpha: float,
) -> float:
    feature_count = len(test_features)
    means = [
        statistics.mean(row[column] for row in train_features)
        for column in range(feature_count)
    ]
    scales = []
    for column in range(feature_count):
        values = [row[column] for row in train_features]
        scale = statistics.stdev(values) if len(values) > 1 else 1.0
        scales.append(scale if scale > 1e-12 else 1.0)
    design = [
        [1.0]
        + [
            (row[column] - means[column]) / scales[column]
            for column in range(feature_count)
        ]
        for row in train_features
    ]
    width = feature_count + 1
    gram = [
        [
            sum(row[left] * row[right] for row in design)
            for right in range(width)
        ]
        for left in range(width)
    ]
    projected = [
        sum(row[column] * target for row, target in zip(design, train_targets))
        for column in range(width)
    ]
    for column in range(1, width):
        gram[column][column] += alpha
    coefficients = solve(gram, projected)
    transformed = [1.0] + [
        (test_features[column] - means[column]) / scales[column]
        for column in range(feature_count)
    ]
    return sum(value * coefficient for value, coefficient in zip(transformed, coefficients))


def choose_alpha(features: list[list[float]], targets: list[float]) -> tuple[float, list[dict]]:
    trials = []
    for alpha in RIDGE_ALPHAS:
        errors = []
        for held_out in range(len(features)):
            prediction = math.exp(
                ridge_predict(
                    [row for index, row in enumerate(features) if index != held_out],
                    [value for index, value in enumerate(targets) if index != held_out],
                    features[held_out],
                    alpha,
                )
            )
            target = math.exp(targets[held_out])
            errors.append(abs(prediction - target) / target * 100.0)
        trials.append(
            {
                "alpha": alpha,
                "inner_mean_ape_percent": statistics.mean(errors),
                "inner_median_ape_percent": statistics.median(errors),
            }
        )
    winner = min(
        trials,
        key=lambda row: (
            row["inner_mean_ape_percent"],
            row["inner_median_ape_percent"],
            row["alpha"],
        ),
    )
    return float(winner["alpha"]), trials


def load_sample(detection: dict[str, str], cache_root: Path, runs_root: Path) -> dict:
    workload_id = detection["workload_id"]
    database = cache_root / workload_id / "nsys_trace" / "profile.sqlite"
    capture_start = int(detection["capture_start_ns"])
    capture_end = int(detection["capture_end_ns"])
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    try:
        kernels = connection.execute(
            "SELECT end-start,shortName FROM CUPTI_ACTIVITY_KIND_KERNEL "
            "WHERE start BETWEEN ? AND ?",
            (capture_start, capture_end),
        ).fetchall()
    finally:
        connection.close()
    durations_us = [float(duration) / 1000.0 for duration, _ in kernels]
    horizon_seconds = float(detection["horizon_seconds"])
    event_rate = len(kernels) / horizon_seconds
    unique_kernel_count = len({short_name for _, short_name in kernels})
    busy_fraction = sum(durations_us) / 1000.0 / (horizon_seconds * 1000.0)
    features = [
        math.log(float(detection["detected_period_ms"])),
        math.log(event_rate),
        math.log(unique_kernel_count),
        math.log(statistics.median(durations_us)),
        math.log(statistics.mean(durations_us)),
        math.log(max(busy_fraction, 1e-12)),
        float(detection["anchor_robust_cv"]),
    ]
    baseline_values = []
    for path in sorted((runs_root / workload_id).glob("baseline_*/summary.json")):
        baseline_values.append(
            float(json.loads(path.read_text(encoding="utf-8"))["steady_window_mean_iter_ms"])
        )
    if len(baseline_values) != 5:
        raise RuntimeError(f"{workload_id}: expected five baseline repeats")
    trace_summary = json.loads(
        (runs_root / workload_id / "nsys_trace_01" / "summary.json").read_text(encoding="utf-8")
    )
    baseline = statistics.median(baseline_values)
    trace_runtime = float(trace_summary["steady_window_mean_iter_ms"])
    row = {
        "device_id": "RTX5090",
        "workload_id": workload_id,
        "emission_horizon_seconds": horizon_seconds,
        "baseline_median_ms": baseline,
        "baseline_repeat_mean_ms": statistics.mean(baseline_values),
        "baseline_repeat_cv_percent": statistics.stdev(baseline_values) / statistics.mean(baseline_values) * 100.0,
        "trace_runtime_ms": trace_runtime,
        "observed_overhead_ms": trace_runtime - baseline,
        "observed_overhead_percent": (trace_runtime - baseline) / baseline * 100.0,
        "kernel_event_count": len(kernels),
        "kernel_event_rate_hz": event_rate,
        "unique_kernel_count": unique_kernel_count,
        "median_kernel_duration_us": statistics.median(durations_us),
        "mean_kernel_duration_us": statistics.mean(durations_us),
        "kernel_busy_fraction": busy_fraction,
        "detected_period_ms": float(detection["detected_period_ms"]),
        "anchor_robust_cv": float(detection["anchor_robust_cv"]),
    }
    row.update(dict(zip(FEATURE_NAMES, features)))
    row["_features"] = features
    return row


def summarize_predictions(rows: list[dict]) -> dict:
    absolute_errors = [abs(row["prediction_error_ms"]) for row in rows]
    percentage_errors = [row["baseline_ape_percent"] for row in rows]
    overhead_pp_errors = [abs(row["overhead_error_pp"]) for row in rows]
    return {
        "profiles": len(rows),
        "baseline_mae_ms": statistics.mean(absolute_errors),
        "baseline_rmse_ms": math.sqrt(statistics.mean([value * value for value in absolute_errors])),
        "baseline_mape_percent": statistics.mean(percentage_errors),
        "baseline_median_ape_percent": statistics.median(percentage_errors),
        "baseline_p90_ape_percent": percentile(percentage_errors, 0.90),
        "baseline_max_ape_percent": max(percentage_errors),
        "baseline_within_10_percent": sum(value <= 10 for value in percentage_errors) / len(percentage_errors),
        "overhead_mae_percentage_points": statistics.mean(overhead_pp_errors),
        "overhead_p90_ae_percentage_points": percentile(overhead_pp_errors, 0.90),
        "mean_baseline_repeat_cv_percent": statistics.mean(row["baseline_repeat_cv_percent"] for row in rows),
        "median_baseline_repeat_cv_percent": statistics.median(row["baseline_repeat_cv_percent"] for row in rows),
        "max_baseline_repeat_cv_percent": max(row["baseline_repeat_cv_percent"] for row in rows),
        "observed_overhead_median_percent": statistics.median(row["observed_overhead_percent"] for row in rows),
        "observed_overhead_min_percent": min(row["observed_overhead_percent"] for row in rows),
        "observed_overhead_max_percent": max(row["observed_overhead_percent"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace-only high-load runtime and overhead experiment.")
    parser.add_argument("--detections", type=Path, default=DEFAULT_DETECTIONS)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    detections = read_csv(args.detections)
    samples = [load_sample(row, args.cache_root, args.runs_root) for row in detections]
    predictions = []
    tuning_rows = []
    for held_out, sample in enumerate(samples):
        train = [row for index, row in enumerate(samples) if index != held_out]
        train_features = [row["_features"] for row in train]
        train_targets = [math.log(row["baseline_median_ms"]) for row in train]
        alpha, trials = choose_alpha(train_features, train_targets)
        for trial in trials:
            tuning_rows.append(
                {"held_out_workload_id": sample["workload_id"], **trial}
            )
        predicted_baseline = math.exp(
            ridge_predict(train_features, train_targets, sample["_features"], alpha)
        )
        predicted_overhead_ms = sample["trace_runtime_ms"] - predicted_baseline
        predicted_overhead_percent = predicted_overhead_ms / predicted_baseline * 100.0
        output = {key: value for key, value in sample.items() if key != "_features"}
        output.update(
            {
                "selected_alpha": alpha,
                "predicted_baseline_ms": predicted_baseline,
                "prediction_error_ms": predicted_baseline - sample["baseline_median_ms"],
                "baseline_ape_percent": abs(predicted_baseline - sample["baseline_median_ms"])
                / sample["baseline_median_ms"]
                * 100.0,
                "predicted_overhead_ms": predicted_overhead_ms,
                "predicted_overhead_percent": predicted_overhead_percent,
                "overhead_error_pp": predicted_overhead_percent - sample["observed_overhead_percent"],
            }
        )
        predictions.append(output)

    metrics = summarize_predictions(predictions)
    direct_errors = [
        abs(row["detected_period_ms"] - row["baseline_median_ms"])
        / row["baseline_median_ms"]
        * 100.0
        for row in predictions
    ]
    metrics["direct_period_baseline_mape_percent"] = statistics.mean(direct_errors)
    metrics["evaluation"] = "nested leave-one-workload-out"
    metrics["target"] = "median steady_window_mean_iter_ms across five high-load unprofiled baselines"

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "samples.csv", [{key: value for key, value in row.items() if key != "_features"} for row in samples])
    write_csv(output / "predictions.csv", predictions)
    write_csv(output / "inner_alpha_selection.csv", tuning_rows)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    contract = {
        "program": Path(__file__).name,
        "program_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "evaluation": "nested leave-one-workload-out",
        "features": FEATURE_NAMES,
        "forbidden_features": [
            "NVTX",
            "iterations.csv",
            "workload ID or configuration",
            "batch size",
            "image size",
            "AMP",
            "GPU metrics",
            "Netdata during-run metrics",
        ],
        "target": metrics["target"],
        "ridge_alphas_selected_in_inner_loocv": RIDGE_ALPHAS,
    }
    (output / "experiment_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    report = [
        "# Trace-only high-load runtime / overhead prediction",
        "",
        "All features are computed from target-process CUDA kernel events available by the marker-blind emission horizon. "
        "NVTX and workload configuration are not model inputs.",
        "",
        f"- Evaluation: {metrics['evaluation']}",
        f"- Profiles: {metrics['profiles']}",
        f"- Baseline MAPE / median / P90 / max APE: {metrics['baseline_mape_percent']:.2f}% / "
        f"{metrics['baseline_median_ape_percent']:.2f}% / {metrics['baseline_p90_ape_percent']:.2f}% / "
        f"{metrics['baseline_max_ape_percent']:.2f}%",
        f"- Baseline within 10%: {metrics['baseline_within_10_percent']:.1%}",
        f"- Baseline MAE / RMSE: {metrics['baseline_mae_ms']:.2f} / {metrics['baseline_rmse_ms']:.2f} ms",
        f"- Overhead MAE / P90 absolute error: {metrics['overhead_mae_percentage_points']:.2f} / "
        f"{metrics['overhead_p90_ae_percentage_points']:.2f} percentage points",
        f"- Direct detected-period baseline MAPE: {metrics['direct_period_baseline_mape_percent']:.2f}%",
        "",
        "## Label stability",
        "",
        f"The five unprofiled baseline repeats have mean / median / max CV of "
        f"{metrics['mean_baseline_repeat_cv_percent']:.2f}% / {metrics['median_baseline_repeat_cv_percent']:.2f}% / "
        f"{metrics['max_baseline_repeat_cv_percent']:.2f}%.",
        f"Observed trace overhead ranges from {metrics['observed_overhead_min_percent']:.2f}% to "
        f"{metrics['observed_overhead_max_percent']:.2f}% (median {metrics['observed_overhead_median_percent']:.2f}%).",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
