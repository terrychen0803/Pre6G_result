"""PRIMARY SCENARIO — Stage 2: runtime prediction from marker-free trace features.

This is the canonical clean/high-load runtime-model evaluation for the main
research scenario. It consumes Stage 1 adaptive detections and trace-derived
features, then writes samples, predictions, metrics, tuning records, and a report.

See PRIMARY_SCENARIO.md for the authoritative two-stage workflow.
"""

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

from run_trace_runtime_model import (
    FEATURE_NAMES,
    RIDGE_ALPHAS,
    percentile,
    ridge_predict,
    write_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("PRE6G_DATA_ROOT", PROJECT_ROOT / "data" / "raw")).expanduser()
DEFAULT_CLEAN_DETECTIONS = (
    PROJECT_ROOT / "analysis" / "clean_rtx5090" / "trace_event_period" / "adaptive_deployment.csv"
)
DEFAULT_HIGH_DETECTIONS = (
    PROJECT_ROOT / "analysis" / "high_load_01" / "trace_event_period" / "adaptive_deployment.csv"
)
DEFAULT_CLEAN_CACHE = PROJECT_ROOT / "analysis_cache" / "nsys" / "RTX5090"
DEFAULT_HIGH_CACHE = (
    PROJECT_ROOT / "analysis" / "high_load_01" / "cache" / "nsys_trace_events" / "RTX5090"
)
DEFAULT_CLEAN_RUNS = DATA_ROOT / "RTX5090_yolo26_collection_20260817" / "runs" / "RTX5090"
DEFAULT_HIGH_RUNS = (
    DATA_ROOT
    / "high_load_01_RTX5090_20260923"
    / "runs_loaded"
    / "high_load_01"
    / "RTX5090"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "analysis" / "unified_trace_model"
PRE_WINDOW_SECONDS = 5.0
SOFT_GATE_TEMPERATURES = (0.25, 0.5, 1.0, 2.0)
TELEMETRY_COLUMNS = (
    "GPU Util%",
    "GPU Mem Used(MB)",
    "GPU Power(W)",
    "CPU User%",
    "CPU System%",
    "Load 1min",
)
TELEMETRY_FEATURES = tuple(
    "pre_" + name.lower().replace("%", "pct").replace(" ", "_").replace("(", "").replace(")", "")
    for name in TELEMETRY_COLUMNS
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def finite(value: float) -> bool:
    return math.isfinite(value)


def parse_float(value: str | None) -> float:
    try:
        parsed = float(value) if value not in (None, "") else math.nan
        return parsed if finite(parsed) else math.nan
    except (TypeError, ValueError):
        return math.nan


def pre_run_telemetry(run_dir: Path) -> dict[str, float]:
    netdata_path = run_dir / "netdata.csv"
    iterations_path = run_dir / "iterations.csv"
    if not netdata_path.exists() or not iterations_path.exists():
        return {feature: math.nan for feature in TELEMETRY_FEATURES}
    iterations = read_csv(iterations_path)
    first_start = min(int(row["start_unix_ns"]) for row in iterations)
    lower = first_start - int(PRE_WINDOW_SECONDS * 1e9)
    netdata = [
        row
        for row in read_csv(netdata_path)
        if row.get("TimestampUnixNs")
        and lower <= int(row["TimestampUnixNs"]) < first_start
    ]
    result = {}
    for column, feature in zip(TELEMETRY_COLUMNS, TELEMETRY_FEATURES):
        values = [parse_float(row.get(column)) for row in netdata]
        values = [value for value in values if finite(value)]
        result[feature] = statistics.mean(values) if values else math.nan
    result["pre_sample_count"] = float(len(netdata))
    return result


def load_sample(
    condition: str,
    detection: dict[str, str],
    cache_root: Path,
    runs_root: Path,
) -> dict:
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
    trace_features = [
        math.log(float(detection["detected_period_ms"])),
        math.log(event_rate),
        math.log(unique_kernel_count),
        math.log(statistics.median(durations_us)),
        math.log(statistics.mean(durations_us)),
        math.log(max(busy_fraction, 1e-12)),
        float(detection["anchor_robust_cv"]),
    ]
    baseline_values = [
        float(json.loads(path.read_text(encoding="utf-8"))["steady_window_mean_iter_ms"])
        for path in sorted((runs_root / workload_id).glob("baseline_*/summary.json"))
    ]
    expected_repeats = 3 if condition == "clean" else 5
    if len(baseline_values) != expected_repeats:
        raise RuntimeError(
            f"{condition}/{workload_id}: expected {expected_repeats} baselines, got {len(baseline_values)}"
        )
    trace_summary = json.loads(
        (runs_root / workload_id / "nsys_trace_01" / "summary.json").read_text(encoding="utf-8")
    )
    baseline = statistics.median(baseline_values)
    trace_runtime = float(trace_summary["steady_window_mean_iter_ms"])
    telemetry = pre_run_telemetry(runs_root / workload_id / "nsys_trace_01")
    row = {
        "condition": condition,
        "load_flag": 0.0 if condition == "clean" else 1.0,
        "device_id": "RTX5090",
        "workload_id": workload_id,
        "baseline_repeat_count": len(baseline_values),
        "baseline_median_ms": baseline,
        "baseline_repeat_cv_percent": statistics.stdev(baseline_values)
        / statistics.mean(baseline_values)
        * 100.0,
        "trace_runtime_ms": trace_runtime,
        "observed_overhead_percent": (trace_runtime - baseline) / baseline * 100.0,
        "emission_horizon_seconds": horizon_seconds,
        "detected_period_ms": float(detection["detected_period_ms"]),
        "kernel_event_rate_hz": event_rate,
        "unique_kernel_count": unique_kernel_count,
        "median_kernel_duration_us": statistics.median(durations_us),
        "mean_kernel_duration_us": statistics.mean(durations_us),
        "kernel_busy_fraction": busy_fraction,
        "anchor_robust_cv": float(detection["anchor_robust_cv"]),
        **telemetry,
    }
    row.update(dict(zip(FEATURE_NAMES, trace_features)))
    row["_trace_features"] = trace_features
    row["_telemetry_features"] = [telemetry[name] for name in TELEMETRY_FEATURES]
    return row


def raw_features(sample: dict, model: str) -> list[float]:
    values = list(sample["_trace_features"])
    if model == "trace_plus_load_flag":
        values.append(float(sample["load_flag"]))
    elif model == "trace_plus_load_interactions":
        flag = float(sample["load_flag"])
        values.extend([flag, *[flag * value for value in sample["_trace_features"]]])
    elif model == "trace_plus_pre_netdata":
        values.extend(sample["_telemetry_features"])
    return values


def impute(
    train: list[list[float]], test: list[float]
) -> tuple[list[list[float]], list[float]]:
    width = len(test)
    medians = []
    missing_columns = []
    for column in range(width):
        observed = [row[column] for row in train if finite(row[column])]
        medians.append(statistics.median(observed) if observed else 0.0)
        missing_columns.append(any(not finite(row[column]) for row in train))
    transformed_train = []
    for row in train:
        values = [
            row[column] if finite(row[column]) else medians[column]
            for column in range(width)
        ]
        values.extend(
            1.0 if not finite(row[column]) else 0.0
            for column in range(width)
            if missing_columns[column]
        )
        transformed_train.append(values)
    transformed_test = [
        test[column] if finite(test[column]) else medians[column]
        for column in range(width)
    ]
    transformed_test.extend(
        1.0 if not finite(test[column]) else 0.0
        for column in range(width)
        if missing_columns[column]
    )
    return transformed_train, transformed_test


def predict_one(train: list[dict], test: dict, model: str, alpha: float) -> float:
    train_raw = [raw_features(sample, model) for sample in train]
    test_raw = raw_features(test, model)
    train_features, test_features = impute(train_raw, test_raw)
    targets = [math.log(sample["baseline_median_ms"]) for sample in train]
    return math.exp(ridge_predict(train_features, targets, test_features, alpha))


def choose_alpha(train: list[dict], model: str) -> tuple[float, list[dict]]:
    workloads = sorted({sample["workload_id"] for sample in train})
    trials = []
    for alpha in RIDGE_ALPHAS:
        errors = []
        for workload_id in workloads:
            inner_train = [sample for sample in train if sample["workload_id"] != workload_id]
            validation = [sample for sample in train if sample["workload_id"] == workload_id]
            for sample in validation:
                prediction = predict_one(inner_train, sample, model, alpha)
                errors.append(
                    abs(prediction - sample["baseline_median_ms"])
                    / sample["baseline_median_ms"]
                    * 100.0
                )
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


def evaluate_scenario(
    samples: list[dict],
    experiment: str,
    model: str,
    train_condition: str | None,
    test_conditions: tuple[str, ...],
) -> tuple[list[dict], list[dict]]:
    predictions = []
    tuning = []
    for held_out in sorted({sample["workload_id"] for sample in samples}):
        train = [sample for sample in samples if sample["workload_id"] != held_out]
        if train_condition is not None:
            train = [sample for sample in train if sample["condition"] == train_condition]
        tests = [
            sample
            for sample in samples
            if sample["workload_id"] == held_out
            and sample["condition"] in test_conditions
        ]
        alpha, trials = choose_alpha(train, model)
        for trial in trials:
            tuning.append(
                {
                    "experiment": experiment,
                    "model": model,
                    "held_out_workload_id": held_out,
                    **trial,
                }
            )
        for test in tests:
            prediction = predict_one(train, test, model, alpha)
            predicted_overhead = (test["trace_runtime_ms"] - prediction) / prediction * 100.0
            predictions.append(
                {
                    "experiment": experiment,
                    "model": model,
                    "held_out_workload_id": held_out,
                    "test_condition": test["condition"],
                    "selected_alpha": alpha,
                    "actual_baseline_ms": test["baseline_median_ms"],
                    "predicted_baseline_ms": prediction,
                    "baseline_ape_percent": abs(prediction - test["baseline_median_ms"])
                    / test["baseline_median_ms"]
                    * 100.0,
                    "actual_overhead_percent": test["observed_overhead_percent"],
                    "predicted_overhead_percent": predicted_overhead,
                    "overhead_error_pp": predicted_overhead - test["observed_overhead_percent"],
                    "baseline_repeat_cv_percent": test["baseline_repeat_cv_percent"],
                    "detected_period_ms": test["detected_period_ms"],
                    "emission_horizon_seconds": test["emission_horizon_seconds"],
                }
            )
    return predictions, tuning


def infer_condition(train: list[dict], test: dict) -> str:
    """Infer the load regime from trace features only using fold-local centroids."""
    width = len(test["_trace_features"])
    means = [
        statistics.mean(sample["_trace_features"][column] for sample in train)
        for column in range(width)
    ]
    scales = []
    for column in range(width):
        values = [sample["_trace_features"][column] for sample in train]
        scale = statistics.stdev(values)
        scales.append(scale if scale > 1e-12 else 1.0)
    centroids = {}
    for condition in ("clean", "high_load_01"):
        condition_rows = [sample for sample in train if sample["condition"] == condition]
        centroids[condition] = [
            statistics.mean(
                (sample["_trace_features"][column] - means[column]) / scales[column]
                for sample in condition_rows
            )
            for column in range(width)
        ]
    point = [
        (test["_trace_features"][column] - means[column]) / scales[column]
        for column in range(width)
    ]
    return min(
        centroids,
        key=lambda condition: sum(
            (value - center) ** 2
            for value, center in zip(point, centroids[condition])
        ),
    )


def soft_regime_score(train: list[dict], test: dict, temperature: float) -> float:
    """Return a trace-only soft high-load score from distances to regime centroids."""
    width = len(test["_trace_features"])
    means = [
        statistics.mean(sample["_trace_features"][column] for sample in train)
        for column in range(width)
    ]
    scales = []
    for column in range(width):
        values = [sample["_trace_features"][column] for sample in train]
        scale = statistics.stdev(values)
        scales.append(scale if scale > 1e-12 else 1.0)
    point = [
        (test["_trace_features"][column] - means[column]) / scales[column]
        for column in range(width)
    ]
    distances = {}
    for condition in ("clean", "high_load_01"):
        condition_rows = [sample for sample in train if sample["condition"] == condition]
        centroid = [
            statistics.mean(
                (sample["_trace_features"][column] - means[column]) / scales[column]
                for sample in condition_rows
            )
            for column in range(width)
        ]
        distances[condition] = sum(
            (value - center) ** 2 for value, center in zip(point, centroid)
        )
    logit = (distances["clean"] - distances["high_load_01"]) / temperature
    logit = max(-50.0, min(50.0, logit))
    return 1.0 / (1.0 + math.exp(-logit))


def evaluate_regime_moe(
    samples: list[dict], model: str, use_oracle_condition: bool
) -> tuple[list[dict], list[dict]]:
    predictions = []
    tuning = []
    for held_out in sorted({sample["workload_id"] for sample in samples}):
        outer_train = [sample for sample in samples if sample["workload_id"] != held_out]
        tests = [sample for sample in samples if sample["workload_id"] == held_out]
        expert_settings = {}
        for condition in ("clean", "high_load_01"):
            expert_train = [
                sample for sample in outer_train if sample["condition"] == condition
            ]
            alpha, trials = choose_alpha(expert_train, "trace_only")
            expert_settings[condition] = (expert_train, alpha)
            for trial in trials:
                tuning.append(
                    {
                        "experiment": "mixed_grouped_loow",
                        "model": model,
                        "held_out_workload_id": held_out,
                        "expert_condition": condition,
                        **trial,
                    }
                )
        for test in tests:
            inferred = infer_condition(outer_train, test)
            selected_condition = test["condition"] if use_oracle_condition else inferred
            expert_train, alpha = expert_settings[selected_condition]
            prediction = predict_one(expert_train, test, "trace_only", alpha)
            predicted_overhead = (test["trace_runtime_ms"] - prediction) / prediction * 100.0
            predictions.append(
                {
                    "experiment": "mixed_grouped_loow",
                    "model": model,
                    "held_out_workload_id": held_out,
                    "test_condition": test["condition"],
                    "inferred_condition": inferred,
                    "condition_classification_correct": inferred == test["condition"],
                    "selected_alpha": alpha,
                    "actual_baseline_ms": test["baseline_median_ms"],
                    "predicted_baseline_ms": prediction,
                    "baseline_ape_percent": abs(prediction - test["baseline_median_ms"])
                    / test["baseline_median_ms"]
                    * 100.0,
                    "actual_overhead_percent": test["observed_overhead_percent"],
                    "predicted_overhead_percent": predicted_overhead,
                    "overhead_error_pp": predicted_overhead - test["observed_overhead_percent"],
                    "baseline_repeat_cv_percent": test["baseline_repeat_cv_percent"],
                    "detected_period_ms": test["detected_period_ms"],
                    "emission_horizon_seconds": test["emission_horizon_seconds"],
                }
            )
    return predictions, tuning


def evaluate_soft_gate(samples: list[dict]) -> tuple[list[dict], list[dict]]:
    model = "trace_soft_regime_interactions"
    predictions = []
    tuning = []
    workloads = sorted({sample["workload_id"] for sample in samples})
    for held_out in workloads:
        outer_train = [sample for sample in samples if sample["workload_id"] != held_out]
        tests = [sample for sample in samples if sample["workload_id"] == held_out]
        alpha, alpha_trials = choose_alpha(outer_train, "trace_plus_load_interactions")
        for trial in alpha_trials:
            tuning.append(
                {
                    "experiment": "mixed_grouped_loow",
                    "model": model,
                    "held_out_workload_id": held_out,
                    "tuning_type": "alpha",
                    **trial,
                }
            )
        temperature_trials = []
        for temperature in SOFT_GATE_TEMPERATURES:
            errors = []
            for inner_held_out in sorted(
                {sample["workload_id"] for sample in outer_train}
            ):
                inner_train = [
                    sample
                    for sample in outer_train
                    if sample["workload_id"] != inner_held_out
                ]
                validation = [
                    sample
                    for sample in outer_train
                    if sample["workload_id"] == inner_held_out
                ]
                for sample in validation:
                    scored = dict(sample)
                    scored["load_flag"] = soft_regime_score(
                        inner_train, sample, temperature
                    )
                    prediction = predict_one(
                        inner_train,
                        scored,
                        "trace_plus_load_interactions",
                        alpha,
                    )
                    errors.append(
                        abs(prediction - sample["baseline_median_ms"])
                        / sample["baseline_median_ms"]
                        * 100.0
                    )
            temperature_trials.append(
                {
                    "temperature": temperature,
                    "inner_mean_ape_percent": statistics.mean(errors),
                    "inner_median_ape_percent": statistics.median(errors),
                }
            )
        winner = min(
            temperature_trials,
            key=lambda row: (
                row["inner_mean_ape_percent"],
                row["inner_median_ape_percent"],
                row["temperature"],
            ),
        )
        temperature = float(winner["temperature"])
        for trial in temperature_trials:
            tuning.append(
                {
                    "experiment": "mixed_grouped_loow",
                    "model": model,
                    "held_out_workload_id": held_out,
                    "tuning_type": "temperature",
                    **trial,
                }
            )
        for test in tests:
            score = soft_regime_score(outer_train, test, temperature)
            scored = dict(test)
            scored["load_flag"] = score
            prediction = predict_one(
                outer_train, scored, "trace_plus_load_interactions", alpha
            )
            predicted_overhead = (test["trace_runtime_ms"] - prediction) / prediction * 100.0
            predictions.append(
                {
                    "experiment": "mixed_grouped_loow",
                    "model": model,
                    "held_out_workload_id": held_out,
                    "test_condition": test["condition"],
                    "inferred_load_score": score,
                    "inferred_condition": "high_load_01" if score >= 0.5 else "clean",
                    "condition_classification_correct": (score >= 0.5)
                    == (test["condition"] == "high_load_01"),
                    "selected_temperature": temperature,
                    "selected_alpha": alpha,
                    "actual_baseline_ms": test["baseline_median_ms"],
                    "predicted_baseline_ms": prediction,
                    "baseline_ape_percent": abs(prediction - test["baseline_median_ms"])
                    / test["baseline_median_ms"]
                    * 100.0,
                    "actual_overhead_percent": test["observed_overhead_percent"],
                    "predicted_overhead_percent": predicted_overhead,
                    "overhead_error_pp": predicted_overhead - test["observed_overhead_percent"],
                    "baseline_repeat_cv_percent": test["baseline_repeat_cv_percent"],
                    "detected_period_ms": test["detected_period_ms"],
                    "emission_horizon_seconds": test["emission_horizon_seconds"],
                }
            )
    return predictions, tuning


def summarize(predictions: list[dict]) -> list[dict]:
    keys = sorted(
        {
            (row["experiment"], row["model"], row["test_condition"])
            for row in predictions
        }
    )
    rows = []
    for experiment, model, condition in keys:
        group = [
            row
            for row in predictions
            if row["experiment"] == experiment
            and row["model"] == model
            and row["test_condition"] == condition
        ]
        errors = [row["baseline_ape_percent"] for row in group]
        overhead = [abs(row["overhead_error_pp"]) for row in group]
        rows.append(
            {
                "experiment": experiment,
                "model": model,
                "test_condition": condition,
                "n": len(group),
                "mape_percent": statistics.mean(errors),
                "median_ape_percent": statistics.median(errors),
                "p90_ape_percent": percentile(errors, 0.90),
                "max_ape_percent": max(errors),
                "within_10_percent": sum(error <= 10.0 for error in errors) / len(errors),
                "overhead_mae_pp": statistics.mean(overhead),
                "baseline_repeat_cv_mean_percent": statistics.mean(
                    row["baseline_repeat_cv_percent"] for row in group
                ),
            }
        )
    # Add combined-condition rows for each mixed model.
    for model in (
        "trace_only",
        "trace_plus_load_flag",
        "trace_plus_load_interactions",
        "trace_plus_pre_netdata",
        "trace_regime_moe",
        "trace_oracle_regime_moe",
        "trace_soft_regime_interactions",
    ):
        group = [
            row
            for row in predictions
            if row["experiment"] == "mixed_grouped_loow" and row["model"] == model
        ]
        errors = [row["baseline_ape_percent"] for row in group]
        overhead = [abs(row["overhead_error_pp"]) for row in group]
        summary = {
                "experiment": "mixed_grouped_loow",
                "model": model,
                "test_condition": "both",
                "n": len(group),
                "mape_percent": statistics.mean(errors),
                "median_ape_percent": statistics.median(errors),
                "p90_ape_percent": percentile(errors, 0.90),
                "max_ape_percent": max(errors),
                "within_10_percent": sum(error <= 10.0 for error in errors) / len(errors),
                "overhead_mae_pp": statistics.mean(overhead),
                "baseline_repeat_cv_mean_percent": statistics.mean(
                    row["baseline_repeat_cv_percent"] for row in group
                ),
            }
        classified = [row for row in group if "condition_classification_correct" in row]
        if classified:
            summary["condition_classification_accuracy"] = sum(
                bool(row["condition_classification_correct"]) for row in classified
            ) / len(classified)
        rows.append(summary)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified clean/high-load trace-only runtime model.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    clean_detections = read_csv(DEFAULT_CLEAN_DETECTIONS)
    high_detections = read_csv(DEFAULT_HIGH_DETECTIONS)
    samples = [
        load_sample("clean", row, DEFAULT_CLEAN_CACHE, DEFAULT_CLEAN_RUNS)
        for row in clean_detections
    ] + [
        load_sample("high_load_01", row, DEFAULT_HIGH_CACHE, DEFAULT_HIGH_RUNS)
        for row in high_detections
    ]

    scenarios = [
        ("clean_to_clean_loow", "trace_only", "clean", ("clean",)),
        ("clean_to_high_strict", "trace_only", "clean", ("high_load_01",)),
        ("high_to_clean_strict", "trace_only", "high_load_01", ("clean",)),
        ("mixed_grouped_loow", "trace_only", None, ("clean", "high_load_01")),
        ("mixed_grouped_loow", "trace_plus_load_flag", None, ("clean", "high_load_01")),
        ("mixed_grouped_loow", "trace_plus_load_interactions", None, ("clean", "high_load_01")),
        ("mixed_grouped_loow", "trace_plus_pre_netdata", None, ("clean", "high_load_01")),
    ]
    predictions = []
    tuning = []
    for experiment, model, train_condition, test_conditions in scenarios:
        print(f"{experiment}/{model}", flush=True)
        scenario_predictions, scenario_tuning = evaluate_scenario(
            samples, experiment, model, train_condition, test_conditions
        )
        predictions.extend(scenario_predictions)
        tuning.extend(scenario_tuning)
    for model, use_oracle in (
        ("trace_regime_moe", False),
        ("trace_oracle_regime_moe", True),
    ):
        print(f"mixed_grouped_loow/{model}", flush=True)
        scenario_predictions, scenario_tuning = evaluate_regime_moe(
            samples, model, use_oracle
        )
        predictions.extend(scenario_predictions)
        tuning.extend(scenario_tuning)
    print("mixed_grouped_loow/trace_soft_regime_interactions", flush=True)
    scenario_predictions, scenario_tuning = evaluate_soft_gate(samples)
    predictions.extend(scenario_predictions)
    tuning.extend(scenario_tuning)
    metrics = summarize(predictions)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    public_samples = [
        {key: value for key, value in sample.items() if not key.startswith("_")}
        for sample in samples
    ]
    write_csv(output / "samples.csv", public_samples)
    write_csv(output / "predictions.csv", predictions)
    write_csv(output / "metrics.csv", metrics)
    write_csv(output / "inner_alpha_selection.csv", tuning)
    contract = {
        "program": Path(__file__).name,
        "program_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "conditions": ["clean", "high_load_01"],
        "device": "RTX5090",
        "outer_split": "leave one workload out; both condition rows excluded for mixed experiments",
        "inner_split": "leave one workload out within outer training data",
        "trace_features": FEATURE_NAMES,
        "load_flag_ablation": True,
        "load_interaction_ablation": True,
        "trace_regime_moe": "nearest standardized trace-feature centroid selects a condition-specific expert",
        "trace_soft_regime_interactions": {
            "gate": "soft distance to trace-feature regime centroids",
            "temperature_candidates": SOFT_GATE_TEMPERATURES,
            "temperature_selection": "nested grouped leave-one-workload-out",
        },
        "pre_run_telemetry_features": TELEMETRY_FEATURES,
        "clean_pre_run_telemetry": "not collected; fold-local median imputation plus missing indicators",
        "forbidden_features": [
            "NVTX",
            "iterations.csv",
            "workload configuration",
            "batch size",
            "image size",
            "AMP",
            "GPU metrics",
            "during-run Netdata",
        ],
        "target": "condition-specific median unprofiled steady_window_mean_iter_ms",
    }
    (output / "experiment_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")

    def metric(experiment: str, model: str, condition: str) -> dict:
        return next(
            row
            for row in metrics
            if row["experiment"] == experiment
            and row["model"] == model
            and row["test_condition"] == condition
        )

    clean = metric("clean_to_clean_loow", "trace_only", "clean")
    clean_high = metric("clean_to_high_strict", "trace_only", "high_load_01")
    high_clean = metric("high_to_clean_strict", "trace_only", "clean")
    unified = metric("mixed_grouped_loow", "trace_only", "both")
    flagged = metric("mixed_grouped_loow", "trace_plus_load_flag", "both")
    interacted = metric("mixed_grouped_loow", "trace_plus_load_interactions", "both")
    telemetry = metric("mixed_grouped_loow", "trace_plus_pre_netdata", "both")
    regime = metric("mixed_grouped_loow", "trace_regime_moe", "both")
    oracle_regime = metric("mixed_grouped_loow", "trace_oracle_regime_moe", "both")
    soft_regime = metric("mixed_grouped_loow", "trace_soft_regime_interactions", "both")
    report = [
        "# Unified clean/high-load RTX5090 trace model",
        "",
        "The same frozen marker-free detector and trace feature schema are used for both conditions. "
        "Mixed evaluation excludes both clean and high-load rows of the held-out workload.",
        "",
        "| Experiment | Model | Test | MAPE | Median APE | P90 APE | <=10% |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in metrics:
        report.append(
            f"| {row['experiment']} | {row['model']} | {row['test_condition']} | "
            f"{row['mape_percent']:.2f}% | {row['median_ape_percent']:.2f}% | "
            f"{row['p90_ape_percent']:.2f}% | {row['within_10_percent']:.1%} |"
        )
    report.extend(
        [
            "",
            "## Headline results",
            "",
            f"- Clean-only LOOW MAPE: {clean['mape_percent']:.2f}%.",
            f"- Strict clean-to-high MAPE: {clean_high['mape_percent']:.2f}%.",
            f"- Strict high-to-clean MAPE: {high_clean['mape_percent']:.2f}%.",
            f"- Unified trace-only grouped LOOW MAPE: {unified['mape_percent']:.2f}%.",
            f"- Unified trace + load flag MAPE: {flagged['mape_percent']:.2f}%.",
            f"- Unified trace + load interactions MAPE: {interacted['mape_percent']:.2f}%.",
            f"- Unified trace + pre-run telemetry MAPE: {telemetry['mape_percent']:.2f}%.",
            f"- Trace-inferred regime mixture MAPE: {regime['mape_percent']:.2f}% "
            f"(classification accuracy {regime['condition_classification_accuracy']:.1%}).",
            f"- Oracle-regime mixture upper-bound MAPE: {oracle_regime['mape_percent']:.2f}%.",
            f"- Trace-only soft-regime interaction MAPE: {soft_regime['mape_percent']:.2f}% "
            f"(classification accuracy {soft_regime['condition_classification_accuracy']:.1%}).",
            "",
            "Clean runs contain no Netdata files. The telemetry ablation therefore uses fold-local imputation and "
            "missingness indicators; it is informative but confounded with condition identity.",
        ]
    )
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"headline": report[-7:-1]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
