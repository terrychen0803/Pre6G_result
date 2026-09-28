from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("PRE6G_DATA_ROOT", PROJECT_ROOT / "data" / "raw")).expanduser()
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

from evaluate_marker_free_cycles import (  # noqa: E402
    ADAPTIVE_HORIZONS,
    apply_adaptive_policy,
    choose_confidence_threshold,
    classify_ratio,
)


DEFAULT_CLEAN_CANDIDATES = PROJECT_ROOT / "analysis_outputs" / "marker_free_cycle_candidates.csv"
DEFAULT_CLEAN_CACHE = PROJECT_ROOT / "analysis_cache" / "nsys"
DEFAULT_HIGH_CANDIDATES = (
    PROJECT_ROOT / "analysis" / "high_load_01" / "period_detection" / "marker_free_cycle_candidates.csv"
)
DEFAULT_HIGH_CACHE = PROJECT_ROOT / "analysis" / "high_load_01" / "cache" / "nsys"
DEFAULT_HIGH_RUNS = (
    DATA_ROOT
    / "high_load_01_RTX5090_20260923"
    / "runs_loaded"
    / "high_load_01"
    / "RTX5090"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "analysis" / "high_load_01" / "period_detection" / "clean_to_high"


def sqlite_oracle(cache_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(cache_dir.resolve().glob("*/*/nsys_gpu_metrics/profile.sqlite")):
        device, workload = path.parts[-4], path.parts[-3]
        connection = sqlite3.connect(path)
        try:
            ranges = connection.execute(
                "select text,start,end from NVTX_EVENTS "
                "where text >= 'TRAIN_ITER_0021' and text <= 'TRAIN_ITER_0128' "
                "order by text"
            ).fetchall()
        finally:
            connection.close()
        starts = np.asarray([row[1] for row in ranges], dtype=float)
        periods = np.diff(starts) / 1e6
        rows.append(
            {
                "device_id": device,
                "workload_id": workload,
                "nvtx_period_mean_ms": float(np.mean(periods)),
                "nvtx_period_median_ms": float(np.median(periods)),
                "nvtx_period_std_ms": float(np.std(periods, ddof=1)),
                "nvtx_period_cv_percent": float(np.std(periods, ddof=1) / np.mean(periods) * 100),
                "nvtx_range_count": len(ranges),
                "nvtx_spacing_count": len(periods),
            }
        )
    return pd.DataFrame(rows)


def iteration_oracle(high_runs: Path) -> pd.DataFrame:
    rows = []
    for workload in [f"C{index:02d}" for index in range(1, 25)]:
        path = high_runs / workload / "nsys_gpu_metrics_01" / "iterations.csv"
        data = pd.read_csv(path)
        steady = data[~data.warmup.astype(str).str.lower().isin({"true", "1", "yes"})]
        starts = steady.sort_values("iteration_id").start_unix_ns.astype(np.int64).to_numpy()
        periods = np.diff(starts) / 1e6
        rows.append(
            {
                "device_id": "RTX5090",
                "workload_id": workload,
                "gt_period_mean_ms": float(np.mean(periods)),
                "gt_period_median_ms": float(np.median(periods)),
                "gt_period_std_ms": float(np.std(periods, ddof=1)),
                "gt_period_cv_percent": float(np.std(periods, ddof=1) / np.mean(periods) * 100),
                "gt_steady_iteration_count": len(starts),
                "gt_spacing_count": len(periods),
            }
        )
    return pd.DataFrame(rows)


def attach_oracle(candidates: pd.DataFrame, oracle: pd.DataFrame) -> pd.DataFrame:
    result = candidates.merge(oracle, on=["device_id", "workload_id"], validate="many_to_one")
    result["oracle_period_ms"] = result.nvtx_period_median_ms
    result["period_ape_percent"] = (
        (result.detected_period_ms - result.oracle_period_ms).abs()
        / result.oracle_period_ms
        * 100
    )
    result["period_ratio"] = result.detected_period_ms / result.oracle_period_ms
    result["error_class"] = result.period_ratio.map(classify_ratio)
    return result


def choose_clean_hyperparameters(clean: pd.DataFrame) -> tuple[pd.DataFrame, float, pd.DataFrame, pd.DataFrame]:
    selections = []
    selected_rows = []
    for horizon in sorted(clean.horizon_seconds.unique()):
        horizon_data = clean[clean.horizon_seconds == horizon]
        scores = (
            horizon_data.groupby("configuration")
            .period_ape_percent.agg(["mean", "median", "max"])
            .reset_index()
            .sort_values(["mean", "median", "max", "configuration"])
        )
        winner = scores.iloc[0]
        selections.append(
            {
                "horizon_seconds": horizon,
                "configuration": winner.configuration,
                "clean_mean_ape_percent": float(winner["mean"]),
                "clean_median_ape_percent": float(winner["median"]),
                "clean_max_ape_percent": float(winner["max"]),
            }
        )
        selected_rows.append(horizon_data[horizon_data.configuration == winner.configuration])
    selected = pd.concat(selected_rows, ignore_index=True)
    threshold, threshold_trials = choose_confidence_threshold(
        selected[selected.horizon_seconds.isin(ADAPTIVE_HORIZONS)]
    )
    return pd.DataFrame(selections), threshold, threshold_trials, selected


def select_configs(data: pd.DataFrame, selections: pd.DataFrame) -> pd.DataFrame:
    selected = data.merge(
        selections[["horizon_seconds", "configuration"]],
        on=["horizon_seconds", "configuration"],
        validate="many_to_one",
    )
    return selected.sort_values(["device_id", "workload_id", "horizon_seconds"])


def summarize(name: str, data: pd.DataFrame, error_column: str) -> dict:
    accepted = data[data.accepted]
    return {
        "evaluation": name,
        "n_profiles": len(data),
        "n_accepted": len(accepted),
        "coverage": len(accepted) / len(data),
        "mean_ape_percent": float(accepted[error_column].mean()),
        "median_ape_percent": float(accepted[error_column].median()),
        "p90_ape_percent": float(accepted[error_column].quantile(0.9)),
        "max_ape_percent": float(accepted[error_column].max()),
        "within_5_percent": float((accepted[error_column] <= 5).mean()),
        "within_10_percent": float((accepted[error_column] <= 10).mean()),
        "mean_emission_horizon_seconds": float(accepted.emission_horizon_seconds.mean()),
    }


def summarize_forced_high(name: str, data: pd.DataFrame) -> dict:
    error_ms = data.detected_period_ms - data.gt_period_median_ms
    ape = data.ape_vs_gt_median_percent
    return {
        "evaluation": name,
        "n_profiles": len(data),
        "n_accepted": len(data),
        "coverage": 1.0,
        "mean_ape_percent": float(ape.mean()),
        "median_ape_percent": float(ape.median()),
        "p90_ape_percent": float(ape.quantile(0.9)),
        "max_ape_percent": float(ape.max()),
        "within_5_percent": float((ape <= 5).mean()),
        "within_10_percent": float((ape <= 10).mean()),
        "mean_emission_horizon_seconds": 15.0,
        "mae_ms": float(error_ms.abs().mean()),
        "rmse_ms": float(np.sqrt(np.mean(np.square(error_ms)))),
        "bias_ms": float(error_ms.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the original period policy on clean data and test high-load.")
    parser.add_argument("--clean-candidates", type=Path, default=DEFAULT_CLEAN_CANDIDATES)
    parser.add_argument("--clean-cache", type=Path, default=DEFAULT_CLEAN_CACHE)
    parser.add_argument("--high-candidates", type=Path, default=DEFAULT_HIGH_CANDIDATES)
    parser.add_argument("--high-cache", type=Path, default=DEFAULT_HIGH_CACHE)
    parser.add_argument("--high-runs", type=Path, default=DEFAULT_HIGH_RUNS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    clean_candidates = pd.read_csv(args.clean_candidates)
    clean = attach_oracle(clean_candidates, sqlite_oracle(args.clean_cache))
    selections, threshold, threshold_trials, clean_selected = choose_clean_hyperparameters(clean)
    clean_adaptive = apply_adaptive_policy(
        clean_selected[clean_selected.horizon_seconds.isin(ADAPTIVE_HORIZONS)], threshold
    )

    high_candidates = pd.read_csv(args.high_candidates)
    nvtx = sqlite_oracle(args.high_cache)
    iteration = iteration_oracle(args.high_runs)
    high_oracle = nvtx.merge(iteration, on=["device_id", "workload_id"], validate="one_to_one")
    high_oracle["nvtx_vs_iteration_median_abs_diff_ms"] = (
        high_oracle.nvtx_period_median_ms - high_oracle.gt_period_median_ms
    ).abs()
    high_oracle["nvtx_vs_iteration_median_ape_percent"] = (
        high_oracle.nvtx_vs_iteration_median_abs_diff_ms / high_oracle.gt_period_median_ms * 100
    )
    high = attach_oracle(high_candidates, high_oracle)
    high_selected = select_configs(high, selections)
    high_adaptive = apply_adaptive_policy(
        high_selected[high_selected.horizon_seconds.isin(ADAPTIVE_HORIZONS)], threshold
    )
    high_adaptive["ape_vs_gt_median_percent"] = (
        (high_adaptive.detected_period_ms - high_adaptive.gt_period_median_ms).abs()
        / high_adaptive.gt_period_median_ms
        * 100
    )
    high_adaptive["ape_vs_gt_mean_percent"] = (
        (high_adaptive.detected_period_ms - high_adaptive.gt_period_mean_ms).abs()
        / high_adaptive.gt_period_mean_ms
        * 100
    )
    high_adaptive["absolute_error_vs_gt_median_ms"] = (
        high_adaptive.detected_period_ms - high_adaptive.gt_period_median_ms
    ).abs()
    high_adaptive["signed_error_vs_gt_median_ms"] = (
        high_adaptive.detected_period_ms - high_adaptive.gt_period_median_ms
    )

    metrics = pd.DataFrame(
        [
            summarize("clean_train_in_sample", clean_adaptive, "period_ape_percent"),
            summarize("clean_to_high_confidence_gated", high_adaptive, "ape_vs_gt_median_percent"),
            summarize_forced_high("clean_to_high_forced_terminal", high_adaptive),
        ]
    )
    selections["clean_selected_confidence_threshold"] = threshold
    selections.to_csv(output / "clean_trained_hyperparameters.csv", index=False)
    threshold_trials.to_csv(output / "clean_threshold_selection.csv", index=False)
    high_oracle.to_csv(output / "high_load_period_ground_truth.csv", index=False)
    high_selected.to_csv(output / "clean_config_high_horizon_predictions.csv", index=False)
    high_adaptive.to_csv(output / "clean_to_high_adaptive_predictions.csv", index=False)
    metrics.to_csv(output / "clean_to_high_metrics.csv", index=False)
    (output / "experiment_contract.json").write_text(
        json.dumps(
            {
                "detector": "original Part 7 marker-free detector",
                "training_domain": "clean RTX4090 + RTX5090 GPU-metrics profiles",
                "test_domain": "high_load_01 RTX5090 GPU-metrics profiles",
                "configuration_selection": "all clean profiles only",
                "confidence_threshold_selection": "all clean profiles only",
                "detector_inputs": ["five GPU metric channels", "capture horizon"],
                "oracle_usage": "post-detection evaluation and clean-only hyperparameter selection",
                "trace_profiles_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = metrics[metrics.evaluation == "clean_to_high_confidence_gated"].iloc[0]
    forced = metrics[metrics.evaluation == "clean_to_high_forced_terminal"].iloc[0]
    lines = [
        "# Original clean-trained period detector on high-load RTX5090",
        "",
        "The detector and its adaptive policy are selected only from the existing clean GPU-metrics data. High-load markers/timestamps are used only after detection as evaluation labels.",
        "",
        f"- Clean-selected confidence threshold: {threshold:.2f}",
        f"- High-load accepted profiles: {int(result.n_accepted)}/{int(result.n_profiles)} ({result.coverage:.1%})",
        f"- Mean APE versus iteration-start median: {result.mean_ape_percent:.2f}%",
        f"- Median APE: {result.median_ape_percent:.2f}%",
        f"- P90 APE: {result.p90_ape_percent:.2f}%",
        f"- Maximum APE: {result.max_ape_percent:.2f}%",
        f"- Within 10%: {result.within_10_percent:.1%}",
        f"- Mean emission horizon: {result.mean_emission_horizon_seconds:.2f}s",
        "",
        "## Forced terminal output (diagnostic only)",
        "",
        "The following treats each rejected profile's last 15-second candidate as if it had been emitted. This measures period error across all workloads, but is not the confidence-gated deployment behavior.",
        "",
        f"- Profiles: {int(forced.n_profiles)}",
        f"- Mean APE: {forced.mean_ape_percent:.2f}%",
        f"- Median APE: {forced.median_ape_percent:.2f}%",
        f"- P90 APE: {forced.p90_ape_percent:.2f}%",
        f"- Maximum APE: {forced.max_ape_percent:.2f}%",
        f"- MAE: {forced.mae_ms:.2f} ms",
        f"- RMSE: {forced.rmse_ms:.2f} ms",
        f"- Bias: {forced.bias_ms:+.2f} ms",
        f"- Within 10%: {forced.within_10_percent:.1%}",
        "",
        "## Oracle consistency",
        "",
        f"- Median NVTX-versus-iterations median APE: {high_oracle.nvtx_vs_iteration_median_ape_percent.median():.4f}%",
        f"- Maximum NVTX-versus-iterations median APE: {high_oracle.nvtx_vs_iteration_median_ape_percent.max():.4f}%",
        "",
        "## Important interpretation",
        "",
        "A large error here is a clean-to-high domain-shift result, not evidence that the reports are misaligned: workload configs, software versions, metric descriptors, and iteration labels are checked separately.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote clean-to-high period evaluation to {output}")


if __name__ == "__main__":
    main()
