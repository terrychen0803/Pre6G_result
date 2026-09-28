from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

from common import DEFAULT_OUTPUT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize short-window accuracy/cost tradeoffs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    output = root / "model_experiments" / "part6_short_window"
    features = pd.read_csv(root / "short_window_profile_features.csv")
    predictions = pd.read_csv(output / "predictions.csv")
    metrics = pd.read_csv(output / "metrics.csv")

    probe = (
        features.groupby(["device_id", "window_iterations"])
        .quality_total_probe_elapsed_ms.agg(
            min_ms="min", median_ms="median", p90_ms=lambda x: x.quantile(0.9), max_ms="max"
        )
        .reset_index()
    )
    probe["median_seconds"] = probe.median_ms / 1000
    probe["p90_seconds"] = probe.p90_ms / 1000
    probe.to_csv(output / "probe_cost.csv", index=False)

    rows = []
    selected_models = [
        "behavior_mean_ridge",
        "behavior_distribution_ridge",
        "behavior_compact_cycle_ridge",
        "behavior_mean_duration_ridge",
    ]
    for direction in sorted(predictions.direction.unique()):
        source = direction.split("_to_", 1)[0]
        for window in sorted(predictions.window_iterations.unique()):
            cost = probe[(probe.device_id == source) & (probe.window_iterations == window)].iloc[0]
            for model in selected_models:
                metric = metrics[
                    (metrics.direction == direction)
                    & (metrics.window_iterations == window)
                    & (metrics.model == model)
                ].iloc[0]
                rows.append(
                    {
                        "direction": direction,
                        "window_iterations": window,
                        "source_probe_median_seconds": cost.median_seconds,
                        "source_probe_p90_seconds": cost.p90_seconds,
                        "model": model,
                        "duration_feature_used": "duration" in model,
                        "mape_percent": metric.mape_percent,
                        "mae_ms": metric.mae_ms,
                        "max_ape_percent": metric.max_ape_percent,
                    }
                )
    tradeoff = pd.DataFrame(rows)
    tradeoff.to_csv(output / "window_tradeoff.csv", index=False)

    paired_rows = []
    for direction in sorted(predictions.direction.unique()):
        for model in ("behavior_mean_ridge", "behavior_mean_duration_ridge"):
            reference = predictions[
                (predictions.direction == direction)
                & (predictions.window_iterations == 108)
                & (predictions.model == model)
            ][["workload_id", "ape_percent"]].rename(columns={"ape_percent": "reference_ape"})
            for window in (5, 10, 20, 50):
                current = predictions[
                    (predictions.direction == direction)
                    & (predictions.window_iterations == window)
                    & (predictions.model == model)
                ][["workload_id", "ape_percent"]].rename(columns={"ape_percent": "current_ape"})
                paired = current.merge(reference, on="workload_id", validate="one_to_one")
                paired_rows.append(
                    {
                        "direction": direction,
                        "model": model,
                        "window_iterations": window,
                        "reference_window": 108,
                        "mean_ape_change_pp": float(
                            (paired.current_ape - paired.reference_ape).mean()
                        ),
                        "current_better_count": int(
                            (paired.current_ape < paired.reference_ape).sum()
                        ),
                        "wilcoxon_p": float(
                            wilcoxon(paired.current_ape, paired.reference_ape).pvalue
                        ),
                    }
                )
    paired_frame = pd.DataFrame(paired_rows)
    paired_frame.to_csv(output / "paired_window_comparisons.csv", index=False)

    lines = [
        "# Short-window profiling decision summary",
        "",
        "Probe time is measured from TRAIN_ITER_0001 start through the last requested steady iteration. It includes 20 warmup iterations and model-run startup inside the NVTX ranges, but excludes Nsight process startup/export overhead.",
        "",
        "Behavior-only models do not use timestamp duration, GPU sample count, window span, source ground-truth runtime, or target runtime. Duration-assisted is shown separately.",
        "",
        "| Direction | K | Median probe s | P90 probe s | Behavior mean MAPE | Distribution MAPE | Compact-cycle MAPE | Mean + duration MAPE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for direction in sorted(tradeoff.direction.unique()):
        for window in sorted(tradeoff.window_iterations.unique()):
            group = tradeoff[
                (tradeoff.direction == direction)
                & (tradeoff.window_iterations == window)
            ].set_index("model")
            first = group.iloc[0]
            lines.append(
                f"| {direction} | {window} | {first.source_probe_median_seconds:.2f} | {first.source_probe_p90_seconds:.2f} | "
                f"{group.loc['behavior_mean_ridge'].mape_percent:.2f}% | "
                f"{group.loc['behavior_distribution_ridge'].mape_percent:.2f}% | "
                f"{group.loc['behavior_compact_cycle_ridge'].mape_percent:.2f}% | "
                f"{group.loc['behavior_mean_duration_ridge'].mape_percent:.2f}% |"
            )
    lines += [
        "",
        "## Interpretation",
        "",
        "- The full training job is not required. Every K row is causal and uses no sample after steady iteration 20+K.",
        "- RTX4090-to-RTX5090 is already strong at K=5. RTX5090-to-RTX4090 needs a longer prefix for stable low error, especially when using the observed source duration.",
        "- Expanded per-metric cycle features are unstable with only 24 training workloads. Five aggregated cycle descriptors avoid catastrophic outliers but do not beat simple metric means/distributions.",
        "- A practical current policy is K=20 for a balanced probe, or K=50 when reverse-direction accuracy is more important. K=5 is reasonable only for a latency-first policy with a wider error budget.",
    ]
    (output / "decision_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote probe cost, tradeoff, paired comparisons, and decision summary to {output}.")


if __name__ == "__main__":
    main()
