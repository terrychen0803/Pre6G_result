from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from common import DEFAULT_CACHE_DIR, DEFAULT_OUTPUT_DIR


ADAPTIVE_HORIZONS = (9, 12, 15)
CONFIDENCE_THRESHOLDS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7)


def ground_truth(cache_dir: Path) -> pd.DataFrame:
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
        periods_ms = np.diff(starts) / 1e6
        rows.append(
            {
                "device_id": device,
                "workload_id": workload,
                "oracle_period_ms": float(np.median(periods_ms)),
                "oracle_period_cv": float(np.std(periods_ms, ddof=1) / np.mean(periods_ms)),
                "oracle_range_count": len(ranges),
            }
        )
    return pd.DataFrame(rows)


def classify_ratio(ratio: float) -> str:
    if 0.45 <= ratio <= 0.55:
        return "half_period"
    if 1.8 <= ratio <= 2.2:
        return "double_period"
    if 0.9 <= ratio <= 1.1:
        return "within_10_percent"
    return "other_error"


def apply_adaptive_policy(data: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Return one accepted row per profile, or an explicit rejection row."""
    rows = []
    for (device, workload), profile in data.groupby(
        ["device_id", "workload_id"], sort=True
    ):
        chosen = None
        for horizon in ADAPTIVE_HORIZONS:
            candidate = profile[profile.horizon_seconds == horizon]
            if len(candidate) and float(candidate.iloc[0].confidence) >= threshold:
                chosen = candidate.iloc[0].to_dict()
                break
        if chosen is None:
            last = profile.sort_values("horizon_seconds").iloc[-1].to_dict()
            last["accepted"] = False
            last["emission_horizon_seconds"] = np.nan
            rows.append(last)
        else:
            chosen["accepted"] = True
            chosen["emission_horizon_seconds"] = chosen["horizon_seconds"]
            rows.append(chosen)
    return pd.DataFrame(rows)


def choose_confidence_threshold(training: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Maximize coverage while requiring <=10% error on accepted training profiles."""
    trials = []
    for threshold in CONFIDENCE_THRESHOLDS:
        adaptive = apply_adaptive_policy(training, threshold)
        accepted = adaptive[adaptive.accepted]
        trials.append(
            {
                "threshold": threshold,
                "coverage": len(accepted) / len(adaptive),
                "all_within_10_percent": bool(
                    len(accepted) and (accepted.period_ape_percent <= 10).all()
                ),
                "mean_ape": float(accepted.period_ape_percent.mean())
                if len(accepted)
                else np.inf,
                "mean_horizon": float(accepted.emission_horizon_seconds.mean())
                if len(accepted)
                else np.inf,
            }
        )
    scores = pd.DataFrame(trials)
    feasible = scores[scores.all_within_10_percent]
    pool = feasible if len(feasible) else scores
    winner = pool.sort_values(
        ["coverage", "mean_ape", "mean_horizon", "threshold"],
        ascending=[False, True, True, True],
    ).iloc[0]
    return float(winner.threshold), scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate marker-free periods; this is the only stage that reads oracle ranges."
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR / "nsys")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    candidates = pd.read_csv(root / "marker_free_cycle_candidates.csv")
    oracle = ground_truth(args.cache_dir)
    data = candidates.merge(
        oracle, on=["device_id", "workload_id"], validate="many_to_one"
    )
    data["period_ratio"] = data.detected_period_ms / data.oracle_period_ms
    data["period_ape_percent"] = (
        (data.detected_period_ms - data.oracle_period_ms).abs()
        / data.oracle_period_ms
        * 100
    )
    data["error_class"] = data.period_ratio.map(classify_ratio)

    selected_rows = []
    adaptive_rows = []
    threshold_selection_rows = []
    for workload in sorted(data.workload_id.unique()):
        training = data[data.workload_id != workload]
        test = data[data.workload_id == workload]
        fold_training_selected = []
        fold_test_selected = []
        for horizon in sorted(data.horizon_seconds.unique()):
            training_horizon = training[training.horizon_seconds == horizon]
            configuration_scores = (
                training_horizon.groupby("configuration")
                .period_ape_percent.agg(["mean", "median", "max"])
                .reset_index()
                .sort_values(["mean", "median", "max", "configuration"])
            )
            selected_configuration = configuration_scores.iloc[0].configuration
            selected = test[
                (test.horizon_seconds == horizon)
                & (test.configuration == selected_configuration)
            ].copy()
            selected["selected_configuration"] = selected_configuration
            selected["training_selection_mean_ape"] = configuration_scores.iloc[0]["mean"]
            selected_rows.extend(selected.to_dict("records"))
            fold_test_selected.extend(selected.to_dict("records"))
            trained = training_horizon[
                training_horizon.configuration == selected_configuration
            ].copy()
            trained["selected_configuration"] = selected_configuration
            trained["training_selection_mean_ape"] = configuration_scores.iloc[0]["mean"]
            fold_training_selected.extend(trained.to_dict("records"))

        fold_training_selected = pd.DataFrame(fold_training_selected)
        fold_test_selected = pd.DataFrame(fold_test_selected)
        threshold, threshold_scores = choose_confidence_threshold(
            fold_training_selected[
                fold_training_selected.horizon_seconds.isin(ADAPTIVE_HORIZONS)
            ]
        )
        threshold_scores["held_out_workload"] = workload
        threshold_selection_rows.extend(threshold_scores.to_dict("records"))
        adaptive_test = apply_adaptive_policy(
            fold_test_selected[
                fold_test_selected.horizon_seconds.isin(ADAPTIVE_HORIZONS)
            ],
            threshold,
        )
        adaptive_test["selected_confidence_threshold"] = threshold
        adaptive_rows.extend(adaptive_test.to_dict("records"))
    selected = pd.DataFrame(selected_rows).sort_values(
        ["horizon_seconds", "device_id", "workload_id"]
    )
    selected.to_csv(root / "marker_free_cycle_oof_predictions.csv", index=False)
    adaptive = pd.DataFrame(adaptive_rows).sort_values(["device_id", "workload_id"])
    adaptive.to_csv(root / "marker_free_cycle_adaptive_oof_predictions.csv", index=False)
    deployment_columns = [
        "device_id",
        "workload_id",
        "accepted",
        "emission_horizon_seconds",
        "detected_period_ms",
        "confidence",
        "configuration",
        "bin_ms",
        "tail_fraction",
        "segment_start_seconds",
        "acf_height",
        "acf_prominence",
        "channel_agreement",
        "cycles_observed",
        "candidate_count",
        "selected_confidence_threshold",
    ]
    deployment = adaptive[deployment_columns].copy()
    deployment.to_csv(root / "marker_free_cycle_deployment_oof.csv", index=False)

    metric_rows = []
    for (horizon, device), group in selected.groupby(
        ["horizon_seconds", "device_id"], sort=True
    ):
        metric_rows.append(
            {
                "horizon_seconds": horizon,
                "device_id": device,
                "n": len(group),
                "coverage": float(group.detected_period_ms.notna().mean()),
                "mean_period_ape_percent": float(group.period_ape_percent.mean()),
                "median_period_ape_percent": float(group.period_ape_percent.median()),
                "p90_period_ape_percent": float(group.period_ape_percent.quantile(0.9)),
                "max_period_ape_percent": float(group.period_ape_percent.max()),
                "within_5_percent": float((group.period_ape_percent <= 5).mean()),
                "within_10_percent": float((group.period_ape_percent <= 10).mean()),
                "half_period_count": int((group.error_class == "half_period").sum()),
                "double_period_count": int((group.error_class == "double_period").sum()),
                "median_confidence": float(group.confidence.median()),
            }
        )
    for horizon, group in selected.groupby("horizon_seconds", sort=True):
        metric_rows.append(
            {
                "horizon_seconds": horizon,
                "device_id": "both_devices",
                "n": len(group),
                "coverage": float(group.detected_period_ms.notna().mean()),
                "mean_period_ape_percent": float(group.period_ape_percent.mean()),
                "median_period_ape_percent": float(group.period_ape_percent.median()),
                "p90_period_ape_percent": float(group.period_ape_percent.quantile(0.9)),
                "max_period_ape_percent": float(group.period_ape_percent.max()),
                "within_5_percent": float((group.period_ape_percent <= 5).mean()),
                "within_10_percent": float((group.period_ape_percent <= 10).mean()),
                "half_period_count": int((group.error_class == "half_period").sum()),
                "double_period_count": int((group.error_class == "double_period").sum()),
                "median_confidence": float(group.confidence.median()),
            }
        )
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["horizon_seconds", "device_id"]
    )
    metrics.to_csv(root / "marker_free_cycle_metrics.csv", index=False)

    adaptive_metric_rows = []
    for device, group in list(adaptive.groupby("device_id", sort=True)) + [
        ("both_devices", adaptive)
    ]:
        accepted = group[group.accepted]
        adaptive_metric_rows.append(
            {
                "device_id": device,
                "n_profiles": len(group),
                "n_accepted": len(accepted),
                "coverage": len(accepted) / len(group),
                "mean_period_ape_percent": float(accepted.period_ape_percent.mean()),
                "median_period_ape_percent": float(accepted.period_ape_percent.median()),
                "p90_period_ape_percent": float(accepted.period_ape_percent.quantile(0.9)),
                "max_period_ape_percent": float(accepted.period_ape_percent.max()),
                "within_10_percent": float((accepted.period_ape_percent <= 10).mean()),
                "mean_emission_horizon_seconds": float(
                    accepted.emission_horizon_seconds.mean()
                ),
            }
        )
    adaptive_metrics = pd.DataFrame(adaptive_metric_rows)
    adaptive_metrics.to_csv(root / "marker_free_cycle_adaptive_metrics.csv", index=False)

    output = root / "model_experiments" / "part7_marker_free_cycles"
    output.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output / "oof_cycle_predictions.csv", index=False)
    metrics.to_csv(output / "cycle_metrics.csv", index=False)
    adaptive.to_csv(output / "adaptive_oof_predictions.csv", index=False)
    deployment.to_csv(output / "deployment_oof_detections.csv", index=False)
    adaptive_metrics.to_csv(output / "adaptive_metrics.csv", index=False)
    pd.DataFrame(threshold_selection_rows).to_csv(
        output / "threshold_selection_audit.csv", index=False
    )
    lines = [
        "# Part 7 checkpoint: marker-free cycle detection",
        "",
        "The detector reads only cached GPU metric time series. It never queries marker ranges. This evaluator reads marker ranges only after detection to create hidden period labels.",
        "",
        "Each outer fold holds out one workload across all available devices. Detector bin width, tail fraction, and confidence threshold are selected only from the remaining workloads.",
        "",
        "| Horizon | Device | N | Coverage | Mean period APE | Median | P90 | Max | <=5% | <=10% | Half | Double |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples():
        lines.append(
            f"| {row.horizon_seconds:.0f}s | {row.device_id} | {row.n} | {row.coverage:.1%} | {row.mean_period_ape_percent:.2f}% | {row.median_period_ape_percent:.2f}% | {row.p90_period_ape_percent:.2f}% | {row.max_period_ape_percent:.2f}% | {row.within_5_percent:.1%} | {row.within_10_percent:.1%} | {row.half_period_count} | {row.double_period_count} |"
        )
    overall_adaptive = adaptive_metrics[
        adaptive_metrics.device_id == "both_devices"
    ].iloc[0]
    if (
        float(overall_adaptive.coverage) == 1.0
        and float(overall_adaptive.within_10_percent) == 1.0
    ):
        checkpoint = (
            "The adaptive confidence-gated policy accepted all profiles and all accepted "
            "periods were within 10% of the hidden marker period on this dataset."
        )
    else:
        checkpoint = (
            f"The adaptive confidence-gated policy accepted "
            f"{int(overall_adaptive.n_accepted)}/{int(overall_adaptive.n_profiles)} profiles; "
            f"{float(overall_adaptive.within_10_percent):.1%} of accepted periods were within "
            "10% of the hidden marker period. This does not meet the original clean-data result."
        )
    lines += [
        "",
        "## Adaptive marker-free policy (nested threshold selection)",
        "",
        "Each held-out workload uses a confidence threshold selected only from the other 23 workloads. The detector tries 9s, then continues to 12s and 15s only if confidence is insufficient.",
        "",
        "| Device | Profiles | Accepted | Coverage | Mean APE | Median | P90 | Max | <=10% | Mean emission |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in adaptive_metrics.itertuples():
        lines.append(
            f"| {row.device_id} | {row.n_profiles} | {row.n_accepted} | {row.coverage:.1%} | {row.mean_period_ape_percent:.2f}% | {row.median_period_ape_percent:.2f}% | {row.p90_period_ape_percent:.2f}% | {row.max_period_ape_percent:.2f}% | {row.within_10_percent:.1%} | {row.mean_emission_horizon_seconds:.2f}s |"
        )
    lines += [
        "",
        "## Checkpoint decision",
        "",
        checkpoint,
        "",
        f"This evaluation contains {adaptive.workload_id.nunique()} workload configurations on {adaptive.device_id.nunique()} device(s). The source profiles were collected from instrumented runs even though marker/callback fields are never exposed to the detector. Results do not establish generalization to arbitrary workloads or load conditions.",
    ]
    lines += ["", "## Worst out-of-fold detections", "", "| Horizon | Device | Workload | Detected ms | Oracle ms | APE | Class | Configuration |", "|---:|---|---|---:|---:|---:|---|---|"]
    for row in selected.nlargest(12, "period_ape_percent").itertuples():
        lines.append(
            f"| {row.horizon_seconds:.0f}s | {row.device_id} | {row.workload_id} | {row.detected_period_ms:.2f} | {row.oracle_period_ms:.2f} | {row.period_ape_percent:.2f}% | {row.error_class} | `{row.selected_configuration}` |"
        )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "detector_contract.json").write_text(
        json.dumps(
            {
                "detector_inputs": [
                    "timestamped GPU metric values",
                    "capture horizon",
                ],
                "forbidden_detector_inputs": [
                    "iteration markers",
                    "training callbacks",
                    "baseline runtime",
                    "target runtime",
                ],
                "oracle_usage": "evaluation and training-fold parameter selection only",
                "split": "outer leave-one-workload-out, holding both devices out",
                "adaptive_horizons_seconds": list(ADAPTIVE_HORIZONS),
                "confidence_threshold_selection": "inside each outer training fold; maximize coverage subject to all accepted training periods having <=10% APE",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(selected)} OOF detections and {len(metrics)} metric rows to {output}.")


if __name__ == "__main__":
    main()
