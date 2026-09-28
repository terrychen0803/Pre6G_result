from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from common import DEFAULT_CACHE_DIR, DEFAULT_OUTPUT_DIR
from detect_marker_free_cycles import autocorrelation, robust_scale
from evaluate_marker_free_cycles import ground_truth
from extract_marker_free_cycle_features import extract_profile


HORIZONS_SECONDS = (9, 12, 15)
BIN_FACTORS = (1, 2, 4, 8)
TAIL_FRACTIONS = (0.5, 0.65, 0.8)
MIN_PERIOD_MS = 15.0
ABSOLUTE_MAX_PERIOD_MS = 1000.0
MIN_CYCLES_IN_SEGMENT = 8.0
TOP_PEAKS_PER_CONFIGURATION = 6
MIN_PROMINENCE = 0.01
NINE_SECOND_ACCEPTANCE_THRESHOLD = 0.90
CONFIRMED_ACCEPTANCE_THRESHOLD = 0.65


def nearby_channel_agreement(
    channel_acf: np.ndarray,
    valid_indices: np.ndarray,
    lags_ms: np.ndarray,
    period_ms: float,
    bin_ms: float,
) -> float:
    tolerance_ms = max(3.0 * bin_ms, 0.04 * period_ms)
    votes = []
    minimum_distance = max(1, int(round(8.0 / bin_ms)))
    for acf in channel_acf:
        peaks, _ = find_peaks(
            acf[valid_indices], prominence=MIN_PROMINENCE, distance=minimum_distance
        )
        periods = lags_ms[valid_indices[peaks]]
        votes.append(
            bool(len(periods) and np.min(np.abs(periods - period_ms)) <= tolerance_ms)
        )
    return float(np.mean(votes))


def configuration_candidates(
    prefix: np.ndarray,
    base_bin_ms: float,
    factor: int,
    tail_fraction: float,
) -> list[dict]:
    usable = (prefix.shape[1] // factor) * factor
    values = prefix[:, :usable].reshape(prefix.shape[0], -1, factor).mean(axis=2)
    bin_ms = base_bin_ms * factor
    segment_count = max(64, int(values.shape[1] * tail_fraction))
    segment = robust_scale(values[:, -segment_count:])
    channel_acf = np.asarray([autocorrelation(channel) for channel in segment])
    consensus = np.median(channel_acf, axis=0)
    lags_ms = np.arange(len(consensus)) * bin_ms
    segment_duration_ms = segment.shape[1] * bin_ms
    maximum_period_ms = min(
        ABSOLUTE_MAX_PERIOD_MS, segment_duration_ms / MIN_CYCLES_IN_SEGMENT
    )
    valid_indices = np.where(
        (lags_ms >= MIN_PERIOD_MS) & (lags_ms <= maximum_period_ms)
    )[0]
    if not len(valid_indices):
        return []
    minimum_distance = max(1, int(round(8.0 / bin_ms)))
    peaks, properties = find_peaks(
        consensus[valid_indices],
        prominence=MIN_PROMINENCE,
        distance=minimum_distance,
    )
    if not len(peaks):
        return []
    peak_indices = valid_indices[peaks]
    strengths = np.clip(consensus[peak_indices], 0.0, 1.0) + np.clip(
        properties["prominences"], 0.0, 1.0
    )
    order = np.argsort(strengths)[::-1][:TOP_PEAKS_PER_CONFIGURATION]
    rows = []
    for rank, selected_order in enumerate(order, start=1):
        index = int(peak_indices[selected_order])
        period_ms = float(lags_ms[index])
        rows.append(
            {
                "period_ms": period_ms,
                "bin_ms": bin_ms,
                "tail_fraction": tail_fraction,
                "configuration": f"bin={bin_ms:g},tail={tail_fraction:g}",
                "rank": rank,
                "acf_height": float(consensus[index]),
                "acf_prominence": float(properties["prominences"][selected_order]),
                "channel_agreement": nearby_channel_agreement(
                    channel_acf,
                    valid_indices,
                    lags_ms,
                    period_ms,
                    bin_ms,
                ),
                "cycles_observed": segment_duration_ms / period_ms,
                "configuration_max_period_ms": maximum_period_ms,
            }
        )
    return rows


def cluster_candidates(candidates: list[dict], configuration_count: int) -> list[dict]:
    clusters: list[list[dict]] = []
    for candidate in sorted(candidates, key=lambda item: item["period_ms"]):
        selected_cluster = None
        selected_distance = math.inf
        for cluster in clusters:
            center = float(np.median([item["period_ms"] for item in cluster]))
            tolerance = max(4.0, 0.035 * center)
            distance = abs(candidate["period_ms"] - center)
            if distance <= tolerance and distance < selected_distance:
                selected_cluster = cluster
                selected_distance = distance
        if selected_cluster is None:
            clusters.append([candidate])
        else:
            # A configuration may vote at most once for one period cluster.
            existing = {
                item["configuration"]: item for item in selected_cluster
            }
            prior = existing.get(candidate["configuration"])
            if prior is None or candidate["rank"] < prior["rank"]:
                if prior is not None:
                    selected_cluster.remove(prior)
                selected_cluster.append(candidate)

    summaries = []
    for cluster in clusters:
        periods = np.asarray([item["period_ms"] for item in cluster], dtype=float)
        top1 = np.asarray([item["rank"] == 1 for item in cluster], dtype=float)
        prominence = np.asarray(
            [np.clip(item["acf_prominence"], 0.0, 1.0) for item in cluster]
        )
        height = np.asarray(
            [np.clip(item["acf_height"], 0.0, 1.0) for item in cluster]
        )
        agreement = np.asarray([item["channel_agreement"] for item in cluster])
        support = len(cluster) / configuration_count
        top1_support = float(np.sum(top1) / configuration_count)
        rank_quality = float(
            np.sum([1.0 / item["rank"] for item in cluster]) / configuration_count
        )
        score = float(
            0.25 * support
            + 0.25 * top1_support
            + 0.10 * rank_quality
            + 0.15 * np.median(prominence)
            + 0.10 * np.median(height)
            + 0.15 * np.mean(agreement)
        )
        summaries.append(
            {
                "period_ms": float(np.median(periods)),
                "period_spread_ms": float(np.percentile(periods, 90) - np.percentile(periods, 10)),
                "configuration_support": support,
                "top1_support": top1_support,
                "rank_quality": rank_quality,
                "acf_height": float(np.median(height)),
                "acf_prominence": float(np.median(prominence)),
                "channel_agreement": float(np.mean(agreement)),
                "cycles_observed": float(np.median([item["cycles_observed"] for item in cluster])),
                "cluster_score_raw": score,
                "cluster_members": len(cluster),
            }
        )
    return sorted(summaries, key=lambda item: item["cluster_score_raw"], reverse=True)


def phase_repeatability(
    prefix: np.ndarray, base_bin_ms: float, period_ms: float, tail_fraction: float = 0.65
) -> float:
    start = int(prefix.shape[1] * (1.0 - tail_fraction))
    segment = robust_scale(prefix[:, start:])
    period_samples = max(4, int(round(period_ms / base_bin_ms)))
    cycle_count = segment.shape[1] // period_samples
    if cycle_count < 3:
        return 0.0
    segment = segment[:, -cycle_count * period_samples :]
    cycles = segment.reshape(segment.shape[0], cycle_count, period_samples)
    correlations = []
    for channel_cycles in cycles:
        template = np.median(channel_cycles, axis=0)
        if np.std(template) < 1e-9:
            continue
        for cycle in channel_cycles:
            if np.std(cycle) >= 1e-9:
                value = float(np.corrcoef(cycle, template)[0, 1])
                if np.isfinite(value):
                    correlations.append(value)
    if not correlations:
        return 0.0
    return float(np.clip(np.median(correlations), -1.0, 1.0))


def harmonic_reconciliation(clusters: list[dict]) -> list[dict]:
    reconciled = []
    for candidate in clusters:
        penalty = 0.0
        related_shorter = ""
        for shorter in clusters:
            if shorter["period_ms"] >= candidate["period_ms"] * 0.8:
                continue
            ratio = candidate["period_ms"] / shorter["period_ms"]
            nearest = round(ratio)
            if 2 <= nearest <= 8 and abs(ratio - nearest) <= 0.05 * nearest:
                if shorter["cluster_score_raw"] >= 0.80 * candidate["cluster_score_raw"]:
                    penalty = max(penalty, 0.08)
                    related_shorter = f"{shorter['period_ms']:.3f}x{nearest}"
        item = dict(candidate)
        item["harmonic_penalty"] = penalty
        item["related_shorter_candidate"] = related_shorter
        item["cluster_score"] = item["cluster_score_raw"] - penalty
        reconciled.append(item)
    return sorted(reconciled, key=lambda item: item["cluster_score"], reverse=True)


def detect_horizon(values: np.ndarray, base_bin_ms: float, horizon: int) -> tuple[dict, list[dict]]:
    sample_count = int(horizon * 1000 / base_bin_ms)
    prefix = values[:, :sample_count]
    rows = []
    for factor in BIN_FACTORS:
        for tail_fraction in TAIL_FRACTIONS:
            rows.extend(
                configuration_candidates(prefix, base_bin_ms, factor, tail_fraction)
            )
    configuration_count = len(BIN_FACTORS) * len(TAIL_FRACTIONS)
    clusters = harmonic_reconciliation(cluster_candidates(rows, configuration_count))
    if not clusters:
        return {
            "detected_period_ms": np.nan,
            "confidence": 0.0,
            "accepted": False,
            "candidate_count": 0,
        }, []
    for candidate in clusters:
        candidate["phase_repeatability"] = phase_repeatability(
            prefix, base_bin_ms, candidate["period_ms"]
        )
        candidate["final_score"] = float(
            candidate["cluster_score"]
            + 0.10 * max(0.0, candidate["phase_repeatability"])
        )
    clusters.sort(key=lambda item: item["final_score"], reverse=True)
    winner = clusters[0]
    runner_up = clusters[1]["final_score"] if len(clusters) > 1 else 0.0
    margin = max(0.0, float(winner["final_score"] - runner_up))
    stability = float(
        np.clip(
            1.0
            - winner["period_spread_ms"]
            / max(0.08 * winner["period_ms"], 4.0),
            0.0,
            1.0,
        )
    )
    confidence = float(
        np.clip(
            0.50 * winner["final_score"]
            + 0.20 * winner["top1_support"]
            + 0.10 * winner["channel_agreement"]
            + 0.10 * stability
            + 0.10 * min(1.0, margin / 0.20),
            0.0,
            1.0,
        )
    )
    detection = {
        "detected_period_ms": winner["period_ms"],
        "confidence": confidence,
        "accepted": confidence >= CONFIRMED_ACCEPTANCE_THRESHOLD,
        "candidate_count": len(clusters),
        "winner_margin": margin,
        "configuration_support": winner["configuration_support"],
        "top1_support": winner["top1_support"],
        "period_spread_ms": winner["period_spread_ms"],
        "acf_height": winner["acf_height"],
        "acf_prominence": winner["acf_prominence"],
        "channel_agreement": winner["channel_agreement"],
        "cycles_observed": winner["cycles_observed"],
        "phase_repeatability": winner["phase_repeatability"],
        "harmonic_penalty": winner["harmonic_penalty"],
        "segment_start_seconds": horizon * (1.0 - 0.65),
    }
    return detection, clusters


def detect_profile(cache_path: Path) -> tuple[list[dict], list[dict], dict]:
    cached = np.load(cache_path)
    values = cached["values"].astype(float)
    base_bin_ms = float(cached["base_bin_ms"])
    device_id = (
        str(cached["device_id"])
        if "device_id" in cached.files
        else cache_path.parent.name
    )
    workload_id = (
        str(cached["workload_id"])
        if "workload_id" in cached.files
        else cache_path.stem
    )
    detections = []
    clusters = []
    for horizon in HORIZONS_SECONDS:
        required = int(horizon * 1000 / base_bin_ms)
        if values.shape[1] < required:
            continue
        detection, horizon_clusters = detect_horizon(values, base_bin_ms, horizon)
        detection.update(
            {
                "device_id": device_id,
                "workload_id": workload_id,
                "horizon_seconds": horizon,
            }
        )
        detections.append(detection)
        for rank, cluster in enumerate(horizon_clusters, start=1):
            clusters.append(
                {
                    "device_id": device_id,
                    "workload_id": workload_id,
                    "horizon_seconds": horizon,
                    "candidate_rank": rank,
                    **cluster,
                }
            )
    if not detections:
        raise ValueError(f"No supported horizon for {cache_path}")

    chosen = None
    previous_periods: list[float] = []
    for detection in detections:
        period = float(detection["detected_period_ms"])
        if previous_periods:
            nearest_relative_error = min(
                abs(period - prior) / max(period, prior) for prior in previous_periods
            )
            detection["cross_horizon_stability"] = float(
                np.clip(1.0 - nearest_relative_error / 0.08, 0.0, 1.0)
            )
            detection["confidence"] = float(
                np.clip(
                    0.85 * detection["confidence"]
                    + 0.15 * detection["cross_horizon_stability"],
                    0.0,
                    1.0,
                )
            )
            detection["accepted"] = (
                detection["confidence"] >= CONFIRMED_ACCEPTANCE_THRESHOLD
            )
        else:
            detection["cross_horizon_stability"] = np.nan
        previous_periods.append(period)
    nine_second = next(
        (item for item in detections if item["horizon_seconds"] == 9), None
    )
    if (
        nine_second is not None
        and nine_second["confidence"] >= NINE_SECOND_ACCEPTANCE_THRESHOLD
    ):
        chosen = dict(nine_second)
        chosen["selected_confidence_threshold"] = NINE_SECOND_ACCEPTANCE_THRESHOLD
        chosen["selection_reason"] = "high_confidence_9s"
    else:
        chosen = next(
            (
                dict(item)
                for item in detections
                if item["horizon_seconds"] >= 12
                and item["confidence"] >= CONFIRMED_ACCEPTANCE_THRESHOLD
            ),
            None,
        )
        if chosen is not None:
            chosen["selected_confidence_threshold"] = (
                CONFIRMED_ACCEPTANCE_THRESHOLD
            )
            chosen["selection_reason"] = "cross_horizon_confirmed"
    if chosen is None:
        last_available = dict(detections[-1])
        chosen = last_available
        chosen["selected_confidence_threshold"] = CONFIRMED_ACCEPTANCE_THRESHOLD
        if chosen["confidence"] >= CONFIRMED_ACCEPTANCE_THRESHOLD:
            chosen["accepted"] = True
            chosen["emission_horizon_seconds"] = chosen["horizon_seconds"]
            chosen["selection_reason"] = "terminal_capture_fallback"
        else:
            chosen["accepted"] = False
            chosen["emission_horizon_seconds"] = np.nan
            chosen["selection_reason"] = "confidence_below_threshold"
    else:
        chosen["accepted"] = True
        chosen["emission_horizon_seconds"] = chosen["horizon_seconds"]
    return detections, clusters, chosen


def discover_yolo_profiles(manifest_path: Path) -> list[Path]:
    manifest = pd.read_csv(manifest_path)
    return [Path(value) for value in manifest.cache_path]


def discover_gemma_profiles(timeseries_dir: Path) -> list[Path]:
    return sorted(timeseries_dir.glob("G*.npz"))


def run_collection(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detection_rows = []
    cluster_rows = []
    deployment_rows = []
    for index, path in enumerate(paths, start=1):
        print(f"[{index:02d}/{len(paths):02d}] {path.stem}")
        detections, clusters, deployment = detect_profile(path)
        detection_rows.extend(detections)
        cluster_rows.extend(clusters)
        deployment_rows.append(deployment)
    return (
        pd.DataFrame(detection_rows),
        pd.DataFrame(cluster_rows),
        pd.DataFrame(deployment_rows),
    )


def add_yolo_oracle(deployment: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    oracle = ground_truth(cache_dir)
    result = deployment.merge(
        oracle, on=["device_id", "workload_id"], validate="one_to_one"
    )
    result["period_ape_percent"] = (
        (result.detected_period_ms - result.oracle_period_ms).abs()
        / result.oracle_period_ms
        * 100
    )
    return result


def add_gemma_audit(deployment: pd.DataFrame, manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    columns = [
        "workload_id",
        "source_baseline_median_ms",
        "target_baseline_median_ms",
        "runtime_regression_eligible",
        "target_supported",
    ]
    result = deployment.merge(manifest[columns], on="workload_id", validate="one_to_one")
    result["profile_to_unprofiled_ratio"] = (
        result.detected_period_ms / result.source_baseline_median_ms
    )
    result["profile_vs_unprofiled_gap_percent"] = (
        (result.detected_period_ms - result.source_baseline_median_ms).abs()
        / result.source_baseline_median_ms
        * 100
    )
    return result


def extract_gemma_features(
    deployment: pd.DataFrame, timeseries_dir: Path
) -> pd.DataFrame:
    rows = []
    for item in deployment[deployment.accepted].itertuples(index=False):
        rows.append(
            extract_profile(
                pd.Series(item._asdict()), timeseries_dir / f"{item.workload_id}.npz"
            )
        )
    return pd.DataFrame(rows).sort_values(["device_id", "workload_id"])


def metric_summary(data: pd.DataFrame, error_column: str) -> dict:
    accepted = data[data.accepted]
    evaluated = accepted[accepted[error_column].notna()]
    return {
        "profiles": len(data),
        "accepted": len(accepted),
        "coverage": len(accepted) / len(data),
        "evaluated": len(evaluated),
        "mean": float(evaluated[error_column].mean()),
        "median": float(evaluated[error_column].median()),
        "p90": float(evaluated[error_column].quantile(0.9)),
        "max": float(evaluated[error_column].max()),
        "within_10": float((evaluated[error_column] <= 10).mean()),
    }


def write_report(
    path: Path,
    yolo: pd.DataFrame,
    gemma: pd.DataFrame,
    features: pd.DataFrame,
) -> None:
    yolo_metrics = metric_summary(yolo, "period_ape_percent")
    gemma_metrics = metric_summary(gemma, "profile_vs_unprofiled_gap_percent")
    eligible = gemma[gemma.runtime_regression_eligible]
    eligible_accepted = eligible[eligible.accepted]
    lines = [
        "# Part C2: multiscale marker-free period detector",
        "",
        "## Detector contract",
        "",
        "The detector reads only the five timestamped GPU metric channels. It uses 15-1000 ms dynamic search, four bin widths, three tail fractions, cross-configuration clustering, harmonic reconciliation, phase-fold repeatability, and cross-horizon stability. No marker, callback, baseline runtime, target runtime, or workload metadata is read by the detector.",
        "",
        "The old Part 7 and Part C programs and outputs are unchanged.",
        "",
        "The deployment policy emits early at 9 seconds only when confidence is at least 0.90. Otherwise it waits for a 12/15-second confirmation at confidence 0.65. If a legacy capture ends before 12 seconds, the last available detection may be emitted at the confirmed threshold.",
        "",
        "## YOLO same-run marker evaluation",
        "",
        f"- Profiles: {yolo_metrics['profiles']}",
        f"- Accepted: {yolo_metrics['accepted']} ({yolo_metrics['coverage']:.1%})",
        f"- Mean period APE: {yolo_metrics['mean']:.2f}%",
        f"- Median period APE: {yolo_metrics['median']:.2f}%",
        f"- P90 period APE: {yolo_metrics['p90']:.2f}%",
        f"- Max period APE: {yolo_metrics['max']:.2f}%",
        f"- Within 10%: {yolo_metrics['within_10']:.1%}",
        "",
        "## Gemma profile-period versus separate unprofiled baseline",
        "",
        f"- Profiles: {gemma_metrics['profiles']}",
        f"- Accepted: {gemma_metrics['accepted']} ({gemma_metrics['coverage']:.1%})",
        f"- Runtime-eligible accepted: {len(eligible_accepted)}/{len(eligible)} ({len(eligible_accepted) / len(eligible):.1%})",
        f"- Feature table: {len(features)} rows x {len(features.columns)} columns",
        f"- Mean absolute gap: {gemma_metrics['mean']:.2f}%",
        f"- Median absolute gap: {gemma_metrics['median']:.2f}%",
        f"- P90 absolute gap: {gemma_metrics['p90']:.2f}%",
        f"- Max absolute gap: {gemma_metrics['max']:.2f}%",
        f"- Within 10%: {gemma_metrics['within_10']:.1%}",
        "",
        "This Gemma gap is not a pure detector error: the detected period is measured during Nsight GPU-metric profiling, while the baseline is from a separate unprofiled run.",
        "",
        "## Gemma audit",
        "",
        "| Workload | Accepted | Horizon | Confidence | Profile period ms | Unprofiled baseline ms | Ratio | Runtime eligible |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in gemma.itertuples(index=False):
        baseline = (
            "nan"
            if not np.isfinite(float(row.source_baseline_median_ms))
            else f"{float(row.source_baseline_median_ms):.2f}"
        )
        ratio = (
            "nan"
            if not np.isfinite(float(row.profile_to_unprofiled_ratio))
            else f"{float(row.profile_to_unprofiled_ratio):.3f}"
        )
        horizon = (
            "nan"
            if not np.isfinite(float(row.emission_horizon_seconds))
            else f"{float(row.emission_horizon_seconds):.0f}"
        )
        lines.append(
            f"| {row.workload_id} | {bool(row.accepted)} | {horizon} | {float(row.confidence):.3f} | {float(row.detected_period_ms):.2f} | {baseline} | {ratio} | {bool(row.runtime_regression_eligible)} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="New multiscale marker-free period detector; does not modify Part 7."
    )
    parser.add_argument(
        "--yolo-manifest",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "marker_free_timeseries_manifest.csv",
    )
    parser.add_argument(
        "--yolo-oracle-cache", type=Path, default=DEFAULT_CACHE_DIR / "nsys"
    )
    parser.add_argument(
        "--gemma-timeseries-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR
        / "pre6g_gemma4_5090_to_4090"
        / "timeseries_1ms"
        / "full",
    )
    parser.add_argument(
        "--gemma-manifest",
        type=Path,
        default=DEFAULT_OUTPUT_DIR
        / "pre6g_gemma4_5090_to_4090"
        / "gemma_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR
        / "model_experiments"
        / "pre6g_multiscale_detector_part_c2",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Detecting YOLO profiles")
    yolo_detections, yolo_candidates, yolo_deployment = run_collection(
        discover_yolo_profiles(args.yolo_manifest)
    )
    yolo_deployment = add_yolo_oracle(yolo_deployment, args.yolo_oracle_cache)

    print("Detecting Gemma profiles")
    gemma_detections, gemma_candidates, gemma_deployment = run_collection(
        discover_gemma_profiles(args.gemma_timeseries_dir)
    )
    # Baselines are joined only after all detector decisions have been emitted.
    gemma_deployment = add_gemma_audit(gemma_deployment, args.gemma_manifest)
    gemma_features = extract_gemma_features(
        gemma_deployment, args.gemma_timeseries_dir.resolve()
    )

    yolo_detections.to_csv(output_dir / "yolo_horizon_detections.csv", index=False)
    yolo_candidates.to_csv(output_dir / "yolo_period_candidates.csv", index=False)
    yolo_deployment.to_csv(output_dir / "yolo_deployment_detections.csv", index=False)
    gemma_detections.to_csv(output_dir / "gemma_horizon_detections.csv", index=False)
    gemma_candidates.to_csv(output_dir / "gemma_period_candidates.csv", index=False)
    gemma_deployment.to_csv(output_dir / "gemma_deployment_detections.csv", index=False)
    gemma_features.to_csv(output_dir / "gemma_cycle_features.csv", index=False)
    contract = {
        "program": Path(__file__).name,
        "new_detector": True,
        "old_programs_modified": False,
        "horizons_seconds": list(HORIZONS_SECONDS),
        "bin_factors": list(BIN_FACTORS),
        "tail_fractions": list(TAIL_FRACTIONS),
        "minimum_period_ms": MIN_PERIOD_MS,
        "absolute_maximum_period_ms": ABSOLUTE_MAX_PERIOD_MS,
        "minimum_cycles_in_segment": MIN_CYCLES_IN_SEGMENT,
        "nine_second_acceptance_threshold": NINE_SECOND_ACCEPTANCE_THRESHOLD,
        "confirmed_acceptance_threshold": CONFIRMED_ACCEPTANCE_THRESHOLD,
        "detector_inputs": ["five timestamped GPU metric channels", "horizon"],
        "forbidden_detector_inputs": [
            "NVTX markers",
            "callbacks",
            "baseline runtime",
            "target runtime",
            "workload metadata",
        ],
        "oracle_and_baseline_usage": "post-detection evaluation only",
    }
    (output_dir / "experiment_contract.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )
    write_report(
        output_dir / "report.md",
        yolo_deployment,
        gemma_deployment,
        gemma_features,
    )
    print(
        json.dumps(
            {
                "yolo_profiles": len(yolo_deployment),
                "yolo_accepted": int(yolo_deployment.accepted.sum()),
                "gemma_profiles": len(gemma_deployment),
                "gemma_accepted": int(gemma_deployment.accepted.sum()),
                "gemma_feature_rows": len(gemma_features),
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
