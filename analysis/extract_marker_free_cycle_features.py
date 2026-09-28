from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import DEFAULT_OUTPUT_DIR


def safe_cv(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    if len(values) < 2 or not np.isfinite(mean) or abs(mean) < 1e-12:
        return 0.0
    return float(np.std(values, ddof=1) / abs(mean))


def normalized_slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    slope = float(np.polyfit(np.linspace(0.0, 1.0, len(values)), values, 1)[0])
    center = abs(float(np.mean(values)))
    return slope / center if center > 1e-12 else 0.0


def repeatability(cycles: np.ndarray, template: np.ndarray) -> float:
    if np.std(template) < 1e-12:
        return 0.0
    scores = []
    for cycle in cycles:
        if np.std(cycle) >= 1e-12:
            scores.append(float(np.corrcoef(cycle, template)[0, 1]))
    return float(np.mean(scores)) if scores else 0.0


def harmonic_fractions(template: np.ndarray) -> tuple[float, float]:
    centered = template - np.mean(template)
    power = np.abs(np.fft.rfft(centered)) ** 2
    total = float(np.sum(power[1:]))
    if total < 1e-12:
        return 0.0, 0.0
    first = float(power[1] / total) if len(power) > 1 else 0.0
    second = float(power[2] / total) if len(power) > 2 else 0.0
    return first, second


def extract_profile(detection: pd.Series, cache_path: Path) -> dict:
    cached = np.load(cache_path)
    values = cached["values"].astype(float)
    channels = [str(item) for item in cached["channels"]]
    base_bin_ms = float(cached["base_bin_ms"])
    period_samples = max(4, int(round(float(detection.detected_period_ms) / base_bin_ms)))
    start = max(0, int(round(float(detection.segment_start_seconds) * 1000 / base_bin_ms)))
    stop = min(
        values.shape[1],
        int(round(float(detection.emission_horizon_seconds) * 1000 / base_bin_ms)),
    )
    segment = values[:, start:stop]
    cycle_count = segment.shape[1] // period_samples
    if cycle_count < 3:
        raise ValueError(
            f"Only {cycle_count} cycles for {detection.device_id}/{detection.workload_id}"
        )
    segment = segment[:, -cycle_count * period_samples :]
    cycles = segment.reshape(len(channels), cycle_count, period_samples)
    row = {
        "device_id": detection.device_id,
        "workload_id": detection.workload_id,
        "detected_period_ms": float(detection.detected_period_ms),
        "detector_confidence": float(detection.confidence),
        "emission_horizon_seconds": float(detection.emission_horizon_seconds),
        "feature_segment_start_seconds": start * base_bin_ms / 1000,
        "feature_segment_seconds": segment.shape[1] * base_bin_ms / 1000,
        "feature_cycle_count": cycle_count,
        "period_samples": period_samples,
        "base_bin_ms": base_bin_ms,
    }
    channel_repeatability = []
    channel_cycle_cv = []
    for index, channel in enumerate(channels):
        channel_values = segment[index]
        channel_cycles = cycles[index]
        cycle_means = np.mean(channel_cycles, axis=1)
        template = np.median(channel_cycles, axis=0)
        mean = float(np.mean(channel_values))
        std = float(np.std(channel_values, ddof=1))
        p10, median, p90 = np.percentile(channel_values, [10, 50, 90])
        template_p10, template_p90 = np.percentile(template, [10, 90])
        harmonic_1, harmonic_2 = harmonic_fractions(template)
        channel_repeat = repeatability(channel_cycles, template)
        cycle_cv = safe_cv(cycle_means)
        prefix = f"mf_{channel}"
        row.update(
            {
                f"{prefix}_mean": mean,
                f"{prefix}_std": std,
                f"{prefix}_cv": safe_cv(channel_values),
                f"{prefix}_p10": float(p10),
                f"{prefix}_median": float(median),
                f"{prefix}_p90": float(p90),
                f"{prefix}_active_fraction": float(np.mean(channel_values > 0.05)),
                f"{prefix}_cycle_mean_cv": cycle_cv,
                f"{prefix}_cycle_mean_span": float(
                    np.percentile(cycle_means, 90) - np.percentile(cycle_means, 10)
                ),
                f"{prefix}_cycle_trend": normalized_slope(cycle_means),
                f"{prefix}_template_amplitude": float(
                    (template_p90 - template_p10) / abs(mean)
                )
                if abs(mean) > 1e-12
                else 0.0,
                f"{prefix}_harmonic1_fraction": harmonic_1,
                f"{prefix}_harmonic2_fraction": harmonic_2,
                f"{prefix}_cycle_repeatability": channel_repeat,
            }
        )
        channel_repeatability.append(channel_repeat)
        channel_cycle_cv.append(cycle_cv)

    correlations = np.corrcoef(segment)
    finite_correlations = []
    for left in range(len(channels)):
        for right in range(left + 1, len(channels)):
            value = float(correlations[left, right])
            if not np.isfinite(value):
                value = 0.0
            row[f"mf_cross_corr_{channels[left]}__{channels[right]}"] = value
            finite_correlations.append(value)
    row["mf_summary_cycle_repeatability_mean"] = float(np.mean(channel_repeatability))
    row["mf_summary_cycle_mean_cv_mean"] = float(np.mean(channel_cycle_cv))
    row["mf_summary_cross_correlation_mean"] = float(np.mean(finite_correlations))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create per-cycle features from deployment-safe cycle detections."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    detections = pd.read_csv(root / "marker_free_cycle_deployment_oof.csv")
    manifest = pd.read_csv(root / "marker_free_timeseries_manifest.csv")
    detections = detections[detections.accepted].merge(
        manifest[["device_id", "workload_id", "cache_path"]],
        on=["device_id", "workload_id"],
        validate="one_to_one",
    )
    rows = []
    for detection in detections.itertuples(index=False):
        rows.append(extract_profile(pd.Series(detection._asdict()), Path(detection.cache_path)))
    features = pd.DataFrame(rows).sort_values(["device_id", "workload_id"])
    if len(features) != len(detections):
        raise RuntimeError(f"Expected {len(detections)} feature rows, got {len(features)}")
    output = root / "marker_free_cycle_features.csv"
    features.to_csv(output, index=False)
    print(f"Wrote {len(features)} rows and {len(features.columns)} columns to {output}.")


if __name__ == "__main__":
    main()
