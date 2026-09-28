from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from common import DEFAULT_CACHE_DIR, DEFAULT_OUTPUT_DIR


WINDOWS = (5, 10, 20, 50, 108)
STEADY_FIRST_ITERATION = 21
PHASE_BINS = 4
COMPACT_METRIC_SLUGS = (
    "gpc_clock_frequency_mhz",
    "sys_clock_frequency_mhz",
    "sm_issue_throughput",
    "sms_active_throughput",
    "tensor_active_throughput",
    "gr_active_throughput",
    "dram_read_bandwidth_throughput",
    "dram_write_bandwidth_throughput",
    "compute_warps_in_flight_throughput",
    "compute_warps_in_flight_avg",
    "compute_warps_in_flight_avg_warps_per_cycle",
)
CYCLE_METRIC_SLUGS = (
    "sm_issue_throughput",
    "sms_active_throughput",
    "tensor_active_throughput",
    "dram_read_bandwidth_throughput",
)


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def scale(name: str, values: np.ndarray) -> np.ndarray:
    if "[MHz]" in name or name.endswith("[Avg]"):
        return values / 1_000_000.0
    return values


def safe_cv(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    if not np.isfinite(mean) or abs(mean) < 1e-12:
        return 0.0
    return float(np.std(values, ddof=1) / abs(mean)) if len(values) > 1 else 0.0


def normalized_slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    x = np.linspace(0.0, 1.0, len(values))
    slope = float(np.polyfit(x, values, 1)[0])
    scale_value = abs(float(np.mean(values)))
    return slope / scale_value if scale_value > 1e-12 else 0.0


def phase_features(
    timestamps: np.ndarray,
    values: np.ndarray,
    ranges: list[tuple[int, int]],
) -> tuple[float, float, float]:
    profiles = []
    for start, end in ranges:
        left = int(np.searchsorted(timestamps, start, side="left"))
        right = int(np.searchsorted(timestamps, end, side="right"))
        ts = timestamps[left:right]
        vals = values[left:right]
        profile = np.full(PHASE_BINS, np.nan)
        if len(vals):
            phase = np.clip((ts - start) / max(end - start, 1), 0.0, 1.0 - 1e-12)
            bins = np.floor(phase * PHASE_BINS).astype(int)
            for index in range(PHASE_BINS):
                selected = vals[bins == index]
                if len(selected):
                    profile[index] = float(np.mean(selected))
        profiles.append(profile)
    matrix = np.asarray(profiles, dtype=float)
    with np.errstate(invalid="ignore"):
        template = np.nanmean(matrix, axis=0)
    if not np.isfinite(template).all():
        return 0.0, 0.0, 0.0
    center = abs(float(np.mean(template)))
    amplitude = float((np.max(template) - np.min(template)) / center) if center > 1e-12 else 0.0
    peak_phase = float(np.argmax(template) / (PHASE_BINS - 1))
    correlations = []
    for profile in matrix:
        if not np.isfinite(profile).all() or np.std(profile) < 1e-12 or np.std(template) < 1e-12:
            continue
        correlations.append(float(np.corrcoef(profile, template)[0, 1]))
    repeatability = float(np.mean(correlations)) if correlations else 0.0
    return amplitude, peak_phase, repeatability


def extract_database(path: Path) -> list[dict]:
    device, workload = path.parts[-4], path.parts[-3]
    connection = sqlite3.connect(path)
    try:
        iteration_rows = connection.execute(
            "select text,start,end from NVTX_EVENTS "
            "where text like 'TRAIN_ITER_%' order by text"
        ).fetchall()
        ranges_by_id = {
            int(text.rsplit("_", 1)[1]): (int(start), int(end))
            for text, start, end in iteration_rows
            if start is not None and end is not None
        }
        expected_ids = list(range(STEADY_FIRST_ITERATION, STEADY_FIRST_ITERATION + max(WINDOWS)))
        missing = [iteration for iteration in expected_ids if iteration not in ranges_by_id]
        if missing:
            raise ValueError(f"Missing steady NVTX iterations in {path}: {missing[:5]}")
        all_ranges = [ranges_by_id[iteration] for iteration in expected_ids]
        descriptor_rows = connection.execute(
            "select metricId,metricName from TARGET_INFO_GPU_METRICS order by metricId"
        ).fetchall()
        descriptors = {
            slug(name): (int(metric_id), str(name)) for metric_id, name in descriptor_rows
        }
        missing_metrics = [name for name in COMPACT_METRIC_SLUGS if name not in descriptors]
        if missing_metrics:
            raise ValueError(f"Missing compact metrics in {path}: {missing_metrics}")

        metric_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        full_start, full_end = all_ranges[0][0], all_ranges[-1][1]
        for metric_slug in COMPACT_METRIC_SLUGS:
            metric_id, metric_name = descriptors[metric_slug]
            samples = connection.execute(
                "select timestamp,value from GPU_METRICS "
                "where metricId=? and timestamp between ? and ? order by timestamp",
                (metric_id, full_start, full_end),
            ).fetchall()
            timestamps = np.asarray([row[0] for row in samples], dtype=np.int64)
            values = scale(metric_name, np.asarray([row[1] for row in samples], dtype=float))
            metric_series[metric_slug] = timestamps, values

        results = []
        for window in WINDOWS:
            ranges = all_ranges[:window]
            window_start, window_end = ranges[0][0], ranges[-1][1]
            durations_ms = np.asarray([(end - start) / 1e6 for start, end in ranges])
            starts = np.asarray([start for start, _ in ranges], dtype=float)
            start_intervals_ms = np.diff(starts) / 1e6
            row = {
                "device_id": device,
                "workload_id": workload,
                "window_iterations": window,
                "first_iteration_id": STEADY_FIRST_ITERATION,
                "last_iteration_id": STEADY_FIRST_ITERATION + window - 1,
                "quality_total_probe_elapsed_ms": (window_end - ranges_by_id[1][0]) / 1e6,
                "quality_steady_prefix_elapsed_ms": (window_end - window_start) / 1e6,
                "profile_window_span_per_iter_ms": (window_end - window_start) / 1e6 / window,
                "profile_nvtx_duration_mean_ms": float(np.mean(durations_ms)),
                "profile_nvtx_duration_cv": safe_cv(durations_ms),
                "profile_start_interval_mean_ms": float(np.mean(start_intervals_ms)) if len(start_intervals_ms) else float(durations_ms[0]),
            }
            minimum_samples = None
            for metric_slug, (timestamps, values) in metric_series.items():
                left = int(np.searchsorted(timestamps, window_start, side="left"))
                right = int(np.searchsorted(timestamps, window_end, side="right"))
                selected = values[left:right]
                if not len(selected):
                    raise ValueError(f"No {metric_slug} samples for {device}/{workload}/K={window}")
                minimum_samples = len(selected) if minimum_samples is None else min(minimum_samples, len(selected))
                prefix = f"metric_{metric_slug}"
                row[f"{prefix}_mean"] = float(np.mean(selected))
                row[f"{prefix}_cv"] = safe_cv(selected)
                row[f"{prefix}_p90"] = float(np.percentile(selected, 90))

                iteration_means = []
                for start, end in ranges:
                    il = int(np.searchsorted(timestamps, start, side="left"))
                    ir = int(np.searchsorted(timestamps, end, side="right"))
                    iteration_values = values[il:ir]
                    iteration_means.append(float(np.mean(iteration_values)) if len(iteration_values) else np.nan)
                iteration_means_array = np.asarray(iteration_means)
                finite = iteration_means_array[np.isfinite(iteration_means_array)]
                row[f"{prefix}_iteration_cv"] = safe_cv(finite) if len(finite) else 0.0
                row[f"{prefix}_iteration_trend"] = normalized_slope(finite) if len(finite) else 0.0
                if metric_slug in CYCLE_METRIC_SLUGS:
                    amplitude, peak_phase, repeatability = phase_features(
                        timestamps, values, ranges
                    )
                    row[f"{prefix}_phase_amplitude"] = amplitude
                    row[f"{prefix}_phase_peak"] = peak_phase
                    row[f"{prefix}_phase_repeatability"] = repeatability
            row["quality_min_metric_sample_count"] = minimum_samples
            results.append(row)
        return results
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract causal short-prefix Nsight GPU features.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR / "nsys")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    paths = sorted(args.cache_dir.resolve().glob("*/*/nsys_gpu_metrics/profile.sqlite"))
    rows = []
    for index, path in enumerate(paths, start=1):
        rows.extend(extract_database(path))
        print(f"[{index}/{len(paths)}] {path.parts[-4]}/{path.parts[-3]}", flush=True)
    result = pd.DataFrame(rows).sort_values(["device_id", "workload_id", "window_iterations"])
    expected = len(paths) * len(WINDOWS)
    if len(result) != expected:
        raise RuntimeError(f"Expected {expected} rows, extracted {len(result)}")
    output = args.output_dir.resolve() / "short_window_profile_features.csv"
    result.to_csv(output, index=False)
    print(f"Wrote {len(result)} short-window rows with {len(result.columns)} columns to {output}.")


if __name__ == "__main__":
    main()
