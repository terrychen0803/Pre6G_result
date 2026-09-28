from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import statistics
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("PRE6G_DATA_ROOT", PROJECT_ROOT / "data" / "raw")).expanduser()
DEFAULT_SOURCE_ROOT = (
    DATA_ROOT
    / "Pre6G_5090_dnn_gemma_vit"
    / "gemma4_runtime_prediction"
)
DEFAULT_TARGET_ROOT = (
    DATA_ROOT
    / "Pre6G_4090_dnn_gemma_vit"
    / "Pre6G"
    / "gemma4_runtime_prediction"
)
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "analysis_cache" / "pre6g_gemma4_5090_to_4090"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "analysis_outputs" / "pre6g_gemma4_5090_to_4090"
DEFAULT_NSYS = Path(os.environ.get("PRE6G_NSYS", "nsys"))

CHANNEL_SLUGS = (
    "sm_issue_throughput",
    "sms_active_throughput",
    "tensor_active_throughput",
    "dram_read_bandwidth_throughput",
    "dram_write_bandwidth_throughput",
)
HORIZONS_SECONDS = (9, 12, 15)
BASE_BIN_MS = 1.0
REQUIRED_TABLES = (
    "GPU_METRICS",
    "TARGET_INFO_GPU_METRICS",
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_MEMCPY",
)


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_experiments(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def baseline_summary(root: Path, device: str, workload: str) -> dict[str, Any]:
    values: list[float] = []
    peaks: list[float] = []
    missing: list[int] = []
    instrumentation: set[str] = set()
    measurement_iters: set[int] = set()
    for repeat in range(1, 6):
        path = root / "runs" / device / workload / f"baseline_{repeat:02d}" / "summary.json"
        if not path.exists():
            missing.append(repeat)
            continue
        payload = read_json(path)
        values.append(float(payload["steady_window_mean_iter_ms"]))
        peaks.append(float(payload.get("peak_vram_bytes", math.nan)))
        instrumentation.add(str(payload.get("instrumentation")))
        measurement_iters.add(int(payload.get("measurement_iters", -1)))
    return {
        "count": len(values),
        "missing_repeats": ",".join(map(str, missing)),
        "median_ms": statistics.median(values) if values else math.nan,
        "mean_ms": statistics.mean(values) if values else math.nan,
        "std_ms": statistics.stdev(values) if len(values) > 1 else (0.0 if values else math.nan),
        "median_peak_vram_bytes": statistics.median(peaks) if peaks else math.nan,
        "instrumentation": ",".join(sorted(instrumentation)),
        "measurement_iters": ",".join(map(str, sorted(measurement_iters))),
    }


def table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }


def row_count(connection: sqlite3.Connection, table: str) -> int:
    if table not in table_names(connection):
        return 0
    return int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])


def metric_descriptors(connection: sqlite3.Connection) -> dict[str, tuple[int, int, str]]:
    result: dict[str, tuple[int, int, str]] = {}
    for type_id, metric_id, metric_name in connection.execute(
        "SELECT typeId, metricId, metricName FROM TARGET_INFO_GPU_METRICS"
    ):
        result[slug(str(metric_name))] = (int(type_id), int(metric_id), str(metric_name))
    return result


def export_sqlite(nsys: Path, report: Path, database: Path, force: bool) -> None:
    if database.exists() and database.stat().st_size > 0 and not force:
        return
    database.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(nsys),
        "export",
        "--type",
        "sqlite",
        "--force-overwrite",
        "true",
        "--output",
        str(database),
        str(report),
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0 or not database.exists() or database.stat().st_size == 0:
        tail = "\n".join(completed.stdout.splitlines()[-30:])
        raise RuntimeError(f"Nsight export failed for {report}:\n{tail}")


def validate_sqlite(database: Path) -> tuple[dict[str, Any], dict[str, tuple[int, int, str]]]:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        names = table_names(connection)
        descriptors = metric_descriptors(connection)
        missing_tables = [name for name in REQUIRED_TABLES if name not in names]
        missing_channels = [name for name in CHANNEL_SLUGS if name not in descriptors]
        minimum, maximum, rows, distinct_timestamps = connection.execute(
            "SELECT MIN(timestamp), MAX(timestamp), COUNT(*), COUNT(DISTINCT timestamp) "
            "FROM GPU_METRICS"
        ).fetchone()
        nvtx_tables = sorted(name for name in names if "NVTX" in name.upper())
        nvtx_rows = sum(row_count(connection, name) for name in nvtx_tables)
        capture_seconds = (
            (float(maximum) - float(minimum)) / 1e9
            if minimum is not None and maximum is not None
            else math.nan
        )
        expected_timestamps = capture_seconds * 1000 + 1 if math.isfinite(capture_seconds) else math.nan
        sampling_coverage = (
            float(distinct_timestamps) / expected_timestamps
            if expected_timestamps and math.isfinite(expected_timestamps)
            else math.nan
        )
        result = {
            "sqlite_path": str(database),
            "sqlite_bytes": database.stat().st_size,
            "table_count": len(names),
            "missing_required_tables": ",".join(missing_tables),
            "metric_descriptor_count": len(descriptors),
            "missing_required_channels": ",".join(missing_channels),
            "gpu_metric_rows": int(rows or 0),
            "gpu_metric_distinct_timestamps": int(distinct_timestamps or 0),
            "gpu_metric_capture_seconds": capture_seconds,
            "gpu_metric_sampling_coverage": sampling_coverage,
            "cuda_kernel_rows": row_count(connection, "CUPTI_ACTIVITY_KIND_KERNEL"),
            "cuda_memcpy_rows": row_count(connection, "CUPTI_ACTIVITY_KIND_MEMCPY"),
            "nvtx_table_count": len(nvtx_tables),
            "nvtx_rows": nvtx_rows,
            "schema_valid": not missing_tables and not missing_channels and int(rows or 0) > 0,
        }
        return result, descriptors
    finally:
        connection.close()


def extract_timeseries(
    database: Path,
    descriptors: dict[str, tuple[int, int, str]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        raw: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        global_start: int | None = None
        global_end: int | None = None
        raw_counts: dict[str, int] = {}
        raw_intervals_ms: dict[str, float] = {}
        for channel in CHANNEL_SLUGS:
            type_id, metric_id, _ = descriptors[channel]
            samples = connection.execute(
                "SELECT timestamp, value FROM GPU_METRICS "
                "WHERE typeId=? AND metricId=? ORDER BY timestamp",
                (type_id, metric_id),
            ).fetchall()
            if not samples:
                raise ValueError(f"Empty required channel {channel} in {database}")
            timestamps = np.asarray([row[0] for row in samples], dtype=np.int64)
            values = np.asarray([row[1] for row in samples], dtype=np.float64)
            finite = np.isfinite(values)
            timestamps = timestamps[finite]
            values = values[finite]
            if len(timestamps) < 4:
                raise ValueError(f"Insufficient finite samples for {channel} in {database}")
            raw[channel] = (timestamps, values)
            raw_counts[channel] = len(timestamps)
            raw_intervals_ms[channel] = float(np.median(np.diff(timestamps)) / 1e6)
            global_start = int(timestamps[0]) if global_start is None else max(global_start, int(timestamps[0]))
            global_end = int(timestamps[-1]) if global_end is None else min(global_end, int(timestamps[-1]))
        if global_start is None or global_end is None or global_end <= global_start:
            raise ValueError(f"No common metric time range in {database}")
        step_ns = int(BASE_BIN_MS * 1e6)
        grid_ns = np.arange(global_start, global_end + 1, step_ns, dtype=np.int64)
        matrix = np.asarray(
            [np.interp(grid_ns, *raw[channel]) for channel in CHANNEL_SLUGS],
            dtype=np.float32,
        )
        time_ms = ((grid_ns - grid_ns[0]) / 1e6).astype(np.float32)
        standard_deviations = np.std(matrix, axis=1, dtype=np.float64)
        diagnostics = {
            "sample_count": int(matrix.shape[1]),
            "capture_seconds": float(time_ms[-1] / 1000),
            "nan_count": int(np.isnan(matrix).sum()),
            "infinite_count": int(np.isinf(matrix).sum()),
            "constant_channels": ",".join(
                channel
                for channel, deviation in zip(CHANNEL_SLUGS, standard_deviations)
                if deviation <= 1e-12
            ),
            "channel_std_min": float(np.min(standard_deviations)),
            "channel_std_max": float(np.max(standard_deviations)),
            "raw_sample_count_min": min(raw_counts.values()),
            "raw_sample_count_max": max(raw_counts.values()),
            "raw_median_interval_ms_min": min(raw_intervals_ms.values()),
            "raw_median_interval_ms_max": max(raw_intervals_ms.values()),
        }
        return time_ms, matrix, diagnostics
    finally:
        connection.close()


def save_npz(path: Path, time_ms: np.ndarray, values: np.ndarray, workload: str, horizon: int | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        time_ms=time_ms,
        values=values,
        channels=np.asarray(CHANNEL_SLUGS),
        base_bin_ms=np.asarray(BASE_BIN_MS),
        device_id=np.asarray("RTX5090"),
        workload_id=np.asarray(workload),
        workload_family=np.asarray("gemma4_lora_sft"),
        horizon_seconds=np.asarray(horizon if horizon is not None else -1),
        marker_free=np.asarray(True),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if math.isfinite(value) else "NA"


def main() -> None:
    parser = argparse.ArgumentParser(description="Process Pre6G Gemma Part A/B data")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--nsys", type=Path, default=DEFAULT_NSYS)
    parser.add_argument("--force-export", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    target_root = args.target_root.resolve()
    cache_root = args.cache_root.resolve()
    output_root = args.output_root.resolve()
    nsys = args.nsys.expanduser()
    if not nsys.exists():
        raise FileNotFoundError(nsys)
    experiments = load_experiments(source_root / "configs" / "experiments.csv")
    manifest_rows: list[dict[str, Any]] = []
    sqlite_rows: list[dict[str, Any]] = []
    timeseries_rows: list[dict[str, Any]] = []

    for index, experiment in enumerate(experiments, start=1):
        workload = experiment["workload_id"]
        run_dir = source_root / "runs" / "RTX5090" / workload / "nsys_deployment_01"
        report = run_dir / "profile.nsys-rep"
        profiling_config_path = run_dir / "profiling_config.json"
        target_failure_path = (
            target_root / "runs" / "RTX4090" / workload / "baseline_01" / "failure.json"
        )
        source_baseline = baseline_summary(source_root, "RTX5090", workload)
        target_baseline = baseline_summary(target_root, "RTX4090", workload)
        target_failure = read_json(target_failure_path) if target_failure_path.exists() else {}
        profiling_config = read_json(profiling_config_path) if profiling_config_path.exists() else {}
        report_exists = report.exists() and report.stat().st_size > 0
        target_supported = target_baseline["count"] == 5 and not target_failure
        regression_eligible = (
            report_exists
            and source_baseline["count"] == 5
            and target_baseline["count"] == 5
            and target_supported
        )
        manifest_rows.append(
            {
                "workload_family": "gemma4_lora_sft",
                "workload_id": workload,
                "source_device": "RTX5090",
                "target_device": "RTX4090",
                "profile_report_path": str(report),
                "profile_report_exists": report_exists,
                "profile_report_bytes": report.stat().st_size if report_exists else 0,
                "profile_report_sha256": sha256(report) if report_exists else "",
                "uses_workload_markers": profiling_config.get("uses_workload_markers"),
                "capture_delay_seconds": profiling_config.get("delay_seconds"),
                "capture_duration_seconds": profiling_config.get("duration_seconds"),
                "gpu_metrics_frequency_hz": profiling_config.get("gpu_metrics_frequency_hz"),
                "source_baseline_count": source_baseline["count"],
                "source_baseline_median_ms": source_baseline["median_ms"],
                "source_baseline_std_ms": source_baseline["std_ms"],
                "target_baseline_count": target_baseline["count"],
                "target_baseline_median_ms": target_baseline["median_ms"],
                "target_baseline_std_ms": target_baseline["std_ms"],
                "target_supported": target_supported,
                "target_failure_type": target_failure.get("exception_type", ""),
                "target_failure_message": target_failure.get("message", ""),
                "runtime_regression_eligible": regression_eligible,
                **{f"audit_config_{key}": value for key, value in experiment.items()},
            }
        )
        if not report_exists:
            print(f"[{index:02d}/18] {workload}: missing report")
            continue

        database = cache_root / "sqlite" / f"{workload}.sqlite"
        print(f"[{index:02d}/18] {workload}: export/validate/extract", flush=True)
        export_sqlite(nsys, report, database, args.force_export)
        validation, descriptors = validate_sqlite(database)
        validation["workload_id"] = workload
        sqlite_rows.append(validation)
        if not validation["schema_valid"]:
            continue

        time_ms, values, diagnostics = extract_timeseries(database, descriptors)
        full_path = cache_root / "timeseries_1ms" / "full" / f"{workload}.npz"
        save_npz(full_path, time_ms, values, workload, None)
        row = {
            "workload_id": workload,
            "full_timeseries_path": str(full_path),
            **diagnostics,
        }
        for horizon in HORIZONS_SECONDS:
            limit = int(np.searchsorted(time_ms, horizon * 1000, side="right"))
            if limit < 2 or float(time_ms[limit - 1]) < horizon * 1000 - BASE_BIN_MS:
                row[f"prefix_{horizon}s_valid"] = False
                row[f"prefix_{horizon}s_samples"] = limit
                row[f"prefix_{horizon}s_path"] = ""
                continue
            prefix_path = cache_root / "timeseries_1ms" / f"prefix_{horizon}s" / f"{workload}.npz"
            save_npz(prefix_path, time_ms[:limit], values[:, :limit], workload, horizon)
            row[f"prefix_{horizon}s_valid"] = True
            row[f"prefix_{horizon}s_samples"] = limit
            row[f"prefix_{horizon}s_path"] = str(prefix_path)
        row["timeseries_valid"] = (
            row["nan_count"] == 0
            and row["infinite_count"] == 0
            and not row["constant_channels"]
            and all(row.get(f"prefix_{horizon}s_valid", False) for horizon in HORIZONS_SECONDS)
        )
        timeseries_rows.append(row)

    write_csv(output_root / "gemma_manifest.csv", manifest_rows)
    write_csv(output_root / "sqlite_validation.csv", sqlite_rows)
    write_csv(output_root / "timeseries_validation.csv", timeseries_rows)

    report_count = sum(bool(row["profile_report_exists"]) for row in manifest_rows)
    eligible_count = sum(bool(row["runtime_regression_eligible"]) for row in manifest_rows)
    supported_count = sum(bool(row["target_supported"]) for row in manifest_rows)
    valid_sqlite_count = sum(bool(row["schema_valid"]) for row in sqlite_rows)
    valid_timeseries_count = sum(bool(row["timeseries_valid"]) for row in timeseries_rows)
    missing_channels = sorted(
        {
            item
            for row in sqlite_rows
            for item in str(row["missing_required_channels"]).split(",")
            if item
        }
    )
    constant_channels = sorted(
        {
            item
            for row in timeseries_rows
            for item in str(row["constant_channels"]).split(",")
            if item
        }
    )
    capture_values = [float(row["capture_seconds"]) for row in timeseries_rows]
    coverage_values = [float(row["gpu_metric_sampling_coverage"]) for row in sqlite_rows]
    report_lines = [
        "# Pre6G Gemma 4 Part A/B Processing Report",
        "",
        "## Outcome",
        "",
        f"- Workloads in experiment matrix: {len(manifest_rows)}",
        f"- Non-empty RTX5090 marker-free reports: {report_count}",
        f"- Valid exported SQLite databases: {valid_sqlite_count}",
        f"- Valid five-channel 1 ms time series: {valid_timeseries_count}",
        f"- RTX4090-supported workloads: {supported_count}",
        f"- Runtime-regression-eligible paired workloads: {eligible_count}",
        "- G09/G18 are retained as target OOM/unsupported audit rows.",
        "",
        "## Data contract",
        "",
        "- Source profiling device: RTX5090",
        "- Target label device: RTX4090",
        "- Profiling mode: external marker-free Nsight deployment capture",
        "- Resampling interval: 1 ms",
        "- Prefix horizons: 9, 12, and 15 seconds",
        "- Channels: " + ", ".join(CHANNEL_SLUGS),
        "- Workload metadata and source baseline remain audit-only.",
        "",
        "## Quality checks",
        "",
        f"- Missing required semantic channels: {', '.join(missing_channels) if missing_channels else 'none'}",
        f"- Constant required channels: {', '.join(constant_channels) if constant_channels else 'none'}",
        f"- Time-series capture range: {fmt(min(capture_values) if capture_values else math.nan)} to {fmt(max(capture_values) if capture_values else math.nan)} s",
        f"- GPU-metric sampling coverage range: {fmt(min(coverage_values) if coverage_values else math.nan, 4)} to {fmt(max(coverage_values) if coverage_values else math.nan, 4)}",
        "- NVTX row counts are recorded in sqlite_validation.csv and must remain zero.",
        "",
        "## Produced files",
        "",
        "- gemma_manifest.csv",
        "- sqlite_validation.csv",
        "- timeseries_validation.csv",
        "- analysis_cache/pre6g_gemma4_5090_to_4090/sqlite/*.sqlite",
        "- analysis_cache/pre6g_gemma4_5090_to_4090/timeseries_1ms/full/*.npz",
        "- analysis_cache/pre6g_gemma4_5090_to_4090/timeseries_1ms/prefix_{9,12,15}s/*.npz",
        "",
        "## Boundary",
        "",
        "Part A/B does not use target runtime to construct profiler features and does not detect periods yet. "
        "Period detection and Part 8 feature construction belong to the next stage.",
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "part_ab_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "workloads": len(manifest_rows),
        "reports": report_count,
        "valid_sqlite": valid_sqlite_count,
        "valid_timeseries": valid_timeseries_count,
        "runtime_regression_eligible": eligible_count,
        "output_root": str(output_root),
    }, indent=2))


if __name__ == "__main__":
    main()
