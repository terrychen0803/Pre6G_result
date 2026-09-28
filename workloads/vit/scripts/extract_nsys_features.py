from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from common import DEFAULT_EXPERIMENTS, DEFAULT_RUNS_ROOT, load_workloads, parse_workload_ids, print_command, sanitize_name
from marker_free_features import extract_marker_free_features


def quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def tables(connection: sqlite3.Connection) -> list[str]:
    return [str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]


def columns(connection: sqlite3.Connection, table: str) -> dict[str, str]:
    return {str(row[1]).lower(): str(row[1]) for row in connection.execute(f"PRAGMA table_info({quote(table)})")}


def interval_rows(connection: sqlite3.Connection, names: list[str], token: str) -> list[tuple[float, float, float]]:
    result: list[tuple[float, float, float]] = []
    for table in names:
        if token not in table.upper():
            continue
        available = columns(connection, table)
        start, end = available.get("start"), available.get("end")
        if not start or not end:
            continue
        byte_column = available.get("bytes")
        byte_expression = quote(byte_column) if byte_column else "0"
        query = f"SELECT {quote(start)}, {quote(end)}, {byte_expression} FROM {quote(table)} ORDER BY {quote(start)}"
        for start_ns, end_ns, byte_count in connection.execute(query):
            result.append((float(start_ns), float(end_ns), float(byte_count or 0)))
    return result


def aggregate(rows: list[tuple[float, float, float]]) -> dict[str, float]:
    durations = [end - start for start, end, _ in rows if end > start]
    return {
        "count": float(len(rows)),
        "total_ms": float(sum(durations) / 1e6),
        "mean_ms": float(np.mean(durations) / 1e6) if durations else 0.0,
        "max_ms": float(max(durations) / 1e6) if durations else 0.0,
        "total_bytes": float(sum(item[2] for item in rows)),
    }


def bin_intervals(rows, grid_start: float, count: int, step_ns: float):
    active = np.zeros(count)
    starts = np.zeros(count)
    byte_counts = np.zeros(count)
    for start, end, byte_count in rows:
        if end <= start:
            continue
        first = max(0, int((start - grid_start) // step_ns))
        last = min(count - 1, int((end - grid_start) // step_ns))
        if first >= count or last < 0:
            continue
        starts[first] += 1.0
        byte_counts[first] += byte_count
        for index in range(first, last + 1):
            bin_start = grid_start + index * step_ns
            overlap = max(0.0, min(end, bin_start + step_ns) - max(start, bin_start))
            active[index] += overlap / step_ns
    return active, starts, byte_counts


def semantic_metric_name(table: str, column: str) -> str | None:
    text = f"{table} {column}".lower().replace("_", " ")
    if "tensor" in text and ("active" in text or "util" in text):
        return "profile.gpu.tensor_active"
    if ("sm" in text or "streaming multiprocessor" in text) and ("active" in text or "util" in text):
        return "profile.gpu.sm_active"
    if "sm" in text and "issue" in text:
        return "profile.gpu.sm_issue"
    if ("dram" in text or "memory" in text) and "read" in text:
        return "profile.gpu.dram_read"
    if ("dram" in text or "memory" in text) and "write" in text:
        return "profile.gpu.dram_write"
    if ("dram" in text or "memory" in text) and ("active" in text or "util" in text):
        return "profile.gpu.memory_active"
    if "dram" in text and ("bandwidth" in text or "throughput" in text):
        return "profile.gpu.dram_throughput"
    if ("gr active" in text or "graphics active" in text):
        return "profile.gpu.compute_active"
    if "pcie" in text and "read" in text:
        return "profile.gpu.pcie_read"
    if "pcie" in text and "write" in text:
        return "profile.gpu.pcie_write"
    return None


def numeric_metric_value(value) -> float | None:
    try:
        return float(str(value).strip().rstrip("%").replace(",", ""))
    except (TypeError, ValueError):
        return None


def metric_signals(connection: sqlite3.Connection, names: list[str], grid: np.ndarray) -> dict[str, np.ndarray]:
    candidates: dict[str, list[np.ndarray]] = {}
    table_lookup = {name.upper(): name for name in names}

    # Current documented schema: metric names/IDs and sampled values are split.
    if "GPU_METRICS" in table_lookup and "TARGET_INFO_GPU_METRICS" in table_lookup:
        metrics_table = quote(table_lookup["GPU_METRICS"])
        info_table = quote(table_lookup["TARGET_INFO_GPU_METRICS"])
        query = (
            f"SELECT g.timestamp, i.metricName, g.value FROM {metrics_table} g "
            f"JOIN {info_table} i ON g.typeId=i.typeId AND g.metricId=i.metricId "
            "ORDER BY g.timestamp"
        )
        grouped: dict[str, list[tuple[float, float]]] = {}
        for timestamp, metric_name, value in connection.execute(query):
            semantic = semantic_metric_name("GPU_METRICS", str(metric_name))
            numeric = numeric_metric_value(value)
            if semantic and numeric is not None:
                grouped.setdefault(semantic, []).append((float(timestamp), numeric))
        for semantic, samples in grouped.items():
            candidates.setdefault(semantic, []).append(
                np.interp(
                    grid,
                    np.asarray([item[0] for item in samples]),
                    np.asarray([item[1] for item in samples]),
                )
            )

    # Older exports place GPU metric fields in GENERIC_EVENTS.data JSON.
    if "GENERIC_EVENTS" in table_lookup:
        generic_table = quote(table_lookup["GENERIC_EVENTS"])
        grouped_generic: dict[str, list[tuple[float, float]]] = {}
        for timestamp, encoded in connection.execute(
            f"SELECT timestamp, data FROM {generic_table} ORDER BY timestamp"
        ):
            try:
                payload = json.loads(encoded)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            for metric_name, value in payload.items():
                semantic = semantic_metric_name("GENERIC_EVENTS", str(metric_name))
                numeric = numeric_metric_value(value)
                if semantic and numeric is not None:
                    grouped_generic.setdefault(semantic, []).append(
                        (float(timestamp), numeric)
                    )
        for semantic, samples in grouped_generic.items():
            if len(samples) >= 4:
                candidates.setdefault(semantic, []).append(
                    np.interp(
                        grid,
                        np.asarray([item[0] for item in samples]),
                        np.asarray([item[1] for item in samples]),
                    )
                )

    # Best-effort support for releases exporting numeric wide metric tables.
    for table in names:
        if "METRIC" not in table.upper():
            continue
        if table.upper() in {"GPU_METRICS", "TARGET_INFO_GPU_METRICS"}:
            continue
        available = columns(connection, table)
        timestamp = next((available.get(name) for name in ("timestamp", "time", "start") if available.get(name)), None)
        if timestamp is None:
            continue
        for lowered, original in available.items():
            semantic = semantic_metric_name(table, original)
            if semantic is None or original == timestamp or lowered.endswith("id"):
                continue
            query = f"SELECT {quote(timestamp)}, {quote(original)} FROM {quote(table)} WHERE {quote(original)} IS NOT NULL ORDER BY 1"
            samples: list[tuple[float, float]] = []
            try:
                for timestamp_value, metric_value in connection.execute(query):
                    samples.append((float(timestamp_value), float(metric_value)))
            except (TypeError, ValueError, sqlite3.Error):
                continue
            if len(samples) < 4:
                continue
            sample_times = np.asarray([item[0] for item in samples])
            sample_values = np.asarray([item[1] for item in samples])
            if np.std(sample_values) <= 1e-12:
                continue
            candidates.setdefault(semantic, []).append(np.interp(grid, sample_times, sample_values))
    return {name: np.mean(values, axis=0) for name, values in candidates.items()}


def extract_timeseries(connection: sqlite3.Connection, interval_ms: float):
    names = tables(connection)
    kernels = interval_rows(connection, names, "CUPTI_ACTIVITY_KIND_KERNEL")
    memcopies = interval_rows(connection, names, "CUPTI_ACTIVITY_KIND_MEMCPY")
    combined = kernels + memcopies
    if not combined:
        raise ValueError("No CUDA kernel/memcpy intervals found")
    start = min(item[0] for item in combined)
    end = max(item[1] for item in combined)
    step_ns = interval_ms * 1e6
    count = int(np.ceil((end - start) / step_ns)) + 1
    grid = start + np.arange(count) * step_ns
    kernel_active, kernel_starts, _ = bin_intervals(kernels, start, count, step_ns)
    memcpy_active, _, memcpy_bytes = bin_intervals(memcopies, start, count, step_ns)
    signals = {
        "profile.gpu.kernel_active": kernel_active,
        "profile.gpu.kernel_launch_rate": kernel_starts,
        "profile.gpu.memcpy_active": memcpy_active,
        "profile.gpu.memcpy_bytes": memcpy_bytes,
    }
    signals.update(metric_signals(connection, names, grid))
    return grid, signals, {"kernel": aggregate(kernels), "memcpy": aggregate(memcopies)}


def write_csv(path: Path, timestamps, signals) -> None:
    names = sorted(signals)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp_ns", *names])
        for index, timestamp in enumerate(timestamps):
            writer.writerow([int(timestamp), *(signals[name][index] for name in names)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract semantic marker-free cycle features")
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--workloads", default="all")
    parser.add_argument("--collection-mode", choices=["deployment", "oracle"], default="deployment")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--sample-interval-ms", type=float, default=1)
    parser.add_argument("--min-period-ms", type=float, default=1)
    parser.add_argument("--max-period-ms", type=float, default=10000.0)
    parser.add_argument("--experiments", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--force-export", action="store_true")
    args = parser.parse_args()
    if shutil.which("nsys") is None:
        raise SystemExit("nsys was not found")

    workloads = load_workloads(args.experiments.resolve())
    workload_ids = parse_workload_ids(args.workloads, set(workloads))
    device_label = sanitize_name(args.device_id)
    for workload_id in workload_ids:
        run_dir = args.runs_root.resolve() / device_label / workload_id / f"nsys_{args.collection_mode}_{args.repeat:02d}"
        report = run_dir / "profile.nsys-rep"
        database = run_dir / "profile.sqlite"
        if not report.exists():
            raise FileNotFoundError(report)
        if args.force_export or not database.exists():
            command = ["nsys", "export", "--type", "sqlite", "--force-overwrite", "true", "--output", str(database), str(report)]
            print_command(command)
            if subprocess.run(command, check=False).returncode != 0:
                raise SystemExit("Nsight export failed")
        connection = sqlite3.connect(database)
        try:
            timestamps, signals, diagnostics = extract_timeseries(connection, args.sample_interval_ms)
        finally:
            connection.close()
        deployment_features = extract_marker_free_features(
            timestamps, signals, args.sample_interval_ms, args.min_period_ms, args.max_period_ms
        )
        result: dict[str, Any] = {
            "schema_version": 1,
            "device_id": device_label,
            "workload_id": workload_id,
            "collection_mode": args.collection_mode,
            "repeat": args.repeat,
            "deployment_features": deployment_features,
            "audit_diagnostics": diagnostics,
        }
        write_csv(run_dir / "marker_free_timeseries.csv", timestamps, signals)
        (run_dir / "profile_features.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote {run_dir / 'profile_features.json'}")


if __name__ == "__main__":
    main()
