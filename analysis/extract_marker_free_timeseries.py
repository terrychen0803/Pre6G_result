from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from common import DEFAULT_CACHE_DIR, DEFAULT_OUTPUT_DIR


CHANNEL_SLUGS = (
    "sm_issue_throughput",
    "sms_active_throughput",
    "tensor_active_throughput",
    "dram_read_bandwidth_throughput",
    "dram_write_bandwidth_throughput",
)
BASE_BIN_MS = 1


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache marker-free GPU metric time series on a uniform clock."
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR / "nsys")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--timeseries-cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR / "marker_free_timeseries",
        help="Directory for generated NPZ files (kept separate across conditions).",
    )
    args = parser.parse_args()
    source_paths = sorted(
        args.cache_dir.resolve().glob("*/*/nsys_gpu_metrics/profile.sqlite")
    )
    cache = args.timeseries_cache_dir.resolve()
    rows = []
    for index, path in enumerate(source_paths, start=1):
        device, workload = path.parts[-4], path.parts[-3]
        connection = sqlite3.connect(path)
        try:
            descriptors = {
                slug(name): (int(metric_id), str(name))
                for metric_id, name in connection.execute(
                    "select metricId,metricName from TARGET_INFO_GPU_METRICS"
                )
            }
            missing = [channel for channel in CHANNEL_SLUGS if channel not in descriptors]
            if missing:
                raise ValueError(f"Missing channels in {path}: {missing}")
            raw = {}
            global_start, global_end = None, None
            for channel in CHANNEL_SLUGS:
                metric_id, _ = descriptors[channel]
                samples = connection.execute(
                    "select timestamp,value from GPU_METRICS "
                    "where metricId=? order by timestamp",
                    (metric_id,),
                ).fetchall()
                timestamps = np.asarray([sample[0] for sample in samples], dtype=np.int64)
                values = np.asarray([sample[1] for sample in samples], dtype=float)
                if not len(timestamps):
                    raise ValueError(f"Empty channel {channel} in {path}")
                raw[channel] = (timestamps, values)
                global_start = int(timestamps[0]) if global_start is None else max(global_start, int(timestamps[0]))
                global_end = int(timestamps[-1]) if global_end is None else min(global_end, int(timestamps[-1]))
            step_ns = int(BASE_BIN_MS * 1e6)
            grid_ns = np.arange(global_start, global_end + 1, step_ns, dtype=np.int64)
            matrix = []
            for channel in CHANNEL_SLUGS:
                timestamps, values = raw[channel]
                matrix.append(np.interp(grid_ns, timestamps, values))
            values = np.asarray(matrix, dtype=np.float32)
            relative_time_ms = ((grid_ns - grid_ns[0]) / 1e6).astype(np.float32)
            output_path = cache / device / f"{workload}.npz"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output_path,
                time_ms=relative_time_ms,
                values=values,
                channels=np.asarray(CHANNEL_SLUGS),
                base_bin_ms=np.asarray(BASE_BIN_MS),
            )
            rows.append(
                {
                    "device_id": device,
                    "workload_id": workload,
                    "cache_path": str(output_path),
                    "channel_count": len(CHANNEL_SLUGS),
                    "sample_count": len(grid_ns),
                    "capture_seconds": float(relative_time_ms[-1] / 1000),
                    "base_bin_ms": BASE_BIN_MS,
                }
            )
        finally:
            connection.close()
        print(f"[{index}/{len(source_paths)}] {device}/{workload}", flush=True)
    manifest = pd.DataFrame(rows).sort_values(["device_id", "workload_id"])
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "marker_free_timeseries_manifest.csv"
    manifest.to_csv(output, index=False)
    print(f"Wrote {len(manifest)} marker-free time-series entries to {output}.")


if __name__ == "__main__":
    main()
