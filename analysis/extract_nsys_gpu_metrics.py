from __future__ import annotations

import argparse
import math
import re
import sqlite3
import statistics
from pathlib import Path

from common import DEFAULT_CACHE_DIR, DEFAULT_OUTPUT_DIR, percentile, write_csv


STAT_NAMES = ("mean", "median", "std", "min", "max", "p10", "p25", "p75", "p90", "p95", "cv")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def scale(name: str, value: float) -> float:
    # Nsight SQLite stores frequency in Hz despite the display label [MHz], and
    # fixed-point [Avg] metrics at 1e6 scale. Preserve all other counters as-is.
    if "[MHz]" in name or name.endswith("[Avg]"):
        return value / 1_000_000.0
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR / "nsys")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    rows, long_rows, fields = [], [], {"device_id", "workload_id", "gpu_metrics_available", "gpu_metric_count", "gpu_sample_count", "steady_start_ns", "steady_end_ns", "warning"}
    for path in sorted(args.cache_dir.resolve().glob("*/*/nsys_gpu_metrics/profile.sqlite")):
        device, workload = path.parts[-4], path.parts[-3]
        row = {"device_id": device, "workload_id": workload, "warning": ""}
        con = sqlite3.connect(path)
        try:
            tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
            if not {"GPU_METRICS", "TARGET_INFO_GPU_METRICS", "NVTX_EVENTS"} <= tables:
                row.update(gpu_metrics_available=False, gpu_metric_count=0, gpu_sample_count=0, warning="required GPU metric tables unavailable"); rows.append(row); continue
            window = con.execute("select min(start), max(end) from NVTX_EVENTS where text >= 'TRAIN_ITER_0021' and text <= 'TRAIN_ITER_0128'").fetchone()
            start, end = window
            descriptors = con.execute("select metricId, metricName from TARGET_INFO_GPU_METRICS order by metricId").fetchall()
            row.update(gpu_metrics_available=True, gpu_metric_count=len(descriptors), steady_start_ns=start, steady_end_ns=end)
            sample_counts=[]
            for metric_id, name in descriptors:
                vals = [scale(name, float(r[0])) for r in con.execute("select value from GPU_METRICS where metricId=? and timestamp between ? and ? order by value", (metric_id,start,end))]
                sample_counts.append(len(vals))
                if not vals: continue
                mean, std = statistics.fmean(vals), statistics.stdev(vals) if len(vals)>1 else 0.0
                stats = {"mean": mean, "median": statistics.median(vals), "std": std, "min": vals[0], "max": vals[-1], "p10": percentile(vals,.1), "p25": percentile(vals,.25), "p75": percentile(vals,.75), "p90": percentile(vals,.9), "p95": percentile(vals,.95), "cv": std/mean if mean else None}
                key = slug(name)
                for stat, val in stats.items(): row[f"gpu_{key}_{stat}"] = val; fields.add(f"gpu_{key}_{stat}")
                long_rows.append({"device_id":device,"workload_id":workload,"metric_id":metric_id,"metric_name":name,"sample_count":len(vals),**stats})
            row["gpu_sample_count"] = min(sample_counts) if sample_counts else 0
            rows.append(row)
        finally: con.close()
    base = ["device_id","workload_id","gpu_metrics_available","gpu_metric_count","gpu_sample_count","steady_start_ns","steady_end_ns","warning"]
    write_csv(args.output_dir.resolve()/"nsys_gpu_metrics_features.csv", rows, base+sorted(fields-set(base)))
    write_csv(args.output_dir.resolve()/"nsys_gpu_metrics_long.csv", long_rows, ["device_id","workload_id","metric_id","metric_name","sample_count",*STAT_NAMES])
    print(f"Extracted GPU metrics for {len(rows)} workloads.")


if __name__ == "__main__": main()
