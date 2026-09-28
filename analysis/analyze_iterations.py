from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from common import DEFAULT_OUTPUT_DIR, discover_runs_roots, iter_run_directories, load_json, parse_bool, percentile, write_csv


FIELDS = ["device_id", "workload_id", "run_type", "repeat", "iteration_count", "steady_count", "steady_ids_contiguous", "actual_batch_matches", "host_mean_ms", "host_median_ms", "host_std_ms", "host_p90_ms", "host_p95_ms", "host_min_ms", "host_max_ms", "gpu_mean_ms", "gpu_median_ms", "gpu_std_ms", "gpu_p90_ms", "gpu_p95_ms", "gpu_min_ms", "gpu_max_ms", "summary_steady_window_mean_ms", "summary_minus_host_mean_ms", "warning"]


def stats(values):
    return {"mean": statistics.fmean(values), "median": statistics.median(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0, "p90": percentile(values, .9), "p95": percentile(values, .95), "min": min(values), "max": max(values)} if values else {k: None for k in ("mean", "median", "std", "p90", "p95", "min", "max")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", action="append", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    roots = [p.resolve() for p in args.runs_root] if args.runs_root else discover_runs_roots()
    rows = []
    for item in iter_run_directories(roots):
        path = item["run_directory"] / "iterations.csv"
        warning = []
        try:
            with path.open(newline="", encoding="utf-8-sig") as f: data = list(csv.DictReader(f))
        except Exception as exc:
            data = []; warning.append(f"unreadable iterations.csv: {exc}")
        steady = [r for r in data if parse_bool(r.get("warmup")) is False]
        ids = [int(r["iteration_id"]) for r in steady]
        contiguous = ids == list(range(21, 129))
        batch_matches = all(r.get("actual_batch") == r.get("requested_batch") for r in data)
        if len(data) != 128: warning.append(f"expected 128 rows, found {len(data)}")
        if len(steady) != 108: warning.append(f"expected 108 steady rows, found {len(steady)}")
        if not contiguous: warning.append("steady iteration IDs are not 21..128")
        if not batch_matches: warning.append("actual batch differs from requested batch")
        def nums(key):
            out=[]
            for r in steady:
                try: out.append(float(r[key]))
                except (KeyError, TypeError, ValueError): pass
            return out
        h, g = stats(nums("host_callback_ms")), stats(nums("gpu_ms"))
        summary, _ = load_json(item["run_directory"] / "summary.json")
        try: sw = float((summary or {}).get("steady_window_mean_iter_ms"))
        except (TypeError, ValueError): sw = None
        row = {"device_id": item["device_id"], "workload_id": item["workload_id"], "run_type": item["run_type"], "repeat": item["repeat"], "iteration_count": len(data), "steady_count": len(steady), "steady_ids_contiguous": contiguous, "actual_batch_matches": batch_matches, "summary_steady_window_mean_ms": sw, "summary_minus_host_mean_ms": sw-h["mean"] if sw is not None and h["mean"] is not None else None, "warning": "; ".join(warning)}
        for prefix, value in (("host", h), ("gpu", g)):
            for key, val in value.items(): row[f"{prefix}_{key}_ms"] = val
        rows.append(row)
    write_csv(args.output_dir.resolve() / "iteration_validation.csv", rows, FIELDS)
    print(f"Validated iteration data for {len(rows)} runs.")


if __name__ == "__main__":
    main()
