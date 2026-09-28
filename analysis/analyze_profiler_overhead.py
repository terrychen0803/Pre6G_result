from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from common import DEFAULT_OUTPUT_DIR, discover_runs_roots, iter_run_directories, load_json, percentile, read_csv, write_csv


FIELDS = ["device_id", "workload_id", "profiler_mode", "profile_runtime_ms", "baseline_ground_truth_ms", "overhead_ratio", "overhead_percent", "warning"]
MODES = {"nsys_trace", "nsys_gpu_metrics", "nsys_cpu_metrics", "hpctoolkit_cuda", "hpctoolkit_cuda_trace"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", action="append", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    gt = {(r["device_id"], r["workload_id"]): float(r["ground_truth_iter_ms"]) for r in read_csv(output / "ground_truth.csv") if r["ground_truth_iter_ms"]}
    roots = [p.resolve() for p in args.runs_root] if args.runs_root else discover_runs_roots()
    rows = []
    for item in iter_run_directories(roots):
        if item["run_type"] not in MODES:
            continue
        summary, error = load_json(item["run_directory"] / "summary.json")
        try:
            profile = float((summary or {}).get("steady_window_mean_iter_ms"))
        except (TypeError, ValueError):
            profile = None
        baseline = gt.get((item["device_id"], item["workload_id"]))
        ratio = profile / baseline if profile is not None and baseline else None
        rows.append({"device_id": item["device_id"], "workload_id": item["workload_id"], "profiler_mode": item["run_type"], "profile_runtime_ms": profile, "baseline_ground_truth_ms": baseline, "overhead_ratio": ratio, "overhead_percent": (ratio - 1) * 100 if ratio is not None else None, "warning": error or ("missing runtime or ground truth" if ratio is None else "")})
    write_csv(output / "profiler_overhead.csv", rows, FIELDS)
    lines = ["# Profiler overhead summary", "", "Ratios use each profiler run's steady-window runtime divided by the median of three baseline runs.", "", "| Mode | n | Mean | Median | P90 | Max |", "|---|---:|---:|---:|---:|---:|"]
    for mode in sorted({r["profiler_mode"] for r in rows}):
        vals = [r["overhead_ratio"] for r in rows if r["profiler_mode"] == mode and r["overhead_ratio"] is not None]
        lines.append(f"| {mode} | {len(vals)} | {statistics.fmean(vals):.4f} | {statistics.median(vals):.4f} | {percentile(vals, .9):.4f} | {max(vals):.4f} |")
    (output / "profiler_overhead_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} profiler overhead rows.")


if __name__ == "__main__":
    main()
