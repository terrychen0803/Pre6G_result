from __future__ import annotations

import argparse
import math
import statistics
from collections import defaultdict
from pathlib import Path

from common import DEFAULT_EXPERIMENTS, DEFAULT_OUTPUT_DIR, discover_runs_roots, iter_run_directories, load_experiments, load_json, write_csv


FIELDS = [
    "device_id", "workload_id", "model", "batch", "imgsz", "amp", "nbs",
    "baseline_rep1_ms", "baseline_rep2_ms", "baseline_rep3_ms", "baseline_repeat_count",
    "baseline_mean_ms", "baseline_median_ms", "baseline_std_ms", "baseline_cv",
    "baseline_min_ms", "baseline_max_ms", "baseline_range_ms", "ground_truth_iter_ms",
    "ground_truth_valid", "warning",
]


def aggregate(runs_roots: list[Path], experiments_path: Path) -> list[dict]:
    experiments = load_experiments(experiments_path)
    values = defaultdict(dict)
    for item in iter_run_directories(runs_roots):
        if item["run_type"] != "baseline":
            continue
        summary, error = load_json(item["run_directory"] / "summary.json")
        value = None if error or not summary else summary.get("steady_window_mean_iter_ms")
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = None
        values[(item["device_id"], item["workload_id"])][item["repeat"]] = value

    rows = []
    for (device, workload), repeats in sorted(values.items()):
        exp = experiments.get(workload, {})
        valid = [repeats.get(i) for i in (1, 2, 3) if repeats.get(i) is not None]
        mean = statistics.fmean(valid) if valid else None
        median = statistics.median(valid) if valid else None
        std = statistics.stdev(valid) if len(valid) >= 2 else None
        cv = std / mean if std is not None and mean else None
        warning = []
        if len(valid) != 3:
            warning.append(f"expected 3 valid repeats, found {len(valid)}")
        if cv is not None and cv > 0.05:
            warning.append("CV > 5%")
        rows.append({
            "device_id": device, "workload_id": workload, "model": exp.get("model"),
            "batch": exp.get("batch"), "imgsz": exp.get("imgsz"), "amp": exp.get("amp"), "nbs": exp.get("nbs"),
            "baseline_rep1_ms": repeats.get(1), "baseline_rep2_ms": repeats.get(2), "baseline_rep3_ms": repeats.get(3),
            "baseline_repeat_count": len(valid), "baseline_mean_ms": mean, "baseline_median_ms": median,
            "baseline_std_ms": std, "baseline_cv": cv, "baseline_min_ms": min(valid) if valid else None,
            "baseline_max_ms": max(valid) if valid else None,
            "baseline_range_ms": max(valid) - min(valid) if valid else None,
            "ground_truth_iter_ms": median, "ground_truth_valid": len(valid) == 3 and math.isfinite(median),
            "warning": "; ".join(warning),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", action="append", type=Path)
    parser.add_argument("--experiments", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    roots = [p.resolve() for p in args.runs_root] if args.runs_root else discover_runs_roots()
    rows = aggregate(roots, args.experiments.resolve())
    write_csv(args.output_dir.resolve() / "ground_truth.csv", rows, FIELDS)
    print(f"Wrote {len(rows)} device/workload ground-truth rows.")


if __name__ == "__main__":
    main()
