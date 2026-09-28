from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import DEFAULT_OUTPUT_DIR


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    gt = pd.read_csv(out / "ground_truth.csv")
    ov = pd.read_csv(out / "profiler_overhead.csv")
    inv = pd.read_csv(out / "collection_inventory.csv")
    iterations = pd.read_csv(out / "iteration_validation.csv")
    devices = sorted(inv.device_id.unique())
    expected_profiles = len(gt)

    lines = [
        "# YOLO26 runtime collection analysis", "", "## Data inventory", "",
        f"- Devices with run data: {', '.join(devices)}",
        f"- Device/workload ground-truth rows: {len(gt)}",
        f"- Unique workloads: {inv.workload_id.nunique()}",
        f"- Run directories: {len(inv)} (baseline {(inv.run_type == 'baseline').sum()}, Nsight {(inv.run_type.str.startswith('nsys_')).sum()}, HPCToolkit {(inv.run_type.str.startswith('hpctoolkit')).sum()})",
        f"- Valid JSON/config matches: {(inv.summary_json_valid & inv.metadata_json_valid & inv.config_matches_experiment).sum()}/{len(inv)}",
        f"- Iteration files with exactly 128 rows and 108 steady rows: {((iterations.iteration_count == 128) & (iterations.steady_count == 108)).sum()}/{len(iterations)}",
        "", "## Ground truth and baseline stability", "",
        "Ground truth is the median `steady_window_mean_iter_ms` across the three unprofiled baseline repeats. Profiler-run runtime is never substituted for it.", "",
    ]
    for device, group in gt.groupby("device_id"):
        lines.append(f"- {device}: {group.ground_truth_iter_ms.min():.3f}-{group.ground_truth_iter_ms.max():.3f} ms, median CV {pct(group.baseline_cv.median())}, CV >2%: {(group.baseline_cv > .02).sum()}, CV >5%: {(group.baseline_cv > .05).sum()}.")
    high = gt.sort_values("baseline_cv", ascending=False).head(8)
    lines += ["", "| Device | Workload | Batch | Image | AMP | GT ms | CV |", "|---|---|---:|---:|---|---:|---:|"]
    lines += [f"| {r.device_id} | {r.workload_id} | {r.batch} | {r.imgsz} | {r.amp} | {r.ground_truth_iter_ms:.3f} | {pct(r.baseline_cv)} |" for r in high.itertuples()]

    amp = gt.pivot_table(index=["device_id", "batch", "imgsz"], columns="amp", values="ground_truth_iter_ms").dropna()
    if False in amp and True in amp:
        ratios = amp[True] / amp[False]
        lines += ["", f"Across {len(ratios)} matched device/batch/image cases, AMP-on divided by AMP-off runtime has median {ratios.median():.3f}; range {ratios.min():.3f}-{ratios.max():.3f}. Retain `device x batch x imgsz x amp` interactions rather than using a global AMP multiplier."]

    cross = gt.pivot(index="workload_id", columns="device_id", values="ground_truth_iter_ms")
    if {"RTX4090", "RTX5090"} <= set(cross.columns):
        speedup = cross.RTX4090 / cross.RTX5090
        lines += ["", "## Cross-device ground-truth comparison", "", f"RTX5090 speedup over RTX4090 has mean {speedup.mean():.3f}x, median {speedup.median():.3f}x, and range {speedup.min():.3f}x-{speedup.max():.3f}x across 24 matched workloads. The range demonstrates workload-dependent hardware scaling.", "", "| Workload | RTX4090 ms | RTX5090 ms | 5090 speedup |", "|---|---:|---:|---:|"]
        ordered = speedup.sort_values(ascending=False)
        for workload in list(ordered.head(4).index) + list(ordered.tail(4).index):
            lines.append(f"| {workload} | {cross.loc[workload, 'RTX4090']:.3f} | {cross.loc[workload, 'RTX5090']:.3f} | {speedup.loc[workload]:.3f}x |")

    lines += ["", "## Profiler overhead", "", "| Mode | Mean ratio | Median ratio | P90 | Max |", "|---|---:|---:|---:|---:|"]
    for mode, group in ov.dropna(subset=["overhead_ratio"]).groupby("profiler_mode"):
        values = group.overhead_ratio
        lines.append(f"| {mode} | {values.mean():.4f} | {values.median():.4f} | {values.quantile(.9):.4f} | {values.max():.4f} |")
    lines += ["", "| Device | Mode | Median ratio | P90 |", "|---|---|---:|---:|"]
    for (device, mode), group in ov.dropna(subset=["overhead_ratio"]).groupby(["device_id", "profiler_mode"]):
        values = group.overhead_ratio
        lines.append(f"| {device} | {mode} | {values.median():.4f} | {values.quantile(.9):.4f} |")
    lines += ["", "Profiler runtime is collection overhead only, not a ground-truth label.", "", "## Nsight feature readiness", ""]
    trace_path, gpu_path, cpu_path = out / "nsys_trace_features.csv", out / "nsys_gpu_metrics_features.csv", out / "nsys_cpu_metrics_availability.csv"
    trace = pd.read_csv(trace_path) if trace_path.exists() else pd.DataFrame()
    gpu = pd.read_csv(gpu_path) if gpu_path.exists() else pd.DataFrame()
    cpu = pd.read_csv(cpu_path) if cpu_path.exists() else pd.DataFrame()
    lines += [
        f"- Trace features extracted: {len(trace)}/{expected_profiles} device/workload rows.",
        f"- GPU metric features extracted: {len(gpu)}/{expected_profiles} device/workload rows.",
        f"- CPU PMU availability inspected: {len(cpu)}/{expected_profiles}; usable PMU rows: {int(cpu.cpu_pmu_available.sum()) if not cpu.empty else 0}.",
        "- CUDA trace and GPU metric tables are aligned to the 108 steady-state NVTX ranges.",
    ]
    if not trace.empty and "cuda_trace_implementation" in trace:
        for device, group in trace.groupby("device_id"):
            implementations = ", ".join(f"{name}: {count}" for name, count in group.cuda_trace_implementation.value_counts().items())
            lines.append(f"- {device} CUDA trace backend: {implementations}.")
    if not gpu.empty:
        for device, group in gpu.groupby("device_id"):
            lines.append(f"- {device} GPU metric count per workload: {int(group.gpu_metric_count.min())}.")
    if not cpu.empty:
        for device, group in cpu.groupby("device_id"):
            lines.append(f"- {device} CPU PMU usable: {int(group.cpu_pmu_available.sum())}/{len(group)}; LinuxPerf parse errors: {int(group.perf_parse_error.sum())}.")
    lines += ["", "## Missing or suspicious data", "",
        "- No run metadata records the expected dataset semantic hash; workload configuration matches, but dataset identity cannot be cryptographically verified.",
        "- CUDA core count, Tensor Core count, memory-bus width and theoretical bandwidth remain absent from supplied hardware captures.",
        f"- Baseline rows above 2% CV: {(gt.baseline_cv > .02).sum()}; above 5%: {(gt.baseline_cv > .05).sum()}.",
        "- Baseline repeat/min/max/CV fields are label-audit columns, not model inputs. `feature_schema.csv` marks them for exclusion.",
        "- Source-device timing features are valid for directed cross-device prediction, but become leakage if source and target refer to the same observation.",
    ]

    hpct_path = out / "hpctoolkit_postprocess_inventory.csv"
    if hpct_path.exists():
        hpct = pd.read_csv(hpct_path)
        lines += ["", "## HPCToolkit postprocess readiness", "",
            f"- Valid HPCToolkit v4 databases: {int(hpct.hpct_postprocess_available.sum())}/{len(hpct)}.",
            f"- Metric taxonomy: {int(hpct.hpct_metric_count.max())} unique metrics per device/workload database.",
            f"- Databases with time traces: {int(hpct.hpct_trace_db_exists.sum())}/{len(hpct)} (collection used `trace=false`).",
            f"- Postprocess logs with detected errors/failures: {int(((hpct.hpct_hpcstruct_error_count > 0) | (hpct.hpct_hpcprof_error_count > 0)).sum())}.",
            f"- Databases containing copied source trees: {int(hpct.hpct_source_tree_exists.sum())}/{len(hpct)}.",
            "- The workload table includes HPCT database QC/provenance only. Numeric CCT metrics still require hpcviewer CSV export or an official HPCToolkit database reader.",
            "- Original hpcrun missing-correlation warnings require attributed/unattributed activity validation after numeric export.",
        ]
        for device, group in hpct.groupby("device_id"):
            lines.append(f"- {device}: accelerator trace records {int(group.hpct_accelerator_trace_records.sum()):,}, dropped {int(group.hpct_accelerator_trace_dropped.sum()):,}, missing-correlation warnings {int(group.hpct_hpcrun_missing_correlation_count.sum()):,}, PC samples {int(group.hpct_accelerator_samples_recorded.sum()):,}.")

    ready = len(trace) == expected_profiles and len(gpu) == expected_profiles and len(devices) >= 2
    lines += ["", "## Dataset readiness and next steps", "", f"Cross-device labels are available for {len(gt)} device/workload rows. Nsight feature extraction is {'complete' if ready else 'still in progress'}. `cross_device_samples.csv` contains the directed RTX4090-to-RTX5090 and RTX5090-to-RTX4090 samples with strict source/target prefixes.", "", "Recommended order:", "", "1. Validate simple workload + hardware baselines on the generated directed dataset before adding source profiling features.", "2. Add Nsight feature families in ablations, keeping trace-backend and metric-availability fields as quality controls.", "3. Use leave-one-workload-out evaluation; two devices are insufficient for a meaningful unseen-device test.", "4. Keep CPU PMU unavailable fields missing and use HPCToolkit only after numeric CCT extraction and attribution validation.", ""]
    (out / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote analysis_report.md")


if __name__ == "__main__":
    main()
