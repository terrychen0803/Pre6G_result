# GPU-metrics profiling overhead: clean versus high load

Only `nsys_gpu_metrics_01` is analyzed. Trace profiles are excluded.

## Baseline-median comparison

- Clean median overhead across workloads: 40.71% (mean 32.80%)
- High-load median overhead across workloads: 9.96% (mean 9.87%)
- Clean negative observed overheads: 1/24
- High-load negative observed overheads: 3/24
- High-load profiled runtimes inside the five-baseline range: 9/24

## High-load pre-run GPU-state matched comparison

- Matched-baseline median overhead: 9.91% (mean 12.63%)
- P10/P90: -4.55% / 27.46%
- Negative observed overheads: 5/24

A negative observed value does not imply that profiling accelerates training. It means run-to-run contention variation is larger than the profiler effect for that workload/reference run.
