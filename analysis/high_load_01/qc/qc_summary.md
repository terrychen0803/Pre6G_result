# High-load GPU-metrics QC and clean alignment

## Scope

Only baseline and `nsys_gpu_metrics_01` runs are included. Trace runs and the C01 2024 legacy profile are excluded by exact directory name.

## Alignment

- Clean/high collection config SHA-256 match: True
- Workloads with matching configuration fields: 24/24
- Clean runs scanned: 96
- High-load runs scanned: 144
- Official high-load GPU-metrics profiles: 24
- Official profiles recorded with Nsight 2026.4.1: 24/24

Clean has three baseline repeats per workload and high-load has five. Comparisons therefore use per-workload distributions/medians, not repeat-number pairing.

## High-load quality gates

- Missing required files: 0
- Incorrect iteration count: 0
- Incorrect steady iteration count: 0
- Workload config mismatches: 0
- Timestamp alignment failures: 0
- Netdata gaps over 2 seconds: 0
- Pre-run mean GPU utilization below 50%: 0
- Strong failure strings in logs: 0

## Known comparability caveats

- Clean iteration CSVs do not contain Unix timestamps and clean runs have no paired Netdata, so current-load matching is available only among high-load runs.
- Clean explicitly requested GPU metric set `gb20x`; high-load left the metric set on automatic selection. SQLite metric descriptors must match before applying the old detector.
- Background contention makes five high-load baselines a runtime distribution, not five interchangeable point estimates. No run is removed solely for being fast or slow.
- GPU-metrics profiling and baseline executions occurred at different times. Overhead is reported against baseline mean/median and, separately, against the closest high-load baseline in pre-run load state.

## Baseline variability overview

- clean: median workload CV 0.89%, P90 2.54%, max 2.81%
- high_load_01: median workload CV 5.96%, P90 10.54%, max 13.84%
