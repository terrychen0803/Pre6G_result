# Pre6G Gemma 4 Part A/B Processing Report

## Outcome

- Workloads in experiment matrix: 18
- Non-empty RTX5090 marker-free reports: 18
- Valid exported SQLite databases: 18
- Valid five-channel 1 ms time series: 18
- RTX4090-supported workloads: 16
- Runtime-regression-eligible paired workloads: 16
- G09/G18 are retained as target OOM/unsupported audit rows.

## Data contract

- Source profiling device: RTX5090
- Target label device: RTX4090
- Profiling mode: external marker-free Nsight deployment capture
- Resampling interval: 1 ms
- Prefix horizons: 9, 12, and 15 seconds
- Channels: sm_issue_throughput, sms_active_throughput, tensor_active_throughput, dram_read_bandwidth_throughput, dram_write_bandwidth_throughput
- Workload metadata and source baseline remain audit-only.

## Quality checks

- Missing required semantic channels: none
- Constant required channels: none
- Time-series capture range: 29.999 to 30.038 s
- GPU-metric sampling coverage range: 1.0000 to 1.0001
- NVTX row counts are recorded in sqlite_validation.csv and must remain zero.

## Produced files

- gemma_manifest.csv
- sqlite_validation.csv
- timeseries_validation.csv
- analysis_cache/pre6g_gemma4_5090_to_4090/sqlite/*.sqlite
- analysis_cache/pre6g_gemma4_5090_to_4090/timeseries_1ms/full/*.npz
- analysis_cache/pre6g_gemma4_5090_to_4090/timeseries_1ms/prefix_{9,12,15}s/*.npz

## Boundary

Part A/B does not use target runtime to construct profiler features and does not detect periods yet. Period detection and Part 8 feature construction belong to the next stage.
