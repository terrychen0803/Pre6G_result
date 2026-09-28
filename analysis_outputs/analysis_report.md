# YOLO26 runtime collection analysis

## Data inventory

- Devices with run data: RTX4090, RTX5090
- Device/workload ground-truth rows: 48
- Unique workloads: 24
- Run directories: 336 (baseline 144, Nsight 144, HPCToolkit 48)
- Valid JSON/config matches: 336/336
- Iteration files with exactly 128 rows and 108 steady rows: 336/336

## Ground truth and baseline stability

Ground truth is the median `steady_window_mean_iter_ms` across the three unprofiled baseline repeats. Profiler-run runtime is never substituted for it.

- RTX4090: 30.278-131.249 ms, median CV 0.97%, CV >2%: 9, CV >5%: 1.
- RTX5090: 28.075-104.224 ms, median CV 0.89%, CV >2%: 5, CV >5%: 0.

| Device | Workload | Batch | Image | AMP | GT ms | CV |
|---|---|---:|---:|---|---:|---:|
| RTX4090 | C17 | 4 | 480 | True | 38.002 | 5.06% |
| RTX4090 | C14 | 8 | 320 | True | 35.024 | 4.96% |
| RTX4090 | C13 | 4 | 320 | True | 33.878 | 4.77% |
| RTX4090 | C22 | 8 | 640 | True | 45.275 | 4.58% |
| RTX4090 | C05 | 4 | 480 | False | 32.110 | 2.99% |
| RTX5090 | C01 | 4 | 320 | False | 29.060 | 2.81% |
| RTX5090 | C17 | 4 | 480 | True | 31.537 | 2.75% |
| RTX5090 | C15 | 16 | 320 | True | 33.221 | 2.55% |

Across 24 matched device/batch/image cases, AMP-on divided by AMP-off runtime has median 1.048; range 0.770-1.183. Retain `device x batch x imgsz x amp` interactions rather than using a global AMP multiplier.

## Cross-device ground-truth comparison

RTX5090 speedup over RTX4090 has mean 1.119x, median 1.104x, and range 1.036x-1.259x across 24 matched workloads. The range demonstrates workload-dependent hardware scaling.

| Workload | RTX4090 ms | RTX5090 ms | 5090 speedup |
|---|---:|---:|---:|
| C12 | 131.249 | 104.224 | 1.259x |
| C10 | 49.200 | 39.302 | 1.252x |
| C17 | 38.002 | 31.537 | 1.205x |
| C11 | 73.354 | 61.916 | 1.185x |
| C19 | 45.052 | 42.621 | 1.057x |
| C01 | 30.278 | 29.060 | 1.042x |
| C21 | 35.460 | 34.129 | 1.039x |
| C09 | 31.867 | 30.749 | 1.036x |

## Profiler overhead

| Mode | Mean ratio | Median ratio | P90 | Max |
|---|---:|---:|---:|---:|
| hpctoolkit_cuda | 2.9123 | 3.1696 | 3.6799 | 3.7696 |
| nsys_cpu_metrics | 1.3365 | 1.4032 | 1.5368 | 1.5952 |
| nsys_gpu_metrics | 1.3309 | 1.3928 | 1.5203 | 1.5877 |
| nsys_trace | 1.3240 | 1.3532 | 1.5087 | 1.5848 |

| Device | Mode | Median ratio | P90 |
|---|---|---:|---:|
| RTX4090 | hpctoolkit_cuda | 3.1696 | 3.6908 |
| RTX4090 | nsys_cpu_metrics | 1.4032 | 1.5081 |
| RTX4090 | nsys_gpu_metrics | 1.3886 | 1.5143 |
| RTX4090 | nsys_trace | 1.3431 | 1.5363 |
| RTX5090 | hpctoolkit_cuda | 3.1401 | 3.6671 |
| RTX5090 | nsys_cpu_metrics | 1.3973 | 1.5391 |
| RTX5090 | nsys_gpu_metrics | 1.4071 | 1.5212 |
| RTX5090 | nsys_trace | 1.3851 | 1.4830 |

Profiler runtime is collection overhead only, not a ground-truth label.

## Nsight feature readiness

- Trace features extracted: 48/48 device/workload rows.
- GPU metric features extracted: 48/48 device/workload rows.
- CPU PMU availability inspected: 48/48; usable PMU rows: 0.
- CUDA trace and GPU metric tables are aligned to the 108 steady-state NVTX ranges.
- RTX4090 CUDA trace backend: legacy_software: 24.
- RTX5090 CUDA trace backend: hardware: 24.
- RTX4090 GPU metric count per workload: 35.
- RTX5090 GPU metric count per workload: 25.
- RTX4090 CPU PMU usable: 0/24; LinuxPerf parse errors: 24.
- RTX5090 CPU PMU usable: 0/24; LinuxPerf parse errors: 0.

## Missing or suspicious data

- No run metadata records the expected dataset semantic hash; workload configuration matches, but dataset identity cannot be cryptographically verified.
- CUDA core count, Tensor Core count, memory-bus width and theoretical bandwidth remain absent from supplied hardware captures.
- Baseline rows above 2% CV: 14; above 5%: 1.
- Baseline repeat/min/max/CV fields are label-audit columns, not model inputs. `feature_schema.csv` marks them for exclusion.
- Source-device timing features are valid for directed cross-device prediction, but become leakage if source and target refer to the same observation.

## HPCToolkit postprocess readiness

- Valid HPCToolkit v4 databases: 48/48.
- Metric taxonomy: 65 unique metrics per device/workload database.
- Databases with time traces: 0/48 (collection used `trace=false`).
- Postprocess logs with detected errors/failures: 0.
- Databases containing copied source trees: 0/48.
- The workload table includes HPCT database QC/provenance only. Numeric CCT metrics still require hpcviewer CSV export or an official HPCToolkit database reader.
- Original hpcrun missing-correlation warnings require attributed/unattributed activity validation after numeric export.
- RTX4090: accelerator trace records 245,805,822, dropped 0, missing-correlation warnings 173,034, PC samples 0.
- RTX5090: accelerator trace records 242,476,971, dropped 0, missing-correlation warnings 233,674, PC samples 0.

## Dataset readiness and next steps

Cross-device labels are available for 48 device/workload rows. Nsight feature extraction is complete. `cross_device_samples.csv` contains the directed RTX4090-to-RTX5090 and RTX5090-to-RTX4090 samples with strict source/target prefixes.

Recommended order:

1. Validate simple workload + hardware baselines on the generated directed dataset before adding source profiling features.
2. Add Nsight feature families in ablations, keeping trace-backend and metric-availability fields as quality controls.
3. Use leave-one-workload-out evaluation; two devices are insufficient for a meaningful unseen-device test.
4. Keep CPU PMU unavailable fields missing and use HPCToolkit only after numeric CCT extraction and attribution validation.
