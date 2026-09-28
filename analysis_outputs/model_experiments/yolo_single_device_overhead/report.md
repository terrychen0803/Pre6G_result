# YOLO single-device profiling-overhead prediction

This experiment predicts Nsight GPU Metrics overhead independently for each known device. Inputs are marker-free profiling period, detector quality, and GPU metric cycle behavior. The dry-run baseline and profiler steady-window runtime are labels only.

Outer evaluation is leave-one-workload-out within each device. Ridge alpha and the primary nested feature family are selected using only inner grouped folds.

Overhead error is reported in percentage points (pp), because percentage error of the overhead itself is unstable near zero. Estimated dry runtime uses `detected_period_ms / predicted_overhead_ratio`.

## Outer leave-one-workload-out results

| Device | Model | N | Overhead MAE | Median AE | P90 AE | Max AE | Bias | Estimated dry-runtime MAPE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| RTX4090 | period_only_ridge | 24 | 8.55 pp | 7.27 pp | 18.54 pp | 21.68 pp | -0.20 pp | 6.72% |
| RTX4090 | period_quality_ridge | 24 | 8.74 pp | 7.21 pp | 16.07 pp | 22.54 pp | -0.14 pp | 6.53% |
| RTX4090 | period_compact_ridge | 24 | 8.81 pp | 7.42 pp | 18.47 pp | 23.68 pp | +0.40 pp | 6.75% |
| RTX4090 | constant_median_added_ms | 24 | 9.26 pp | 8.58 pp | 17.64 pp | 21.06 pp | +1.41 pp | 7.51% |
| RTX4090 | period_full_ridge | 24 | 9.68 pp | 7.65 pp | 17.37 pp | 32.99 pp | +0.28 pp | 6.89% |
| RTX4090 | behavior_only_ridge | 24 | 10.00 pp | 7.92 pp | 17.62 pp | 33.17 pp | +0.52 pp | 7.07% |
| RTX4090 | nested_selected_ridge | 24 | 10.71 pp | 9.11 pp | 21.26 pp | 33.17 pp | +0.70 pp | 7.81% |
| RTX4090 | constant_median_ratio | 24 | 16.68 pp | 13.98 pp | 32.94 pp | 36.45 pp | +5.49 pp | 11.45% |
| RTX5090 | period_full_ridge | 24 | 5.40 pp | 5.22 pp | 10.09 pp | 12.73 pp | +0.22 pp | 4.70% |
| RTX5090 | period_compact_ridge | 24 | 5.53 pp | 4.42 pp | 10.53 pp | 14.41 pp | -0.16 pp | 4.85% |
| RTX5090 | behavior_only_ridge | 24 | 5.95 pp | 4.86 pp | 12.30 pp | 16.29 pp | -0.02 pp | 4.86% |
| RTX5090 | nested_selected_ridge | 24 | 6.66 pp | 5.42 pp | 12.62 pp | 16.01 pp | -0.12 pp | 4.98% |
| RTX5090 | period_quality_ridge | 24 | 8.79 pp | 8.74 pp | 17.38 pp | 19.49 pp | -0.91 pp | 6.54% |
| RTX5090 | period_only_ridge | 24 | 9.60 pp | 8.71 pp | 18.39 pp | 25.08 pp | -0.25 pp | 6.72% |
| RTX5090 | constant_median_added_ms | 24 | 10.14 pp | 8.40 pp | 22.61 pp | 30.20 pp | +4.04 pp | 7.54% |
| RTX5090 | constant_median_ratio | 24 | 18.16 pp | 13.87 pp | 38.21 pp | 46.45 pp | +7.91 pp | 11.58% |

## Production fits

| Device | Selected family | Alpha | Grouped-CV overhead MAE | Features |
|---|---|---:|---:|---:|
| RTX4090 | period_only_ridge | 0.1 | 8.95 pp | 1 |
| RTX5090 | behavior_only_ridge | 1 | 4.77 pp | 83 |

## Largest primary out-of-fold errors

| Device | Workload | Actual overhead | Predicted overhead | Absolute error | Actual dry ms | Estimated dry ms |
|---|---|---:|---:|---:|---:|---:|
| RTX4090 | C20 | 8.69% | 41.87% | 33.17 pp | 68.034 | 50.047 |
| RTX4090 | C09 | 58.77% | 35.88% | 22.89 pp | 31.867 | 35.693 |
| RTX4090 | C21 | 56.75% | 35.07% | 21.68 pp | 35.460 | 40.720 |
| RTX4090 | C16 | 47.35% | 27.07% | 20.28 pp | 45.763 | 50.365 |
| RTX4090 | C07 | 18.11% | 36.86% | 18.75 pp | 47.109 | 40.188 |
| RTX4090 | C10 | 17.26% | 35.33% | 18.07 pp | 49.200 | 41.751 |
| RTX5090 | C01 | 43.22% | 59.23% | 16.01 pp | 29.060 | 25.121 |
| RTX5090 | C06 | 48.41% | 34.01% | 14.41 pp | 31.588 | 33.954 |
| RTX4090 | C17 | 40.36% | 54.30% | 13.95 pp | 38.002 | 32.403 |
| RTX4090 | C04 | 23.12% | 36.59% | 13.47 pp | 45.086 | 40.266 |

## Interpretation and limits

- The validated scope is a held-out YOLO configuration on RTX4090 or RTX5090, not an unseen model family or unseen GPU.
- This model applies only to `nsys_gpu_metrics`; profiler modes have materially different overhead distributions.
- Each profiler/workload label currently comes from one profiling run, while dry ground truth is the median of three baselines. Small or negative observed overheads can therefore contain run-to-run noise.
- The model estimates total observed steady-window slowdown; it does not separately identify counter sampling, CUDA interception, host scheduling, or detector error.
- Workload metadata, iteration markers, callbacks, and dry runtime are excluded from deployment inputs.
