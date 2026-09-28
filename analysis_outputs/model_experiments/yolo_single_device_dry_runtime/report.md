# YOLO single-device direct dry-runtime prediction

This experiment directly predicts the unprofiled dry-run iteration runtime from marker-free Nsight GPU Metrics profiling features on the same known device. Derived overhead is calculated from the detected profiling period and predicted dry runtime.

Outer evaluation leaves one YOLO workload out within each device. Ridge alpha and the primary nested feature family are selected using only inner grouped folds.

## Outer leave-one-workload-out results

| Device | Model | N | Dry MAPE | Median APE | P90 APE | Max APE | R2 | Derived overhead MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| RTX4090 | period_only_ridge | 24 | 6.59% | 5.49% | 12.80% | 18.71% | 0.9744 | 9.58 pp |
| RTX4090 | period_quality_ridge | 24 | 6.71% | 5.13% | 13.09% | 15.44% | 0.9782 | 9.12 pp |
| RTX4090 | constant_median_added_ms_correction | 24 | 7.51% | 7.82% | 14.24% | 16.99% | 0.9483 | 9.26 pp |
| RTX4090 | period_compact_ridge | 24 | 8.04% | 6.95% | 15.44% | 26.56% | 0.9350 | 11.05 pp |
| RTX4090 | nested_selected_ridge | 24 | 8.06% | 6.74% | 13.60% | 26.56% | 0.9662 | 11.49 pp |
| RTX4090 | period_full_ridge | 24 | 8.65% | 6.38% | 21.69% | 30.42% | 0.8976 | 12.62 pp |
| RTX4090 | behavior_only_ridge | 24 | 9.32% | 7.71% | 23.05% | 33.02% | 0.8660 | 13.52 pp |
| RTX4090 | constant_median_ratio_correction | 24 | 11.03% | 10.36% | 21.21% | 22.75% | 0.8368 | 16.52 pp |
| RTX4090 | uncorrected_detected_period | 24 | 29.32% | 32.38% | 47.40% | 55.10% | 0.7387 | 33.37 pp |
| RTX5090 | nested_selected_ridge | 24 | 3.21% | 2.70% | 5.33% | 10.51% | 0.9839 | 5.90 pp |
| RTX5090 | period_compact_ridge | 24 | 3.21% | 2.70% | 5.33% | 10.51% | 0.9839 | 5.90 pp |
| RTX5090 | period_full_ridge | 24 | 4.15% | 3.83% | 8.00% | 9.13% | 0.9771 | 7.23 pp |
| RTX5090 | behavior_only_ridge | 24 | 5.23% | 4.38% | 10.54% | 16.89% | 0.9436 | 8.55 pp |
| RTX5090 | period_quality_ridge | 24 | 6.30% | 6.04% | 14.37% | 15.36% | 0.9625 | 9.48 pp |
| RTX5090 | period_only_ridge | 24 | 7.17% | 6.91% | 12.46% | 20.12% | 0.9478 | 11.18 pp |
| RTX5090 | constant_median_added_ms_correction | 24 | 7.54% | 6.55% | 17.79% | 24.95% | 0.9058 | 10.14 pp |
| RTX5090 | constant_median_ratio_correction | 24 | 10.98% | 8.33% | 23.21% | 29.94% | 0.7746 | 17.07 pp |
| RTX5090 | uncorrected_detected_period | 24 | 28.75% | 34.79% | 44.76% | 47.63% | 0.6726 | 33.07 pp |

## Production fits

| Device | Selected family | Alpha | Grouped-CV dry MAPE | Features |
|---|---|---:|---:|---:|
| RTX4090 | period_only_ridge | 0.01 | 6.90% | 1 |
| RTX5090 | period_compact_ridge | 0.1 | 2.39% | 47 |

## Nested target comparison

| Device | Training target | Dry-runtime MAPE | Overhead MAE |
|---|---|---:|---:|
| RTX4090 | direct_log_dry_runtime | 8.06% | 11.49 pp |
| RTX4090 | direct_log_overhead_ratio | 7.81% | 10.71 pp |
| RTX5090 | direct_log_dry_runtime | 3.21% | 5.90 pp |
| RTX5090 | direct_log_overhead_ratio | 4.98% | 6.66 pp |

## Largest primary out-of-fold errors

| Device | Workload | Actual dry ms | Predicted dry ms | APE | Actual overhead | Predicted overhead |
|---|---|---:|---:|---:|---:|---:|
| RTX4090 | C18 | 36.716 | 46.469 | 26.56% | 44.59% | 3.29% |
| RTX4090 | C21 | 35.460 | 40.935 | 15.44% | 56.75% | 34.36% |
| RTX4090 | C20 | 68.034 | 58.750 | 13.65% | 8.69% | 20.85% |
| RTX4090 | C16 | 45.763 | 51.942 | 13.50% | 47.35% | 23.21% |
| RTX4090 | C10 | 49.200 | 43.100 | 12.40% | 17.26% | 31.09% |
| RTX4090 | C17 | 38.002 | 33.391 | 12.13% | 40.36% | 49.74% |
| RTX4090 | C07 | 47.109 | 41.543 | 11.82% | 18.11% | 32.39% |
| RTX4090 | C09 | 31.867 | 35.395 | 11.07% | 58.77% | 37.02% |
| RTX5090 | C24 | 87.394 | 78.212 | 10.51% | 8.17% | 19.55% |
| RTX4090 | C22 | 45.275 | 40.907 | 9.65% | 29.80% | 36.90% |

## Interpretation and limits

- The primary target is dry runtime, not overhead ratio. This tests whether the cleaner main runtime signal is easier to learn than a noisy ratio residual.
- Derived overhead uses the marker-free detected period. Its error therefore includes both dry-runtime prediction error and the gap between detected period and profiler summary runtime.
- The validated scope is a held-out YOLO configuration on RTX4090 or RTX5090, not an unseen workload family, profiler mode, or GPU.
- Each profiling label currently comes from one run; repeated profiler collections are still needed to separate model error from run-to-run noise.
- Workload metadata, markers, callbacks, profiler summary runtime, and dry runtime are excluded from deployment inputs.
