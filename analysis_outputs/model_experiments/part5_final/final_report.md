# Final model experiment report

## Decision

Use the 15-feature compact GPU Ridge model separately for RTX4090-to-RTX5090 and RTX5090-to-RTX4090. It is deployment-aligned: source profiling and workload fields are inputs; target unprofiled baseline runtime is label-only.

## Stage comparison

| Direction | Stage | Deployment eligible | Best model | MAE ms | MAPE | Max APE | R2 |
|---|---|---|---|---:|---:|---:|---:|
| RTX4090_to_RTX5090 | part1_idealized | no | log_source_ols | 1.835 | 4.10% | 11.97% | 0.9852 |
| RTX4090_to_RTX5090 | part2_idealized | no | huber_interactions_log_target | 1.117 | 3.11% | 11.29% | 0.9930 |
| RTX4090_to_RTX5090 | part3_deployment | yes | compact_gpu_ridge | 1.284 | 3.14% | 10.74% | 0.9935 |
| RTX4090_to_RTX5090 | part4_deployment | yes | static_compact_gpu_ridge | 1.284 | 3.14% | 10.74% | 0.9935 |
| RTX5090_to_RTX4090 | part1_idealized | no | global_ratio_median | 2.493 | 4.23% | 12.50% | 0.9702 |
| RTX5090_to_RTX4090 | part2_idealized | no | ridge_main_log_target | 2.046 | 4.27% | 12.92% | 0.9864 |
| RTX5090_to_RTX4090 | part3_deployment | yes | compact_gpu_ridge | 2.028 | 4.26% | 10.22% | 0.9900 |
| RTX5090_to_RTX4090 | part4_deployment | yes | static_compact_gpu_ridge | 2.028 | 4.26% | 10.22% | 0.9900 |

## Workload-bootstrap uncertainty for the selected deployment model

Intervals resample the 24 held-out workload predictions within each direction. They quantify workload-sampling uncertainty for this GPU pair, not unseen-device uncertainty.

| Direction | Metric | Estimate | 95% bootstrap interval |
|---|---|---:|---:|
| RTX4090_to_RTX5090 | mae_ms | 1.284 ms | 0.962 to 1.651 ms |
| RTX4090_to_RTX5090 | rmse_ms | 1.549 ms | 1.101 to 1.965 ms |
| RTX4090_to_RTX5090 | mape_percent | 3.138% | 2.299 to 4.145% |
| RTX5090_to_RTX4090 | mae_ms | 2.028 ms | 1.526 to 2.605 ms |
| RTX5090_to_RTX4090 | rmse_ms | 2.429 ms | 1.788 to 3.031 ms |
| RTX5090_to_RTX4090 | mape_percent | 4.257% | 3.263 to 5.332% |

## Error slices

### RTX4090_to_RTX5090

| Slice | Value | N | MAE ms | MAPE | Bias ms | Max APE |
|---|---|---:|---:|---:|---:|---:|
| batch | 4 | 6 | 1.449 | 4.79% | +0.212 | 10.74% |
| batch | 8 | 6 | 1.033 | 3.05% | +0.185 | 6.18% |
| batch | 16 | 6 | 0.763 | 1.73% | -0.079 | 2.19% |
| batch | 32 | 6 | 1.890 | 2.98% | -0.737 | 6.78% |
| imgsz | 320 | 8 | 1.520 | 4.69% | +0.030 | 10.74% |
| imgsz | 480 | 8 | 0.838 | 1.97% | -0.124 | 2.60% |
| imgsz | 640 | 8 | 1.493 | 2.76% | -0.221 | 4.84% |
| amp | False | 12 | 1.435 | 3.50% | -0.181 | 10.74% |
| amp | True | 12 | 1.132 | 2.77% | -0.029 | 6.78% |

### RTX5090_to_RTX4090

| Slice | Value | N | MAE ms | MAPE | Bias ms | Max APE |
|---|---|---:|---:|---:|---:|---:|
| batch | 4 | 6 | 2.081 | 6.10% | +0.262 | 9.06% |
| batch | 8 | 6 | 2.224 | 5.43% | -0.211 | 10.22% |
| batch | 16 | 6 | 1.067 | 2.23% | -0.253 | 3.12% |
| batch | 32 | 6 | 2.741 | 3.28% | +1.794 | 5.08% |
| imgsz | 320 | 8 | 1.096 | 3.06% | -0.057 | 5.63% |
| imgsz | 480 | 8 | 1.977 | 4.45% | +0.590 | 9.06% |
| imgsz | 640 | 8 | 3.013 | 5.26% | +0.661 | 10.22% |
| amp | False | 12 | 2.061 | 4.27% | +0.482 | 10.22% |
| amp | True | 12 | 1.996 | 4.25% | +0.314 | 9.06% |

## Interpretation and limits

- Against the best Part 2 idealized model, the deployment model changes mean APE by only +0.026 percentage points for RTX4090-to-RTX5090 (paired Wilcoxon p=0.584) and -0.014 points for RTX5090-to-RTX4090 (p=0.922); no measurable accuracy penalty is evident in this dataset.
- Compact GPU metrics capture most useful profiling signal; adding full GPU summaries, trace, or current HPCT artifact proxies does not improve nested-LOOW error.
- Current evaluation supports unseen-workload generalization only for the known RTX4090/RTX5090 directed pair.
- Hardware descriptors are constant within each direction and cannot establish unseen-device generalization with only two GPU models.
- Current source profiling covers 108 steady iterations. Shorter 5/10/20-iteration window accuracy remains unmeasured.
- RTX4090 and RTX5090 use different CUDA trace backends, so trace features require additional harmonization before treating them as device-independent.
- Current HPCToolkit features are quality/complexity proxies. Numeric CCT attribution must be extracted before HPCToolkit can be judged as a performance-feature source.

## Recommended next evidence

1. Re-extract Nsight metrics from progressively shorter profiling windows and plot accuracy versus profiling cost.
2. Add at least one third GPU and evaluate leave-one-device-out; until then, avoid claims about arbitrary unseen GPU hardware.
3. Add more model architectures and data-pipeline conditions so workload generalization is not limited to YOLO26n and the current batch/image/AMP grid.
4. Extract numeric HPCToolkit CCT metrics only if kernel/calling-context attribution can be validated against Nsight.

## Packaged models

- `RTX4090_to_RTX5090`: `models\RTX4090_to_RTX5090_compact_gpu_ridge.joblib`, alpha=0.1, 15 inputs.
- `RTX5090_to_RTX4090`: `models\RTX5090_to_RTX4090_compact_gpu_ridge.joblib`, alpha=0.1, 15 inputs.
