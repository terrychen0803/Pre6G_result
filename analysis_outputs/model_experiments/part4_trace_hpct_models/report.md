# Part 4: Nsight trace and HPCToolkit proxy ablations

All model inputs are observable on the source device during profiling. The target-device unprofiled baseline runtime is label-only, and `source_ground_truth_iter_ms` is excluded.

Nsight trace inputs contain kernel, idle-gap, memory-operation, synchronization, and CUDA API counts/times. Current HPCToolkit inputs are artifact-structure proxies (database sizes, structure/correlation counts, and accelerator trace records), not extracted numeric CCT performance metrics.

Outer evaluation is leave-one-workload-out. Scaling, variance filtering, and Ridge alpha selection occur only inside each outer training fold.

| Direction | Model | Inputs | MAE ms | RMSE ms | MAPE | Median APE | Max APE | R2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| RTX4090_to_RTX5090 | static_compact_gpu_ridge | 15 | 1.284 | 1.549 | 3.14% | 2.17% | 10.74% | 0.9935 |
| RTX4090_to_RTX5090 | static_trace_ridge | 19 | 1.353 | 1.711 | 3.28% | 2.84% | 9.19% | 0.9921 |
| RTX4090_to_RTX5090 | static_compact_gpu_hpct_proxy_ridge | 21 | 1.436 | 2.026 | 3.33% | 2.78% | 10.68% | 0.9889 |
| RTX4090_to_RTX5090 | static_trace_hpct_proxy_ridge | 25 | 1.422 | 1.800 | 3.48% | 3.10% | 9.64% | 0.9912 |
| RTX4090_to_RTX5090 | trace_only_ridge | 16 | 1.608 | 1.986 | 3.89% | 3.70% | 11.00% | 0.9893 |
| RTX4090_to_RTX5090 | static_compact_gpu_trace_ridge | 31 | 2.115 | 3.650 | 4.32% | 2.87% | 15.08% | 0.9640 |
| RTX4090_to_RTX5090 | full_profile_fusion_ridge | 37 | 2.183 | 3.509 | 4.66% | 3.38% | 14.11% | 0.9667 |
| RTX4090_to_RTX5090 | static_only_ridge | 3 | 6.966 | 10.439 | 14.45% | 13.51% | 38.35% | 0.7055 |
| RTX4090_to_RTX5090 | static_hpct_proxy_ridge | 9 | 7.169 | 10.152 | 15.66% | 14.31% | 36.22% | 0.7215 |
| RTX5090_to_RTX4090 | static_compact_gpu_ridge | 15 | 2.028 | 2.429 | 4.26% | 3.25% | 10.22% | 0.9900 |
| RTX5090_to_RTX4090 | static_compact_gpu_hpct_proxy_ridge | 21 | 2.196 | 2.776 | 4.60% | 3.61% | 11.53% | 0.9869 |
| RTX5090_to_RTX4090 | static_compact_gpu_trace_ridge | 31 | 2.068 | 2.913 | 4.61% | 3.22% | 16.09% | 0.9856 |
| RTX5090_to_RTX4090 | static_trace_ridge | 19 | 2.490 | 4.128 | 4.83% | 3.73% | 21.41% | 0.9711 |
| RTX5090_to_RTX4090 | full_profile_fusion_ridge | 37 | 2.231 | 3.036 | 4.87% | 3.49% | 16.44% | 0.9843 |
| RTX5090_to_RTX4090 | trace_only_ridge | 16 | 2.719 | 4.426 | 4.99% | 4.22% | 20.72% | 0.9667 |
| RTX5090_to_RTX4090 | static_trace_hpct_proxy_ridge | 25 | 2.314 | 3.293 | 5.01% | 3.54% | 21.77% | 0.9816 |
| RTX5090_to_RTX4090 | static_only_ridge | 3 | 10.976 | 16.141 | 19.72% | 18.53% | 44.20% | 0.5574 |
| RTX5090_to_RTX4090 | static_hpct_proxy_ridge | 9 | 11.441 | 17.850 | 20.29% | 19.58% | 52.29% | 0.4587 |
| both_directions_combined | static_compact_gpu_ridge | 15 | 1.656 | 2.037 | 3.70% | 2.58% | 10.74% | 0.9915 |
| both_directions_combined | static_compact_gpu_hpct_proxy_ridge | 21 | 1.816 | 2.430 | 3.96% | 3.30% | 11.53% | 0.9879 |
| both_directions_combined | static_trace_ridge | 19 | 1.921 | 3.159 | 4.05% | 3.16% | 21.41% | 0.9795 |
| both_directions_combined | static_trace_hpct_proxy_ridge | 25 | 1.868 | 2.654 | 4.24% | 3.20% | 21.77% | 0.9856 |
| both_directions_combined | trace_only_ridge | 16 | 2.164 | 3.430 | 4.44% | 3.77% | 20.72% | 0.9759 |
| both_directions_combined | static_compact_gpu_trace_ridge | 31 | 2.091 | 3.302 | 4.46% | 3.11% | 16.09% | 0.9777 |
| both_directions_combined | full_profile_fusion_ridge | 37 | 2.207 | 3.281 | 4.76% | 3.38% | 16.44% | 0.9779 |
| both_directions_combined | static_only_ridge | 3 | 8.971 | 13.592 | 17.08% | 16.15% | 44.20% | 0.6213 |
| both_directions_combined | static_hpct_proxy_ridge | 9 | 9.305 | 14.521 | 17.97% | 15.70% | 52.29% | 0.5678 |

## Comparison with Part 3

| Direction | Part 3 best | Part 3 MAPE | Part 4 best | Part 4 MAPE | Relative change |
|---|---|---:|---|---:|---:|
| RTX4090_to_RTX5090 | compact_gpu_ridge | 3.14% | static_compact_gpu_ridge | 3.14% | +0.0% |
| RTX5090_to_RTX4090 | compact_gpu_ridge | 4.26% | static_compact_gpu_ridge | 4.26% | +0.0% |

## Hyperparameter selection frequency

- RTX4090_to_RTX5090 / full_profile_fusion_ridge: `{"model__alpha": 10.0}`: 10; `{"model__alpha": 0.1}`: 7; `{"model__alpha": 1.0}`: 6; `{"model__alpha": 0.01}`: 1
- RTX4090_to_RTX5090 / static_compact_gpu_hpct_proxy_ridge: `{"model__alpha": 0.1}`: 17; `{"model__alpha": 1.0}`: 7
- RTX4090_to_RTX5090 / static_compact_gpu_ridge: `{"model__alpha": 0.1}`: 18; `{"model__alpha": 0.01}`: 6
- RTX4090_to_RTX5090 / static_compact_gpu_trace_ridge: `{"model__alpha": 0.1}`: 15; `{"model__alpha": 1.0}`: 5; `{"model__alpha": 10.0}`: 3; `{"model__alpha": 0.01}`: 1
- RTX4090_to_RTX5090 / static_hpct_proxy_ridge: `{"model__alpha": 0.01}`: 19; `{"model__alpha": 1.0}`: 3; `{"model__alpha": 0.1}`: 2
- RTX4090_to_RTX5090 / static_only_ridge: `{"model__alpha": 0.01}`: 15; `{"model__alpha": 1.0}`: 7; `{"model__alpha": 0.1}`: 2
- RTX4090_to_RTX5090 / static_trace_hpct_proxy_ridge: `{"model__alpha": 1.0}`: 24
- RTX4090_to_RTX5090 / static_trace_ridge: `{"model__alpha": 1.0}`: 24
- RTX4090_to_RTX5090 / trace_only_ridge: `{"model__alpha": 1.0}`: 21; `{"model__alpha": 0.1}`: 3
- RTX5090_to_RTX4090 / full_profile_fusion_ridge: `{"model__alpha": 10.0}`: 18; `{"model__alpha": 1.0}`: 6
- RTX5090_to_RTX4090 / static_compact_gpu_hpct_proxy_ridge: `{"model__alpha": 0.1}`: 21; `{"model__alpha": 1.0}`: 3
- RTX5090_to_RTX4090 / static_compact_gpu_ridge: `{"model__alpha": 0.1}`: 21; `{"model__alpha": 0.01}`: 3
- RTX5090_to_RTX4090 / static_compact_gpu_trace_ridge: `{"model__alpha": 10.0}`: 20; `{"model__alpha": 1.0}`: 4
- RTX5090_to_RTX4090 / static_hpct_proxy_ridge: `{"model__alpha": 10.0}`: 17; `{"model__alpha": 1.0}`: 6; `{"model__alpha": 0.1}`: 1
- RTX5090_to_RTX4090 / static_only_ridge: `{"model__alpha": 1.0}`: 13; `{"model__alpha": 10.0}`: 10; `{"model__alpha": 0.1}`: 1
- RTX5090_to_RTX4090 / static_trace_hpct_proxy_ridge: `{"model__alpha": 1.0}`: 23; `{"model__alpha": 0.01}`: 1
- RTX5090_to_RTX4090 / static_trace_ridge: `{"model__alpha": 1.0}`: 23; `{"model__alpha": 0.01}`: 1
- RTX5090_to_RTX4090 / trace_only_ridge: `{"model__alpha": 1.0}`: 23; `{"model__alpha": 0.01}`: 1

## Worst workloads for the best Part 4 model

### RTX4090_to_RTX5090: `static_compact_gpu_ridge`

| Workload | Actual ms | Predicted ms | APE |
|---|---:|---:|---:|
| C01 | 29.060 | 25.940 | 10.74% |
| C16 | 41.930 | 39.087 | 6.78% |
| C13 | 30.371 | 32.367 | 6.57% |
| C02 | 28.075 | 29.809 | 6.18% |
| C09 | 30.749 | 32.236 | 4.84% |

### RTX5090_to_RTX4090: `static_compact_gpu_ridge`

| Workload | Actual ms | Predicted ms | APE |
|---|---:|---:|---:|
| C10 | 49.200 | 44.170 | 10.22% |
| C17 | 38.002 | 34.560 | 9.06% |
| C09 | 31.867 | 34.640 | 8.70% |
| C06 | 34.708 | 37.431 | 7.85% |
| C21 | 35.460 | 37.809 | 6.62% |
