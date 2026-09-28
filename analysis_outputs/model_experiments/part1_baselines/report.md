# Part 1: leakage-safe runtime baselines

Each direction is trained independently. Every outer fold holds out one workload, and the reverse direction is never used as training data for that fold.

| Direction | Model | MAE ms | RMSE ms | MAPE | Median APE | Max APE | R2 |
|---|---|---:|---:|---:|---:|---:|---:|
| RTX4090_to_RTX5090 | log_source_ols | 1.835 | 2.342 | 4.10% | 3.67% | 11.97% | 0.9852 |
| RTX4090_to_RTX5090 | global_ratio_median | 2.260 | 3.802 | 4.40% | 3.66% | 14.29% | 0.9609 |
| RTX4090_to_RTX5090 | linear_source_ols | 1.999 | 2.658 | 4.42% | 4.10% | 11.25% | 0.9809 |
| RTX4090_to_RTX5090 | identity_source | 5.854 | 8.068 | 11.93% | 10.43% | 25.93% | 0.8241 |
| RTX4090_to_RTX5090 | target_train_median | 15.568 | 22.278 | 30.33% | 25.95% | 67.05% | -0.3414 |
| RTX5090_to_RTX4090 | global_ratio_median | 2.493 | 4.191 | 4.23% | 3.80% | 12.50% | 0.9702 |
| RTX5090_to_RTX4090 | log_source_ols | 2.262 | 3.137 | 4.32% | 3.58% | 11.45% | 0.9833 |
| RTX5090_to_RTX4090 | linear_source_ols | 2.494 | 3.355 | 4.83% | 4.24% | 11.28% | 0.9809 |
| RTX5090_to_RTX4090 | identity_source | 5.854 | 8.068 | 10.40% | 9.45% | 20.59% | 0.8894 |
| RTX5090_to_RTX4090 | target_train_median | 18.821 | 27.601 | 32.67% | 30.13% | 70.38% | -0.2943 |
| both_directions_macro_rows | log_source_ols | 2.048 | 2.768 | 4.21% | 3.58% | 11.97% | 0.9843 |
| both_directions_macro_rows | global_ratio_median | 2.377 | 4.001 | 4.31% | 3.72% | 14.29% | 0.9672 |
| both_directions_macro_rows | linear_source_ols | 2.246 | 3.027 | 4.63% | 4.23% | 11.28% | 0.9812 |
| both_directions_macro_rows | identity_source | 5.854 | 8.068 | 11.16% | 9.88% | 25.93% | 0.8666 |
| both_directions_macro_rows | target_train_median | 17.194 | 25.082 | 31.50% | 28.22% | 70.38% | -0.2894 |

## Worst workloads for the best model in each direction

### RTX4090_to_RTX5090: `log_source_ols`

| Workload | Source ms | Actual ms | Predicted ms | APE |
|---|---:|---:|---:|---:|
| C10 | 49.200 | 39.302 | 44.008 | 11.97% |
| C17 | 38.002 | 31.537 | 34.671 | 9.94% |
| C15 | 38.872 | 33.221 | 35.338 | 6.37% |
| C12 | 131.249 | 104.224 | 110.715 | 6.23% |
| C21 | 35.460 | 34.129 | 32.254 | 5.50% |

### RTX5090_to_RTX4090: `global_ratio_median`

| Workload | Source ms | Actual ms | Predicted ms | APE |
|---|---:|---:|---:|---:|
| C12 | 104.224 | 131.249 | 114.837 | 12.50% |
| C10 | 39.302 | 49.200 | 43.304 | 11.98% |
| C17 | 31.537 | 38.002 | 34.748 | 8.56% |
| C11 | 61.916 | 73.354 | 68.220 | 7.00% |
| C09 | 30.749 | 31.867 | 34.035 | 6.80% |
