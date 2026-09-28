# Part 3: deployment-aligned Nsight GPU profiling models

The target-device unprofiled baseline runtime is used only as the label. Model inputs contain only workload fields and source-device Nsight GPU profiling fields available before prediction. `source_ground_truth_iter_ms` is excluded from every Part 3 model.

Common complete GPU summary inputs: 66 columns (mean, CV, and P90 summaries).

Outer evaluation is leave-one-workload-out. Variance filtering, scaling, PCA, and hyperparameter selection are fitted only inside each outer training fold.

| Direction | Model | Inputs | MAE ms | RMSE ms | MAPE | Median APE | Max APE | R2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| RTX4090_to_RTX5090 | compact_gpu_ridge | 15 | 1.284 | 1.549 | 3.14% | 2.17% | 10.74% | 0.9935 |
| RTX4090_to_RTX5090 | static_gpu_summary_pca_ridge | 70 | 4.212 | 14.227 | 5.99% | 3.28% | 66.41% | 0.4530 |
| RTX4090_to_RTX5090 | profile_duration_proxy_ridge | 4 | 2.998 | 4.332 | 6.31% | 4.93% | 16.94% | 0.9493 |
| RTX4090_to_RTX5090 | static_gpu_summary_ridge | 70 | 4.626 | 13.339 | 6.95% | 4.89% | 61.42% | 0.5191 |
| RTX4090_to_RTX5090 | gpu_summary_pca_ridge | 67 | 4.986 | 16.515 | 7.07% | 3.34% | 76.75% | 0.2628 |
| RTX4090_to_RTX5090 | static_only_ridge | 3 | 6.966 | 10.439 | 14.45% | 13.51% | 38.35% | 0.7055 |
| RTX5090_to_RTX4090 | compact_gpu_ridge | 15 | 2.028 | 2.429 | 4.26% | 3.25% | 10.22% | 0.9900 |
| RTX5090_to_RTX4090 | static_gpu_summary_ridge | 70 | 2.642 | 3.668 | 4.66% | 4.15% | 9.56% | 0.9771 |
| RTX5090_to_RTX4090 | gpu_summary_pca_ridge | 67 | 2.580 | 3.546 | 4.67% | 3.70% | 11.03% | 0.9786 |
| RTX5090_to_RTX4090 | static_gpu_summary_pca_ridge | 70 | 2.666 | 3.881 | 4.73% | 3.84% | 11.39% | 0.9744 |
| RTX5090_to_RTX4090 | profile_duration_proxy_ridge | 4 | 3.527 | 4.708 | 7.30% | 6.24% | 16.47% | 0.9623 |
| RTX5090_to_RTX4090 | static_only_ridge | 3 | 10.976 | 16.141 | 19.72% | 18.53% | 44.20% | 0.5574 |
| both_directions_combined | compact_gpu_ridge | 15 | 1.656 | 2.037 | 3.70% | 2.58% | 10.74% | 0.9915 |
| both_directions_combined | static_gpu_summary_pca_ridge | 70 | 3.439 | 10.428 | 5.36% | 3.62% | 66.41% | 0.7771 |
| both_directions_combined | static_gpu_summary_ridge | 70 | 3.634 | 9.782 | 5.80% | 4.39% | 61.42% | 0.8039 |
| both_directions_combined | gpu_summary_pca_ridge | 67 | 3.783 | 11.944 | 5.87% | 3.34% | 76.75% | 0.7076 |
| both_directions_combined | profile_duration_proxy_ridge | 4 | 3.263 | 4.524 | 6.80% | 5.56% | 16.94% | 0.9581 |
| both_directions_combined | static_only_ridge | 3 | 8.971 | 13.592 | 17.08% | 16.15% | 44.20% | 0.6213 |

## Comparison with earlier idealized baselines

Part 1 and Part 2 use the source device's unprofiled ground-truth runtime, which is a stronger input than the short-profiling deployment scenario. They are reported as reference upper baselines, not deployment-equivalent competitors.

| Direction | Part 1 best | Part 1 MAPE | Part 2 best | Part 2 MAPE | Part 3 best | Part 3 MAPE |
|---|---|---:|---|---:|---|---:|
| RTX4090_to_RTX5090 | log_source_ols | 4.10% | huber_interactions_log_target | 3.11% | compact_gpu_ridge | 3.14% |
| RTX5090_to_RTX4090 | global_ratio_median | 4.23% | ridge_main_log_target | 4.27% | compact_gpu_ridge | 4.26% |

## Hyperparameter selection frequency

- RTX4090_to_RTX5090 / compact_gpu_ridge: `{"model__alpha": 0.1}`: 18; `{"model__alpha": 0.01}`: 6
- RTX4090_to_RTX5090 / gpu_summary_pca_ridge: `{"model__alpha": 1.0, "pca__n_components": 12}`: 16; `{"model__alpha": 1.0, "pca__n_components": 16}`: 3; `{"model__alpha": 0.1, "pca__n_components": 8}`: 3; `{"model__alpha": 0.1, "pca__n_components": 12}`: 1; `{"model__alpha": 10.0, "pca__n_components": 12}`: 1
- RTX4090_to_RTX5090 / profile_duration_proxy_ridge: `{"model__alpha": 1.0}`: 17; `{"model__alpha": 0.1}`: 7
- RTX4090_to_RTX5090 / static_gpu_summary_pca_ridge: `{"model__alpha": 1.0, "pca__n_components": 12}`: 11; `{"model__alpha": 0.1, "pca__n_components": 8}`: 7; `{"model__alpha": 0.1, "pca__n_components": 12}`: 4; `{"model__alpha": 1.0, "pca__n_components": 16}`: 1; `{"model__alpha": 1.0, "pca__n_components": 8}`: 1
- RTX4090_to_RTX5090 / static_gpu_summary_ridge: `{"model__alpha": 1.0}`: 10; `{"model__alpha": 0.1}`: 8; `{"model__alpha": 10.0}`: 3; `{"model__alpha": 0.01}`: 3
- RTX4090_to_RTX5090 / static_only_ridge: `{"model__alpha": 0.01}`: 15; `{"model__alpha": 1.0}`: 7; `{"model__alpha": 0.1}`: 2
- RTX5090_to_RTX4090 / compact_gpu_ridge: `{"model__alpha": 0.1}`: 21; `{"model__alpha": 0.01}`: 3
- RTX5090_to_RTX4090 / gpu_summary_pca_ridge: `{"model__alpha": 1.0, "pca__n_components": 16}`: 17; `{"model__alpha": 0.1, "pca__n_components": 16}`: 6; `{"model__alpha": 1.0, "pca__n_components": 8}`: 1
- RTX5090_to_RTX4090 / profile_duration_proxy_ridge: `{"model__alpha": 1.0}`: 24
- RTX5090_to_RTX4090 / static_gpu_summary_pca_ridge: `{"model__alpha": 1.0, "pca__n_components": 16}`: 13; `{"model__alpha": 0.1, "pca__n_components": 16}`: 6; `{"model__alpha": 0.1, "pca__n_components": 8}`: 2; `{"model__alpha": 1.0, "pca__n_components": 8}`: 2; `{"model__alpha": 0.1, "pca__n_components": 12}`: 1
- RTX5090_to_RTX4090 / static_gpu_summary_ridge: `{"model__alpha": 1.0}`: 16; `{"model__alpha": 0.1}`: 7; `{"model__alpha": 0.01}`: 1
- RTX5090_to_RTX4090 / static_only_ridge: `{"model__alpha": 1.0}`: 13; `{"model__alpha": 10.0}`: 10; `{"model__alpha": 0.1}`: 1

## Worst workloads for the best Part 3 model

### RTX4090_to_RTX5090: `compact_gpu_ridge`

| Workload | Actual ms | Predicted ms | APE |
|---|---:|---:|---:|
| C01 | 29.060 | 25.940 | 10.74% |
| C16 | 41.930 | 39.087 | 6.78% |
| C13 | 30.371 | 32.367 | 6.57% |
| C02 | 28.075 | 29.809 | 6.18% |
| C09 | 30.749 | 32.236 | 4.84% |

### RTX5090_to_RTX4090: `compact_gpu_ridge`

| Workload | Actual ms | Predicted ms | APE |
|---|---:|---:|---:|
| C10 | 49.200 | 44.170 | 10.22% |
| C17 | 38.002 | 34.560 | 9.06% |
| C09 | 31.867 | 34.640 | 8.70% |
| C06 | 34.708 | 37.431 | 7.85% |
| C21 | 35.460 | 37.809 | 6.62% |
