# Part 2: workload-aware regularized models

Outer evaluation is leave-one-workload-out. Scaling and hyperparameter selection use only the outer training fold; selection uses an inner leave-one-workload-out loop.

| Direction | Model | MAE ms | RMSE ms | MAPE | Median APE | Max APE | R2 |
|---|---|---:|---:|---:|---:|---:|---:|
| RTX4090_to_RTX5090 | huber_interactions_log_target | 1.117 | 1.613 | 3.11% | 1.48% | 11.29% | 0.9930 |
| RTX4090_to_RTX5090 | ridge_main_log_target | 1.310 | 1.813 | 3.42% | 2.50% | 13.01% | 0.9911 |
| RTX4090_to_RTX5090 | ridge_interactions_log_ratio | 1.373 | 1.826 | 3.56% | 2.36% | 12.94% | 0.9910 |
| RTX4090_to_RTX5090 | elasticnet_interactions_log_target | 1.380 | 1.829 | 3.62% | 2.40% | 12.85% | 0.9910 |
| RTX4090_to_RTX5090 | ridge_interactions_log_target | 1.434 | 1.870 | 3.75% | 2.03% | 13.05% | 0.9905 |
| RTX5090_to_RTX4090 | ridge_main_log_target | 2.046 | 2.833 | 4.27% | 2.43% | 12.92% | 0.9864 |
| RTX5090_to_RTX4090 | huber_interactions_log_target | 1.933 | 2.421 | 4.40% | 2.93% | 12.48% | 0.9900 |
| RTX5090_to_RTX4090 | elasticnet_interactions_log_target | 2.479 | 3.761 | 4.80% | 3.46% | 12.60% | 0.9760 |
| RTX5090_to_RTX4090 | ridge_interactions_log_target | 2.300 | 2.993 | 4.85% | 3.87% | 12.60% | 0.9848 |
| RTX5090_to_RTX4090 | ridge_interactions_log_ratio | 2.678 | 3.815 | 5.12% | 3.94% | 12.58% | 0.9753 |
| both_directions_combined | huber_interactions_log_target | 1.525 | 2.057 | 3.75% | 2.36% | 12.48% | 0.9913 |
| both_directions_combined | ridge_main_log_target | 1.678 | 2.379 | 3.85% | 2.48% | 13.01% | 0.9884 |
| both_directions_combined | elasticnet_interactions_log_target | 1.930 | 2.957 | 4.21% | 2.84% | 12.85% | 0.9821 |
| both_directions_combined | ridge_interactions_log_target | 1.867 | 2.496 | 4.30% | 3.08% | 13.05% | 0.9872 |
| both_directions_combined | ridge_interactions_log_ratio | 2.026 | 2.991 | 4.34% | 3.04% | 12.94% | 0.9817 |

## Change relative to the best Part 1 baseline

| Direction | Part 1 best | Part 1 MAPE | Part 2 best | Part 2 MAPE | Relative change |
|---|---|---:|---|---:|---:|
| RTX4090_to_RTX5090 | log_source_ols | 4.10% | huber_interactions_log_target | 3.11% | -24.2% |
| RTX5090_to_RTX4090 | global_ratio_median | 4.23% | ridge_main_log_target | 4.27% | +1.0% |

## Hyperparameter selection frequency

- RTX4090_to_RTX5090 / elasticnet_interactions_log_target: `{"model__alpha": 0.001, "model__l1_ratio": 0.9}`: 21; `{"model__alpha": 0.001, "model__l1_ratio": 0.1}`: 1; `{"model__alpha": 0.01, "model__l1_ratio": 0.1}`: 1; `{"model__alpha": 0.001, "model__l1_ratio": 0.5}`: 1
- RTX4090_to_RTX5090 / huber_interactions_log_target: `{"model__alpha": 1.0, "model__epsilon": 1.1}`: 13; `{"model__alpha": 0.1, "model__epsilon": 1.1}`: 8; `{"model__alpha": 1.0, "model__epsilon": 1.35}`: 1; `{"model__alpha": 0.01, "model__epsilon": 1.1}`: 1; `{"model__alpha": 0.0, "model__epsilon": 1.1}`: 1
- RTX4090_to_RTX5090 / ridge_interactions_log_ratio: `{"model__alpha": 1.0}`: 22; `{"model__alpha": 0.1}`: 2
- RTX4090_to_RTX5090 / ridge_interactions_log_target: `{"model__alpha": 0.1}`: 24
- RTX4090_to_RTX5090 / ridge_main_log_target: `{"model__alpha": 0.01}`: 12; `{"model__alpha": 0.0}`: 8; `{"model__alpha": 0.1}`: 4
- RTX5090_to_RTX4090 / elasticnet_interactions_log_target: `{"model__alpha": 0.01, "model__l1_ratio": 0.5}`: 15; `{"model__alpha": 0.001, "model__l1_ratio": 0.9}`: 6; `{"model__alpha": 0.01, "model__l1_ratio": 0.9}`: 2; `{"model__alpha": 0.001, "model__l1_ratio": 0.1}`: 1
- RTX5090_to_RTX4090 / huber_interactions_log_target: `{"model__alpha": 1.0, "model__epsilon": 1.35}`: 15; `{"model__alpha": 1.0, "model__epsilon": 1.1}`: 4; `{"model__alpha": 0.0, "model__epsilon": 1.1}`: 3; `{"model__alpha": 0.01, "model__epsilon": 1.35}`: 1; `{"model__alpha": 0.1, "model__epsilon": 1.1}`: 1
- RTX5090_to_RTX4090 / ridge_interactions_log_ratio: `{"model__alpha": 1.0}`: 12; `{"model__alpha": 100.0}`: 10; `{"model__alpha": 0.1}`: 1; `{"model__alpha": 0.0}`: 1
- RTX5090_to_RTX4090 / ridge_interactions_log_target: `{"model__alpha": 0.1}`: 16; `{"model__alpha": 0.01}`: 7; `{"model__alpha": 0.0}`: 1
- RTX5090_to_RTX4090 / ridge_main_log_target: `{"model__alpha": 0.0}`: 22; `{"model__alpha": 0.01}`: 1; `{"model__alpha": 0.1}`: 1

## Worst workloads for the best Part 2 model

### RTX4090_to_RTX5090: `huber_interactions_log_target`

| Workload | Source ms | Actual ms | Predicted ms | APE |
|---|---:|---:|---:|---:|
| C10 | 49.200 | 39.302 | 43.738 | 11.29% |
| C01 | 30.278 | 29.060 | 25.995 | 10.55% |
| C17 | 38.002 | 31.537 | 34.437 | 9.20% |
| C15 | 38.872 | 33.221 | 36.206 | 8.99% |
| C09 | 31.867 | 30.749 | 28.686 | 6.71% |

### RTX5090_to_RTX4090: `ridge_main_log_target`

| Workload | Source ms | Actual ms | Predicted ms | APE |
|---|---:|---:|---:|---:|
| C01 | 29.060 | 30.278 | 34.189 | 12.92% |
| C10 | 39.302 | 49.200 | 43.020 | 12.56% |
| C17 | 31.537 | 38.002 | 33.856 | 10.91% |
| C15 | 33.221 | 38.872 | 35.455 | 8.79% |
| C09 | 30.749 | 31.867 | 34.552 | 8.42% |
