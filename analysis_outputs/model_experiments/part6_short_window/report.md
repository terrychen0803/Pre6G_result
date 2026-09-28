# Part 6: causal short-window profiling experiment

Each K-iteration feature row uses only Nsight samples within TRAIN_ITER_0021 through TRAIN_ITER_(0020+K). No later iteration contributes to that row.

Behavior-only models exclude GPU sample count, NVTX duration, timestamps, window span, source ground-truth runtime, and every target runtime field. Duration-assisted models are reported separately.

Outer evaluation is leave-one-workload-out. Ridge alpha is selected inside the 23-workload outer training set using shuffled 5-fold validation; scaling and variance filtering are inner-fold local.

| Direction | K | Model | Inputs | MAE ms | MAPE | Median APE | Max APE | R2 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| RTX4090_to_RTX5090 | 5 | behavior_mean_duration_ridge | 15 | 1.266 | 3.20% | 2.15% | 11.36% | 0.9933 |
| RTX4090_to_RTX5090 | 5 | behavior_distribution_ridge | 36 | 1.630 | 3.58% | 2.87% | 11.23% | 0.9794 |
| RTX4090_to_RTX5090 | 5 | behavior_compact_cycle_ridge | 19 | 1.724 | 4.16% | 3.58% | 13.03% | 0.9871 |
| RTX4090_to_RTX5090 | 5 | behavior_mean_ridge | 14 | 2.482 | 4.72% | 2.60% | 24.82% | 0.9167 |
| RTX4090_to_RTX5090 | 5 | duration_only_ridge | 4 | 3.518 | 7.36% | 6.76% | 16.98% | 0.9280 |
| RTX4090_to_RTX5090 | 5 | behavior_cycle_ridge | 48 | 15.866 | 19.70% | 6.60% | 294.41% | -9.6496 |
| RTX4090_to_RTX5090 | 10 | behavior_mean_duration_ridge | 15 | 1.611 | 3.34% | 2.04% | 12.36% | 0.9766 |
| RTX4090_to_RTX5090 | 10 | behavior_mean_ridge | 14 | 1.952 | 3.96% | 2.64% | 14.84% | 0.9657 |
| RTX4090_to_RTX5090 | 10 | behavior_compact_cycle_ridge | 19 | 2.539 | 4.95% | 4.37% | 17.68% | 0.9499 |
| RTX4090_to_RTX5090 | 10 | duration_only_ridge | 4 | 3.391 | 7.28% | 6.34% | 18.26% | 0.9412 |
| RTX4090_to_RTX5090 | 10 | behavior_distribution_ridge | 36 | 7.396 | 8.74% | 2.33% | 143.70% | -1.5334 |
| RTX4090_to_RTX5090 | 10 | behavior_cycle_ridge | 48 | 53.054 | 53.89% | 4.89% | 1162.80% | -164.4580 |
| RTX4090_to_RTX5090 | 20 | behavior_mean_duration_ridge | 15 | 1.240 | 3.10% | 2.49% | 10.02% | 0.9941 |
| RTX4090_to_RTX5090 | 20 | behavior_mean_ridge | 14 | 2.688 | 4.58% | 2.60% | 31.90% | 0.8685 |
| RTX4090_to_RTX5090 | 20 | behavior_compact_cycle_ridge | 19 | 2.294 | 4.89% | 4.47% | 10.78% | 0.9736 |
| RTX4090_to_RTX5090 | 20 | duration_only_ridge | 4 | 3.271 | 6.98% | 5.79% | 18.89% | 0.9423 |
| RTX4090_to_RTX5090 | 20 | behavior_cycle_ridge | 48 | 6.040 | 8.06% | 3.19% | 100.00% | -0.2397 |
| RTX4090_to_RTX5090 | 20 | behavior_distribution_ridge | 36 | 11.598 | 12.87% | 2.60% | 237.87% | -5.9321 |
| RTX4090_to_RTX5090 | 50 | behavior_mean_duration_ridge | 15 | 1.226 | 3.06% | 2.25% | 10.16% | 0.9945 |
| RTX4090_to_RTX5090 | 50 | behavior_mean_ridge | 14 | 1.815 | 3.64% | 2.41% | 15.46% | 0.9648 |
| RTX4090_to_RTX5090 | 50 | behavior_distribution_ridge | 36 | 3.024 | 4.91% | 2.42% | 38.97% | 0.8044 |
| RTX4090_to_RTX5090 | 50 | behavior_compact_cycle_ridge | 19 | 2.609 | 5.16% | 3.42% | 24.35% | 0.9155 |
| RTX4090_to_RTX5090 | 50 | duration_only_ridge | 4 | 2.978 | 6.15% | 5.68% | 16.36% | 0.9462 |
| RTX4090_to_RTX5090 | 50 | behavior_cycle_ridge | 48 | 7.420 | 12.03% | 5.99% | 78.76% | 0.2225 |
| RTX4090_to_RTX5090 | 108 | behavior_mean_duration_ridge | 15 | 1.799 | 3.60% | 2.17% | 15.72% | 0.9646 |
| RTX4090_to_RTX5090 | 108 | behavior_mean_ridge | 14 | 2.447 | 4.18% | 2.26% | 31.20% | 0.8755 |
| RTX4090_to_RTX5090 | 108 | behavior_compact_cycle_ridge | 19 | 2.278 | 4.38% | 3.20% | 18.74% | 0.9474 |
| RTX4090_to_RTX5090 | 108 | behavior_distribution_ridge | 36 | 3.495 | 5.11% | 2.18% | 52.46% | 0.6555 |
| RTX4090_to_RTX5090 | 108 | duration_only_ridge | 4 | 3.006 | 6.27% | 4.92% | 16.94% | 0.9460 |
| RTX4090_to_RTX5090 | 108 | behavior_cycle_ridge | 48 | 3.700 | 6.72% | 5.95% | 38.71% | 0.7989 |
| RTX5090_to_RTX4090 | 5 | behavior_distribution_ridge | 36 | 4.406 | 7.18% | 5.47% | 20.21% | 0.9087 |
| RTX5090_to_RTX4090 | 5 | behavior_mean_duration_ridge | 15 | 3.474 | 7.39% | 5.49% | 48.72% | 0.9500 |
| RTX5090_to_RTX4090 | 5 | behavior_mean_ridge | 14 | 4.956 | 8.26% | 5.66% | 26.16% | 0.8847 |
| RTX5090_to_RTX4090 | 5 | behavior_compact_cycle_ridge | 19 | 5.031 | 8.50% | 6.89% | 24.15% | 0.8911 |
| RTX5090_to_RTX4090 | 5 | behavior_cycle_ridge | 48 | 4.992 | 8.92% | 9.61% | 19.98% | 0.9174 |
| RTX5090_to_RTX4090 | 5 | duration_only_ridge | 4 | 7.438 | 11.24% | 10.71% | 61.70% | 0.4926 |
| RTX5090_to_RTX4090 | 10 | behavior_mean_duration_ridge | 15 | 2.820 | 6.17% | 4.81% | 33.32% | 0.9752 |
| RTX5090_to_RTX4090 | 10 | behavior_mean_ridge | 14 | 3.872 | 6.76% | 5.70% | 20.51% | 0.9423 |
| RTX5090_to_RTX4090 | 10 | behavior_distribution_ridge | 36 | 4.467 | 7.04% | 5.65% | 20.53% | 0.9071 |
| RTX5090_to_RTX4090 | 10 | behavior_compact_cycle_ridge | 19 | 4.836 | 8.13% | 6.43% | 23.81% | 0.8984 |
| RTX5090_to_RTX4090 | 10 | behavior_cycle_ridge | 48 | 4.891 | 8.15% | 8.00% | 23.28% | 0.9038 |
| RTX5090_to_RTX4090 | 10 | duration_only_ridge | 4 | 5.463 | 9.17% | 9.50% | 35.57% | 0.8114 |
| RTX5090_to_RTX4090 | 20 | behavior_mean_duration_ridge | 15 | 2.657 | 5.17% | 5.16% | 10.81% | 0.9776 |
| RTX5090_to_RTX4090 | 20 | behavior_mean_ridge | 14 | 3.878 | 6.55% | 4.67% | 16.50% | 0.9344 |
| RTX5090_to_RTX4090 | 20 | behavior_distribution_ridge | 36 | 4.038 | 6.65% | 5.72% | 16.62% | 0.9331 |
| RTX5090_to_RTX4090 | 20 | duration_only_ridge | 4 | 3.760 | 7.66% | 7.75% | 16.21% | 0.9578 |
| RTX5090_to_RTX4090 | 20 | behavior_compact_cycle_ridge | 19 | 4.726 | 8.12% | 7.07% | 23.26% | 0.9069 |
| RTX5090_to_RTX4090 | 20 | behavior_cycle_ridge | 48 | 4.865 | 8.44% | 7.89% | 21.46% | 0.9132 |
| RTX5090_to_RTX4090 | 50 | behavior_mean_duration_ridge | 15 | 2.092 | 4.49% | 3.58% | 10.61% | 0.9887 |
| RTX5090_to_RTX4090 | 50 | behavior_cycle_ridge | 48 | 3.852 | 6.55% | 5.12% | 19.45% | 0.9405 |
| RTX5090_to_RTX4090 | 50 | behavior_mean_ridge | 14 | 3.792 | 6.60% | 6.52% | 16.35% | 0.9434 |
| RTX5090_to_RTX4090 | 50 | behavior_distribution_ridge | 36 | 4.544 | 7.15% | 5.18% | 18.66% | 0.9030 |
| RTX5090_to_RTX4090 | 50 | behavior_compact_cycle_ridge | 19 | 4.316 | 7.58% | 6.76% | 18.76% | 0.9237 |
| RTX5090_to_RTX4090 | 50 | duration_only_ridge | 4 | 4.366 | 7.96% | 8.10% | 22.64% | 0.9067 |
| RTX5090_to_RTX4090 | 108 | behavior_mean_duration_ridge | 15 | 1.965 | 4.32% | 3.54% | 10.23% | 0.9907 |
| RTX5090_to_RTX4090 | 108 | behavior_distribution_ridge | 36 | 3.577 | 5.82% | 5.02% | 14.86% | 0.9440 |
| RTX5090_to_RTX4090 | 108 | behavior_compact_cycle_ridge | 19 | 3.681 | 6.10% | 5.83% | 20.66% | 0.9391 |
| RTX5090_to_RTX4090 | 108 | behavior_cycle_ridge | 48 | 3.342 | 6.30% | 5.52% | 18.59% | 0.9573 |
| RTX5090_to_RTX4090 | 108 | behavior_mean_ridge | 14 | 4.147 | 6.91% | 6.14% | 19.00% | 0.9174 |
| RTX5090_to_RTX4090 | 108 | duration_only_ridge | 4 | 3.524 | 7.28% | 6.03% | 16.47% | 0.9630 |
| both_directions_combined | 5 | behavior_mean_duration_ridge | 15 | 2.370 | 5.30% | 3.03% | 48.72% | 0.9673 |
| both_directions_combined | 5 | behavior_distribution_ridge | 36 | 3.018 | 5.38% | 3.62% | 20.21% | 0.9371 |
| both_directions_combined | 5 | behavior_compact_cycle_ridge | 19 | 3.378 | 6.33% | 4.64% | 24.15% | 0.9294 |
| both_directions_combined | 5 | behavior_mean_ridge | 14 | 3.719 | 6.49% | 3.68% | 26.16% | 0.8988 |
| both_directions_combined | 5 | duration_only_ridge | 4 | 5.478 | 9.30% | 9.25% | 61.70% | 0.6666 |
| both_directions_combined | 5 | behavior_cycle_ridge | 48 | 10.429 | 14.31% | 8.02% | 294.41% | -3.0880 |
| both_directions_combined | 10 | behavior_mean_duration_ridge | 15 | 2.216 | 4.76% | 3.64% | 33.32% | 0.9762 |
| both_directions_combined | 10 | behavior_mean_ridge | 14 | 2.912 | 5.36% | 3.48% | 20.51% | 0.9522 |
| both_directions_combined | 10 | behavior_compact_cycle_ridge | 19 | 3.688 | 6.54% | 4.67% | 23.81% | 0.9197 |
| both_directions_combined | 10 | behavior_distribution_ridge | 36 | 5.931 | 7.89% | 3.62% | 143.70% | -0.0167 |
| both_directions_combined | 10 | duration_only_ridge | 4 | 4.427 | 8.23% | 7.49% | 35.57% | 0.8639 |
| both_directions_combined | 10 | behavior_cycle_ridge | 48 | 28.973 | 31.02% | 6.36% | 1162.80% | -61.7984 |
| both_directions_combined | 20 | behavior_mean_duration_ridge | 15 | 1.948 | 4.14% | 2.84% | 10.81% | 0.9842 |
| both_directions_combined | 20 | behavior_mean_ridge | 14 | 3.283 | 5.57% | 3.22% | 31.90% | 0.9105 |
| both_directions_combined | 20 | behavior_compact_cycle_ridge | 19 | 3.510 | 6.50% | 5.35% | 23.26% | 0.9338 |
| both_directions_combined | 20 | duration_only_ridge | 4 | 3.516 | 7.32% | 6.41% | 18.89% | 0.9527 |
| both_directions_combined | 20 | behavior_cycle_ridge | 48 | 5.452 | 8.25% | 6.34% | 100.00% | 0.4776 |
| both_directions_combined | 20 | behavior_distribution_ridge | 36 | 7.818 | 9.76% | 3.90% | 237.87% | -1.6690 |
| both_directions_combined | 50 | behavior_mean_duration_ridge | 15 | 1.659 | 3.77% | 2.73% | 10.61% | 0.9911 |
| both_directions_combined | 50 | behavior_mean_ridge | 14 | 2.804 | 5.12% | 3.24% | 16.35% | 0.9525 |
| both_directions_combined | 50 | behavior_distribution_ridge | 36 | 3.784 | 6.03% | 4.36% | 38.97% | 0.8673 |
| both_directions_combined | 50 | behavior_compact_cycle_ridge | 19 | 3.463 | 6.37% | 4.77% | 24.35% | 0.9219 |
| both_directions_combined | 50 | duration_only_ridge | 4 | 3.672 | 7.06% | 6.43% | 22.64% | 0.9233 |
| both_directions_combined | 50 | behavior_cycle_ridge | 48 | 5.636 | 9.29% | 5.75% | 78.76% | 0.6693 |
| both_directions_combined | 108 | behavior_mean_duration_ridge | 15 | 1.882 | 3.96% | 2.61% | 15.72% | 0.9809 |
| both_directions_combined | 108 | behavior_compact_cycle_ridge | 19 | 2.980 | 5.24% | 4.07% | 20.66% | 0.9433 |
| both_directions_combined | 108 | behavior_distribution_ridge | 36 | 3.536 | 5.46% | 3.69% | 52.46% | 0.8356 |
| both_directions_combined | 108 | behavior_mean_ridge | 14 | 3.297 | 5.55% | 2.74% | 31.20% | 0.9030 |
| both_directions_combined | 108 | behavior_cycle_ridge | 48 | 3.521 | 6.51% | 5.95% | 38.71% | 0.8980 |
| both_directions_combined | 108 | duration_only_ridge | 4 | 3.265 | 6.78% | 5.34% | 16.94% | 0.9572 |

## Best strict behavior-only model by window

| Direction | K | Model | MAPE |
|---|---:|---|---:|
| RTX4090_to_RTX5090 | 5 | behavior_distribution_ridge | 3.58% |
| RTX4090_to_RTX5090 | 10 | behavior_mean_ridge | 3.96% |
| RTX4090_to_RTX5090 | 20 | behavior_mean_ridge | 4.58% |
| RTX4090_to_RTX5090 | 50 | behavior_mean_ridge | 3.64% |
| RTX4090_to_RTX5090 | 108 | behavior_mean_ridge | 4.18% |
| RTX5090_to_RTX4090 | 5 | behavior_distribution_ridge | 7.18% |
| RTX5090_to_RTX4090 | 10 | behavior_mean_ridge | 6.76% |
| RTX5090_to_RTX4090 | 20 | behavior_mean_ridge | 6.55% |
| RTX5090_to_RTX4090 | 50 | behavior_cycle_ridge | 6.55% |
| RTX5090_to_RTX4090 | 108 | behavior_distribution_ridge | 5.82% |

## Hyperparameter selection frequency

- RTX4090_to_RTX5090 / K=5 / behavior_compact_cycle_ridge: `{"model__alpha": 1.0}`: 16; `{"model__alpha": 0.1}`: 5; `{"model__alpha": 0.01}`: 2; `{"model__alpha": 10.0}`: 1
- RTX4090_to_RTX5090 / K=5 / behavior_cycle_ridge: `{"model__alpha": 100.0}`: 13; `{"model__alpha": 0.01}`: 7; `{"model__alpha": 0.1}`: 3; `{"model__alpha": 10.0}`: 1
- RTX4090_to_RTX5090 / K=5 / behavior_distribution_ridge: `{"model__alpha": 1.0}`: 9; `{"model__alpha": 0.01}`: 8; `{"model__alpha": 0.1}`: 7
- RTX4090_to_RTX5090 / K=5 / behavior_mean_duration_ridge: `{"model__alpha": 0.1}`: 20; `{"model__alpha": 0.01}`: 3; `{"model__alpha": 1.0}`: 1
- RTX4090_to_RTX5090 / K=5 / behavior_mean_ridge: `{"model__alpha": 0.1}`: 17; `{"model__alpha": 0.01}`: 6; `{"model__alpha": 1.0}`: 1
- RTX4090_to_RTX5090 / K=5 / duration_only_ridge: `{"model__alpha": 1.0}`: 21; `{"model__alpha": 0.1}`: 2; `{"model__alpha": 0.01}`: 1
- RTX4090_to_RTX5090 / K=10 / behavior_compact_cycle_ridge: `{"model__alpha": 1.0}`: 14; `{"model__alpha": 0.1}`: 8; `{"model__alpha": 0.01}`: 1; `{"model__alpha": 10.0}`: 1
- RTX4090_to_RTX5090 / K=10 / behavior_cycle_ridge: `{"model__alpha": 100.0}`: 10; `{"model__alpha": 10.0}`: 10; `{"model__alpha": 0.01}`: 2; `{"model__alpha": 0.1}`: 2
- RTX4090_to_RTX5090 / K=10 / behavior_distribution_ridge: `{"model__alpha": 1.0}`: 12; `{"model__alpha": 0.01}`: 9; `{"model__alpha": 0.1}`: 3
- RTX4090_to_RTX5090 / K=10 / behavior_mean_duration_ridge: `{"model__alpha": 0.1}`: 17; `{"model__alpha": 0.01}`: 5; `{"model__alpha": 1.0}`: 2
- RTX4090_to_RTX5090 / K=10 / behavior_mean_ridge: `{"model__alpha": 0.1}`: 16; `{"model__alpha": 0.01}`: 7; `{"model__alpha": 1.0}`: 1
- RTX4090_to_RTX5090 / K=10 / duration_only_ridge: `{"model__alpha": 1.0}`: 21; `{"model__alpha": 0.1}`: 2; `{"model__alpha": 0.01}`: 1
- RTX4090_to_RTX5090 / K=20 / behavior_compact_cycle_ridge: `{"model__alpha": 0.1}`: 13; `{"model__alpha": 1.0}`: 10; `{"model__alpha": 0.01}`: 1
- RTX4090_to_RTX5090 / K=20 / behavior_cycle_ridge: `{"model__alpha": 0.01}`: 17; `{"model__alpha": 0.1}`: 3; `{"model__alpha": 10.0}`: 2; `{"model__alpha": 1.0}`: 2
- RTX4090_to_RTX5090 / K=20 / behavior_distribution_ridge: `{"model__alpha": 1.0}`: 15; `{"model__alpha": 0.1}`: 6; `{"model__alpha": 0.01}`: 3
- RTX4090_to_RTX5090 / K=20 / behavior_mean_duration_ridge: `{"model__alpha": 0.1}`: 15; `{"model__alpha": 0.01}`: 8; `{"model__alpha": 1.0}`: 1
- RTX4090_to_RTX5090 / K=20 / behavior_mean_ridge: `{"model__alpha": 0.1}`: 17; `{"model__alpha": 0.01}`: 6; `{"model__alpha": 1.0}`: 1
- RTX4090_to_RTX5090 / K=20 / duration_only_ridge: `{"model__alpha": 1.0}`: 18; `{"model__alpha": 0.1}`: 3; `{"model__alpha": 0.01}`: 3
- RTX4090_to_RTX5090 / K=50 / behavior_compact_cycle_ridge: `{"model__alpha": 0.1}`: 15; `{"model__alpha": 1.0}`: 8; `{"model__alpha": 0.01}`: 1
- RTX4090_to_RTX5090 / K=50 / behavior_cycle_ridge: `{"model__alpha": 10.0}`: 10; `{"model__alpha": 1.0}`: 8; `{"model__alpha": 0.01}`: 5; `{"model__alpha": 0.1}`: 1
- RTX4090_to_RTX5090 / K=50 / behavior_distribution_ridge: `{"model__alpha": 1.0}`: 15; `{"model__alpha": 0.1}`: 8; `{"model__alpha": 0.01}`: 1
- RTX4090_to_RTX5090 / K=50 / behavior_mean_duration_ridge: `{"model__alpha": 0.1}`: 20; `{"model__alpha": 0.01}`: 4
- RTX4090_to_RTX5090 / K=50 / behavior_mean_ridge: `{"model__alpha": 0.1}`: 20; `{"model__alpha": 0.01}`: 4
- RTX4090_to_RTX5090 / K=50 / duration_only_ridge: `{"model__alpha": 1.0}`: 15; `{"model__alpha": 0.1}`: 5; `{"model__alpha": 0.01}`: 4
- RTX4090_to_RTX5090 / K=108 / behavior_compact_cycle_ridge: `{"model__alpha": 0.1}`: 14; `{"model__alpha": 1.0}`: 7; `{"model__alpha": 0.01}`: 3
- RTX4090_to_RTX5090 / K=108 / behavior_cycle_ridge: `{"model__alpha": 1.0}`: 12; `{"model__alpha": 0.1}`: 7; `{"model__alpha": 0.01}`: 4; `{"model__alpha": 10.0}`: 1
- RTX4090_to_RTX5090 / K=108 / behavior_distribution_ridge: `{"model__alpha": 1.0}`: 10; `{"model__alpha": 0.01}`: 7; `{"model__alpha": 0.1}`: 6; `{"model__alpha": 10.0}`: 1
- RTX4090_to_RTX5090 / K=108 / behavior_mean_duration_ridge: `{"model__alpha": 0.1}`: 16; `{"model__alpha": 0.01}`: 5; `{"model__alpha": 1.0}`: 3
- RTX4090_to_RTX5090 / K=108 / behavior_mean_ridge: `{"model__alpha": 0.1}`: 19; `{"model__alpha": 0.01}`: 4; `{"model__alpha": 1.0}`: 1
- RTX4090_to_RTX5090 / K=108 / duration_only_ridge: `{"model__alpha": 1.0}`: 15; `{"model__alpha": 0.1}`: 6; `{"model__alpha": 0.01}`: 3
- RTX5090_to_RTX4090 / K=5 / behavior_compact_cycle_ridge: `{"model__alpha": 1.0}`: 12; `{"model__alpha": 10.0}`: 12
- RTX5090_to_RTX4090 / K=5 / behavior_cycle_ridge: `{"model__alpha": 1.0}`: 17; `{"model__alpha": 10.0}`: 5; `{"model__alpha": 0.1}`: 2
- RTX5090_to_RTX4090 / K=5 / behavior_distribution_ridge: `{"model__alpha": 0.1}`: 8; `{"model__alpha": 0.01}`: 7; `{"model__alpha": 1.0}`: 7; `{"model__alpha": 10.0}`: 2
- RTX5090_to_RTX4090 / K=5 / behavior_mean_duration_ridge: `{"model__alpha": 1.0}`: 11; `{"model__alpha": 0.1}`: 9; `{"model__alpha": 0.01}`: 3; `{"model__alpha": 10.0}`: 1
- RTX5090_to_RTX4090 / K=5 / behavior_mean_ridge: `{"model__alpha": 1.0}`: 15; `{"model__alpha": 0.1}`: 6; `{"model__alpha": 0.01}`: 3
- RTX5090_to_RTX4090 / K=5 / duration_only_ridge: `{"model__alpha": 1.0}`: 17; `{"model__alpha": 0.1}`: 6; `{"model__alpha": 0.01}`: 1
- RTX5090_to_RTX4090 / K=10 / behavior_compact_cycle_ridge: `{"model__alpha": 10.0}`: 14; `{"model__alpha": 1.0}`: 6; `{"model__alpha": 0.1}`: 3; `{"model__alpha": 0.01}`: 1
- RTX5090_to_RTX4090 / K=10 / behavior_cycle_ridge: `{"model__alpha": 10.0}`: 13; `{"model__alpha": 1.0}`: 11
- RTX5090_to_RTX4090 / K=10 / behavior_distribution_ridge: `{"model__alpha": 1.0}`: 17; `{"model__alpha": 0.1}`: 5; `{"model__alpha": 10.0}`: 2
- RTX5090_to_RTX4090 / K=10 / behavior_mean_duration_ridge: `{"model__alpha": 1.0}`: 10; `{"model__alpha": 0.1}`: 10; `{"model__alpha": 0.01}`: 3; `{"model__alpha": 10.0}`: 1
- RTX5090_to_RTX4090 / K=10 / behavior_mean_ridge: `{"model__alpha": 1.0}`: 12; `{"model__alpha": 0.01}`: 7; `{"model__alpha": 0.1}`: 5
- RTX5090_to_RTX4090 / K=10 / duration_only_ridge: `{"model__alpha": 1.0}`: 22; `{"model__alpha": 0.1}`: 2
- RTX5090_to_RTX4090 / K=20 / behavior_compact_cycle_ridge: `{"model__alpha": 10.0}`: 13; `{"model__alpha": 1.0}`: 7; `{"model__alpha": 0.1}`: 4
- RTX5090_to_RTX4090 / K=20 / behavior_cycle_ridge: `{"model__alpha": 10.0}`: 16; `{"model__alpha": 1.0}`: 8
- RTX5090_to_RTX4090 / K=20 / behavior_distribution_ridge: `{"model__alpha": 1.0}`: 19; `{"model__alpha": 10.0}`: 4; `{"model__alpha": 0.1}`: 1
- RTX5090_to_RTX4090 / K=20 / behavior_mean_duration_ridge: `{"model__alpha": 0.1}`: 12; `{"model__alpha": 1.0}`: 9; `{"model__alpha": 0.01}`: 3
- RTX5090_to_RTX4090 / K=20 / behavior_mean_ridge: `{"model__alpha": 0.1}`: 10; `{"model__alpha": 1.0}`: 8; `{"model__alpha": 0.01}`: 6
- RTX5090_to_RTX4090 / K=20 / duration_only_ridge: `{"model__alpha": 1.0}`: 23; `{"model__alpha": 0.1}`: 1
- RTX5090_to_RTX4090 / K=50 / behavior_compact_cycle_ridge: `{"model__alpha": 1.0}`: 20; `{"model__alpha": 10.0}`: 3; `{"model__alpha": 0.1}`: 1
- RTX5090_to_RTX4090 / K=50 / behavior_cycle_ridge: `{"model__alpha": 1.0}`: 18; `{"model__alpha": 10.0}`: 5; `{"model__alpha": 0.1}`: 1
- RTX5090_to_RTX4090 / K=50 / behavior_distribution_ridge: `{"model__alpha": 1.0}`: 17; `{"model__alpha": 10.0}`: 5; `{"model__alpha": 0.1}`: 2
- RTX5090_to_RTX4090 / K=50 / behavior_mean_duration_ridge: `{"model__alpha": 0.1}`: 14; `{"model__alpha": 1.0}`: 7; `{"model__alpha": 0.01}`: 3
- RTX5090_to_RTX4090 / K=50 / behavior_mean_ridge: `{"model__alpha": 0.01}`: 11; `{"model__alpha": 1.0}`: 11; `{"model__alpha": 0.1}`: 2
- RTX5090_to_RTX4090 / K=50 / duration_only_ridge: `{"model__alpha": 1.0}`: 19; `{"model__alpha": 0.1}`: 5
- RTX5090_to_RTX4090 / K=108 / behavior_compact_cycle_ridge: `{"model__alpha": 10.0}`: 12; `{"model__alpha": 1.0}`: 12
- RTX5090_to_RTX4090 / K=108 / behavior_cycle_ridge: `{"model__alpha": 10.0}`: 17; `{"model__alpha": 1.0}`: 7
- RTX5090_to_RTX4090 / K=108 / behavior_distribution_ridge: `{"model__alpha": 1.0}`: 20; `{"model__alpha": 10.0}`: 3; `{"model__alpha": 0.1}`: 1
- RTX5090_to_RTX4090 / K=108 / behavior_mean_duration_ridge: `{"model__alpha": 0.1}`: 18; `{"model__alpha": 1.0}`: 5; `{"model__alpha": 0.01}`: 1
- RTX5090_to_RTX4090 / K=108 / behavior_mean_ridge: `{"model__alpha": 0.01}`: 12; `{"model__alpha": 1.0}`: 8; `{"model__alpha": 0.1}`: 4
- RTX5090_to_RTX4090 / K=108 / duration_only_ridge: `{"model__alpha": 1.0}`: 18; `{"model__alpha": 0.1}`: 6