# Part 8 checkpoint: marker-free cross-device runtime prediction

Outer evaluation holds out one workload on both devices. Ridge alpha is selected by an inner workload-grouped five-fold split. The deployment-valid feature sets contain no workload metadata, source baseline label, target profiling field, or iteration marker.

`direction_specific` matches the earlier two-device experiments. `paired_combined` fits one model across both directions and adds source-to-target hardware ratios; with only two devices it is not an unseen-device test.

## Combined-direction results

| Training scope | Model | Valid at deployment | N | MAE ms | MAPE | Median APE | Max APE | R2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| direction_specific | period_behavior_quality_ridge | yes | 48 | 2.252 | 4.35% | 2.68% | 23.19% | 0.9644 |
| direction_specific | nested_selected_deployment_ridge | yes | 48 | 2.369 | 4.50% | 2.68% | 23.19% | 0.9614 |
| direction_specific | period_compact_behavior_ridge | yes | 48 | 2.515 | 5.11% | 2.86% | 28.25% | 0.9598 |
| direction_specific | period_full_behavior_ridge | yes | 48 | 2.973 | 5.40% | 3.37% | 19.75% | 0.9502 |
| direction_specific | behavior_only_ridge | yes | 48 | 3.268 | 6.09% | 4.01% | 22.49% | 0.9328 |
| direction_specific | period_only_ridge | yes | 48 | 3.950 | 7.53% | 5.33% | 22.10% | 0.9269 |
| direction_specific | oracle_period_only_ridge | no (oracle upper bound) | 48 | 3.995 | 7.56% | 5.63% | 23.57% | 0.9224 |
| direction_specific | period_direction_median_ratio | yes | 48 | 6.462 | 10.61% | 8.61% | 33.42% | 0.7679 |
| direction_specific | oracle_direction_median_ratio | no (oracle upper bound) | 48 | 6.611 | 11.00% | 8.33% | 34.18% | 0.7631 |
| paired_combined | period_compact_behavior_ridge | yes | 48 | 2.367 | 4.88% | 3.88% | 22.06% | 0.9710 |
| paired_combined | nested_selected_deployment_ridge | yes | 48 | 2.440 | 4.97% | 3.88% | 22.06% | 0.9679 |
| paired_combined | period_behavior_quality_ridge | yes | 48 | 2.473 | 5.03% | 3.89% | 22.51% | 0.9668 |
| paired_combined | period_full_behavior_ridge | yes | 48 | 3.309 | 5.75% | 4.20% | 41.00% | 0.9190 |
| paired_combined | behavior_only_ridge | yes | 48 | 3.866 | 6.62% | 4.97% | 42.86% | 0.8939 |
| paired_combined | period_only_ridge | yes | 48 | 3.947 | 7.51% | 6.10% | 31.77% | 0.9033 |
| paired_combined | oracle_period_only_ridge | no (oracle upper bound) | 48 | 3.998 | 7.65% | 5.93% | 29.06% | 0.9101 |

## Best deployment-valid result by direction and scope

| Training scope | Direction | Model | MAPE | Median APE | Max APE |
|---|---|---|---:|---:|---:|
| direction_specific | RTX4090_to_RTX5090 | period_full_behavior_ridge | 2.49% | 1.76% | 14.01% |
| direction_specific | RTX5090_to_RTX4090 | nested_selected_deployment_ridge | 5.75% | 3.36% | 23.19% |
| paired_combined | RTX4090_to_RTX5090 | period_full_behavior_ridge | 4.45% | 2.78% | 41.00% |
| paired_combined | RTX5090_to_RTX4090 | period_compact_behavior_ridge | 5.04% | 4.56% | 14.94% |

## Worst predictions for the primary nested result

Primary result: `direction_specific/nested_selected_deployment_ridge`. The feature family for each held-out workload was selected using inner folds only.

| Direction | Workload | Actual ms | Predicted ms | APE |
|---|---|---:|---:|---:|
| RTX5090_to_RTX4090 | C24 | 101.094 | 77.646 | 23.19% |
| RTX4090_to_RTX5090 | C20 | 62.050 | 53.357 | 14.01% |
| RTX5090_to_RTX4090 | C21 | 35.460 | 39.797 | 12.23% |
| RTX5090_to_RTX4090 | C09 | 31.867 | 35.690 | 11.99% |
| RTX5090_to_RTX4090 | C17 | 38.002 | 33.998 | 10.54% |
| RTX5090_to_RTX4090 | C13 | 33.878 | 37.231 | 9.90% |
| RTX5090_to_RTX4090 | C11 | 73.354 | 80.525 | 9.78% |
| RTX5090_to_RTX4090 | C04 | 45.086 | 49.086 | 8.87% |
| RTX5090_to_RTX4090 | C22 | 45.275 | 49.229 | 8.73% |
| RTX4090_to_RTX5090 | C10 | 39.302 | 42.471 | 8.06% |

## Comparison with earlier stages

| Stage | Model | Deployment validity for an unmodified workload | MAPE | Median APE | Max APE | R2 |
|---|---|---|---:|---:|---:|---:|
| part1_source_baseline | log_source_ols | no: requires completed source baseline timing | 4.21% | 3.58% | 11.97% | 0.9843 |
| part3_marker_aligned_full_profile | compact_gpu_ridge | no: marker-aligned and includes workload metadata | 3.70% | 2.58% | 10.74% | 0.9915 |
| part6_marker_aligned_short_window | behavior_mean_duration_ridge | no: iteration window and duration are marker-aligned | 3.77% | 2.73% | 10.61% | 0.9911 |
| part8_primary_nested | nested_selected_deployment_ridge | yes | 4.50% | 2.68% | 23.19% | 0.9614 |
| part8_best_fixed_family_posthoc | period_behavior_quality_ridge | yes, but family identified after outer evaluation | 4.35% | 2.68% | 23.19% | 0.9644 |
| part8_single_hardware_model_nested | nested_selected_deployment_ridge | yes for known devices; unseen-device validity untested | 4.97% | 3.88% | 22.06% | 0.9679 |

## Checkpoint interpretation

The primary nested, direction-specific result reaches 4.50% MAPE without workload metadata or markers. The best fixed family reaches 4.35% MAPE, but that family was identified after viewing all outer-fold results and is therefore secondary evidence.

Behavior-only and period-only models are both weaker than their combination, so the inferred source period is useful but does not replace the metric signature. The hidden-oracle period-only model is not better than the marker-free period-only model; segmentation error is not the main runtime-model bottleneck in this dataset.

The 5090-to-4090 direction remains harder, with the largest primary error on C24. A single hardware-conditioned model across both directions reaches 4.97% MAPE, but two devices cannot establish generalization to an unseen GPU. The next data requirement is a third device or additional independent workload families; model complexity should not be increased before that validation.