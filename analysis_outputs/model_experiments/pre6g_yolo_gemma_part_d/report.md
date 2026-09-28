# Part D: YOLO + Gemma marker-free runtime prediction

All experiments use RTX5090 marker-free profiling features to predict RTX4090 unprofiled per-iteration runtime. DNN and ViT are excluded.

## Results

| Protocol | Evaluation family | Model | N | MAPE | Median APE | P90 | Max | R2 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| combined_yolo_gemma_loo | all | nested_selected_ridge | 40 | 7.54% | 6.88% | 14.97% | 21.99% | 0.9926 |
| combined_yolo_gemma_loo | all | period_compact_ridge | 40 | 7.54% | 6.88% | 14.97% | 21.99% | 0.9926 |
| combined_yolo_gemma_loo | all | period_quality_ridge | 40 | 9.85% | 6.27% | 22.42% | 28.74% | 0.9610 |
| combined_yolo_gemma_loo | all | period_full_ridge | 40 | 11.00% | 9.38% | 19.91% | 49.15% | 0.9514 |
| combined_yolo_gemma_loo | all | behavior_only_ridge | 40 | 12.23% | 9.44% | 22.37% | 63.52% | 0.9389 |
| combined_yolo_gemma_loo | gemma4 | nested_selected_ridge | 16 | 5.69% | 4.93% | 10.62% | 15.84% | 0.9834 |
| combined_yolo_gemma_loo | gemma4 | period_compact_ridge | 16 | 5.69% | 4.93% | 10.62% | 15.84% | 0.9834 |
| combined_yolo_gemma_loo | gemma4 | period_full_ridge | 16 | 10.85% | 10.64% | 19.57% | 28.87% | 0.8677 |
| combined_yolo_gemma_loo | gemma4 | behavior_only_ridge | 16 | 12.14% | 13.10% | 22.22% | 33.12% | 0.8347 |
| combined_yolo_gemma_loo | gemma4 | period_quality_ridge | 16 | 13.96% | 17.28% | 26.43% | 28.74% | 0.8904 |
| combined_yolo_gemma_loo | yolo26 | period_quality_ridge | 24 | 7.10% | 5.14% | 16.77% | 20.55% | 0.9198 |
| combined_yolo_gemma_loo | yolo26 | nested_selected_ridge | 24 | 8.78% | 7.87% | 18.77% | 21.99% | 0.9025 |
| combined_yolo_gemma_loo | yolo26 | period_compact_ridge | 24 | 8.78% | 7.87% | 18.77% | 21.99% | 0.9025 |
| combined_yolo_gemma_loo | yolo26 | period_full_ridge | 24 | 11.10% | 7.22% | 19.83% | 49.15% | 0.8186 |
| combined_yolo_gemma_loo | yolo26 | behavior_only_ridge | 24 | 12.29% | 7.78% | 20.89% | 63.52% | 0.7495 |
| train_gemma_test_yolo | yolo26 | period_median_ratio | 24 | 32.72% | 38.77% | 50.67% | 51.58% | 0.6743 |
| train_gemma_test_yolo | yolo26 | period_full_ridge | 24 | 82.77% | 84.65% | 138.08% | 146.53% | -5.6177 |
| train_gemma_test_yolo | yolo26 | period_quality_ridge | 24 | 90.00% | 95.40% | 98.81% | 98.96% | -2.6092 |
| train_gemma_test_yolo | yolo26 | period_compact_ridge | 24 | 97.56% | 81.47% | 178.88% | 198.52% | -9.0357 |
| train_gemma_test_yolo | yolo26 | behavior_only_ridge | 24 | 139.72% | 121.75% | 241.31% | 244.04% | -16.0118 |
| train_yolo_test_gemma | gemma4 | period_median_ratio | 16 | 23.98% | 27.93% | 40.28% | 42.35% | 0.3468 |
| train_yolo_test_gemma | gemma4 | period_compact_ridge | 16 | 37.61% | 38.48% | 47.14% | 50.50% | -0.0039 |
| train_yolo_test_gemma | gemma4 | period_quality_ridge | 16 | 48.02% | 46.76% | 68.64% | 71.02% | -0.1314 |
| train_yolo_test_gemma | gemma4 | behavior_only_ridge | 16 | 65.38% | 61.83% | 92.24% | 92.64% | -2.7066 |
| train_yolo_test_gemma | gemma4 | period_full_ridge | 16 | 71.63% | 65.10% | 98.30% | 98.63% | -3.3595 |
| yolo_only_loo | yolo26 | period_compact_ridge | 24 | 5.52% | 2.68% | 11.10% | 22.21% | 0.9524 |
| yolo_only_loo | yolo26 | period_quality_ridge | 24 | 6.47% | 6.05% | 13.66% | 21.27% | 0.9567 |
| yolo_only_loo | yolo26 | nested_selected_ridge | 24 | 7.23% | 7.86% | 11.94% | 17.59% | 0.9580 |
| yolo_only_loo | yolo26 | period_full_ridge | 24 | 7.33% | 5.15% | 14.31% | 19.89% | 0.9352 |
| yolo_only_loo | yolo26 | behavior_only_ridge | 24 | 9.55% | 9.41% | 16.97% | 22.49% | 0.8956 |

## Production fit

- Selected family: period_compact_ridge
- Ridge alpha: 0.1
- Grouped CV MAPE used for selection: 6.66%
- Training rows: 40
- Primary evaluation: combined_yolo_gemma_loo / all / nested_selected
- The grouped-CV selection score is not a replacement for the outer LOO estimate.