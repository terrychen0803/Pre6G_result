# Trace-only high-load runtime / overhead prediction

All features are computed from target-process CUDA kernel events available by the marker-blind emission horizon. NVTX and workload configuration are not model inputs.

- Evaluation: nested leave-one-workload-out
- Profiles: 24
- Baseline MAPE / median / P90 / max APE: 7.42% / 6.87% / 12.31% / 17.50%
- Baseline within 10%: 79.2%
- Baseline MAE / RMSE: 10.80 / 13.44 ms
- Overhead MAE / P90 absolute error: 8.08 / 14.87 percentage points
- Direct detected-period baseline MAPE: 17.76%

## Label stability

The five unprofiled baseline repeats have mean / median / max CV of 6.67% / 5.96% / 13.84%.
Observed trace overhead ranges from -15.55% to 27.53% (median 10.12%).
