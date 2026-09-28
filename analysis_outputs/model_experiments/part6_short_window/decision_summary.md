# Short-window profiling decision summary

Probe time is measured from TRAIN_ITER_0001 start through the last requested steady iteration. It includes 20 warmup iterations and model-run startup inside the NVTX ranges, but excludes Nsight process startup/export overhead.

Behavior-only models do not use timestamp duration, GPU sample count, window span, source ground-truth runtime, or target runtime. Duration-assisted is shown separately.

| Direction | K | Median probe s | P90 probe s | Behavior mean MAPE | Distribution MAPE | Compact-cycle MAPE | Mean + duration MAPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| RTX4090_to_RTX5090 | 5 | 4.95 | 7.20 | 4.72% | 3.58% | 4.16% | 3.20% |
| RTX4090_to_RTX5090 | 10 | 5.42 | 7.51 | 3.96% | 8.74% | 4.95% | 3.34% |
| RTX4090_to_RTX5090 | 20 | 6.36 | 8.17 | 4.58% | 12.87% | 4.89% | 3.10% |
| RTX4090_to_RTX5090 | 50 | 8.48 | 10.13 | 3.64% | 4.91% | 5.16% | 3.06% |
| RTX4090_to_RTX5090 | 108 | 11.53 | 14.53 | 4.18% | 5.11% | 4.38% | 3.60% |
| RTX5090_to_RTX4090 | 5 | 3.72 | 4.68 | 8.26% | 7.18% | 8.50% | 7.39% |
| RTX5090_to_RTX4090 | 10 | 4.11 | 4.94 | 6.76% | 7.04% | 8.13% | 6.17% |
| RTX5090_to_RTX4090 | 20 | 4.89 | 5.53 | 6.55% | 6.65% | 8.12% | 5.17% |
| RTX5090_to_RTX4090 | 50 | 6.50 | 7.52 | 6.60% | 7.15% | 7.58% | 4.49% |
| RTX5090_to_RTX4090 | 108 | 9.18 | 11.46 | 6.91% | 5.82% | 6.10% | 4.32% |

## Interpretation

- The full training job is not required. Every K row is causal and uses no sample after steady iteration 20+K.
- RTX4090-to-RTX5090 is already strong at K=5. RTX5090-to-RTX4090 needs a longer prefix for stable low error, especially when using the observed source duration.
- Expanded per-metric cycle features are unstable with only 24 training workloads. Five aggregated cycle descriptors avoid catastrophic outliers but do not beat simple metric means/distributions.
- A practical current policy is K=20 for a balanced probe, or K=50 when reverse-direction accuracy is more important. K=5 is reasonable only for a latency-first policy with a wider error budget.