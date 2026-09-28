# High-load trace-only marker-free period experiment

The detector reads only CUDA kernel start timestamps and kernel short-name IDs. NVTX is joined after detection for audit only.

| Horizon | Accepted | Evaluated | Prefix mean APE | Prefix median APE | Prefix P90 APE | Prefix <=10% | Full-run median APE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5s | 12/24 | 12 | 4.63% | 1.93% | 13.21% | 75.0% | 7.20% |
| 7s | 17/24 | 15 | 1.83% | 1.79% | 3.02% | 100.0% | 18.95% |
| 9s | 22/24 | 22 | 1.73% | 1.12% | 4.04% | 95.5% | 15.64% |
| 12s | 24/24 | 24 | 2.31% | 1.10% | 3.16% | 95.8% | 10.60% |
| 15s | 24/24 | 24 | 1.48% | 1.24% | 3.03% | 100.0% | 8.63% |
| 20s | 23/24 | 23 | 1.12% | 0.63% | 3.12% | 100.0% | 5.35% |
| 30s | 24/24 | 24 | 1.48% | 0.92% | 3.05% | 100.0% | 3.96% |

## Marker-blind deployment policy

- Coverage: 24/24 (100.0%)
- Mean emission horizon: 13.54s
- Prefix mean / median / P90 / max APE: 1.47% / 1.14% / 3.72% / 4.46%
- Prefix within 10%: 100.0%
- Full-run mean / median APE: 11.04% / 8.41%

## Integrity

- Profiles checked: 24
- All profiles contain one CUDA process: True
- All profiles contain one CUDA context: True
- All profiles used CUDA hardware tracing: True

## Interpretation

Prefix APE measures whether the detector recovers the period actually visible inside the same profiling window. Full-run APE compares an early estimate with the later full steady-state median; under non-stationary background contention, this includes future load drift and is not purely detector error.
