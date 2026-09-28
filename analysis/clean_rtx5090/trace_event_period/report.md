# High-load trace-only marker-free period experiment

The detector reads only CUDA kernel start timestamps and kernel short-name IDs. NVTX is joined after detection for audit only.

| Horizon | Accepted | Evaluated | Prefix mean APE | Prefix median APE | Prefix P90 APE | Prefix <=10% | Full-run median APE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5s | 12/24 | 12 | 0.53% | 0.16% | 0.73% | 100.0% | 0.41% |
| 7s | 21/24 | 21 | 0.53% | 0.33% | 1.28% | 100.0% | 0.40% |
| 9s | 24/24 | 24 | 0.32% | 0.27% | 0.61% | 100.0% | 0.36% |
| 12s | 24/24 | 24 | 0.28% | 0.26% | 0.52% | 100.0% | 0.33% |
| 15s | 24/24 | 24 | 0.31% | 0.28% | 0.63% | 100.0% | 0.24% |
| 20s | 24/24 | 24 | 0.31% | 0.28% | 0.63% | 100.0% | 0.24% |
| 30s | 6/24 | 6 | 0.09% | 0.09% | 0.16% | 100.0% | 0.09% |

## Marker-blind deployment policy

- Coverage: 24/24 (100.0%)
- Mean emission horizon: 9.38s
- Prefix mean / median / P90 / max APE: 0.33% / 0.27% / 0.63% / 1.15%
- Prefix within 10%: 100.0%
- Full-run mean / median APE: 0.56% / 0.34%

## Integrity

- Profiles checked: 24
- All profiles contain one CUDA process: True
- All profiles contain one CUDA context: True
- All profiles used CUDA hardware tracing: True

## Interpretation

Prefix APE measures whether the detector recovers the period actually visible inside the same profiling window. Full-run APE compares an early estimate with the later full steady-state median; under non-stationary background contention, this includes future load drift and is not purely detector error.
