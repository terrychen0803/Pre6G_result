# Collection validation report

## Data inventory

- Discovered devices: RTX4090, RTX5090
- Discovered run directories: 336
- Expected workloads from config: 24

| Device | Workloads | Baseline | Nsight trace | GPU metrics | CPU metrics | HPCToolkit CUDA |
|---|---:|---:|---:|---:|---:|---:|
| RTX4090 | 24 | 72 | 24 | 24 | 24 | 24 |
| RTX5090 | 24 | 72 | 24 | 24 | 24 | 24 |

## Completeness

All required baseline and Nsight run counts are complete.

## File and configuration validation

- Malformed/unreadable runs: 0
- Runs inconsistent with experiments.csv: 0
- Nsight runs missing profile.nsys-rep: 0
- Distinct recorded dataset hashes: 0
- Warning: no run metadata records a dataset semantic hash; dataset identity cannot be cryptographically verified.

HPCToolkit is optional and is not included in the main-data validity gate.
