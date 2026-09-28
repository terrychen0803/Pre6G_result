# Part 7 checkpoint: marker-free cycle detection

The detector reads only cached GPU metric time series. It never queries marker ranges. This evaluator reads marker ranges only after detection to create hidden period labels.

Each outer fold holds out one workload across all available devices. Detector bin width, tail fraction, and confidence threshold are selected only from the remaining workloads.

| Horizon | Device | N | Coverage | Mean period APE | Median | P90 | Max | <=5% | <=10% | Half | Double |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5s | RTX5090 | 24 | 100.0% | 22.10% | 15.99% | 50.52% | 84.29% | 20.8% | 37.5% | 1 | 0 |
| 5s | both_devices | 24 | 100.0% | 22.10% | 15.99% | 50.52% | 84.29% | 20.8% | 37.5% | 1 | 0 |
| 7s | RTX5090 | 24 | 100.0% | 23.69% | 15.02% | 56.64% | 80.65% | 16.7% | 29.2% | 1 | 0 |
| 7s | both_devices | 24 | 100.0% | 23.69% | 15.02% | 56.64% | 80.65% | 16.7% | 29.2% | 1 | 0 |
| 9s | RTX5090 | 24 | 100.0% | 24.47% | 17.34% | 52.02% | 80.65% | 8.3% | 25.0% | 1 | 0 |
| 9s | both_devices | 24 | 100.0% | 24.47% | 17.34% | 52.02% | 80.65% | 8.3% | 25.0% | 1 | 0 |
| 12s | RTX5090 | 24 | 100.0% | 30.28% | 19.70% | 79.26% | 86.50% | 8.3% | 29.2% | 0 | 0 |
| 12s | both_devices | 24 | 100.0% | 30.28% | 19.70% | 79.26% | 86.50% | 8.3% | 29.2% | 0 | 0 |
| 15s | RTX5090 | 24 | 100.0% | 29.26% | 24.56% | 71.54% | 85.24% | 12.5% | 25.0% | 0 | 0 |
| 15s | both_devices | 24 | 100.0% | 29.26% | 24.56% | 71.54% | 85.24% | 12.5% | 25.0% | 0 | 0 |

## Adaptive marker-free policy (nested threshold selection)

Each held-out workload uses a confidence threshold selected only from the other 23 workloads. The detector tries 9s, then continues to 12s and 15s only if confidence is insufficient.

| Device | Profiles | Accepted | Coverage | Mean APE | Median | P90 | Max | <=10% | Mean emission |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RTX5090 | 24 | 24 | 100.0% | 24.47% | 17.34% | 52.02% | 80.65% | 25.0% | 9.00s |
| both_devices | 24 | 24 | 100.0% | 24.47% | 17.34% | 52.02% | 80.65% | 25.0% | 9.00s |

## Checkpoint decision

The adaptive confidence-gated policy accepted 24/24 profiles; 25.0% of accepted periods were within 10% of the hidden marker period. This does not meet the original clean-data result.

This evaluation contains 24 workload configurations on 1 device(s). The source profiles were collected from instrumented runs even though marker/callback fields are never exposed to the detector. Results do not establish generalization to arbitrary workloads or load conditions.

## Worst out-of-fold detections

| Horizon | Device | Workload | Detected ms | Oracle ms | APE | Class | Configuration |
|---:|---|---|---:|---:|---:|---|---|
| 12s | RTX5090 | C23 | 24.00 | 177.79 | 86.50% | other_error | `{"bin_ms": 2.0, "tail_fraction": 0.8}` |
| 15s | RTX5090 | C20 | 28.00 | 189.77 | 85.24% | other_error | `{"bin_ms": 4.0, "tail_fraction": 0.5}` |
| 5s | RTX5090 | C02 | 20.00 | 127.32 | 84.29% | other_error | `{"bin_ms": 4.0, "tail_fraction": 0.65}` |
| 12s | RTX5090 | C04 | 26.00 | 145.23 | 82.10% | other_error | `{"bin_ms": 2.0, "tail_fraction": 0.8}` |
| 7s | RTX5090 | C16 | 28.00 | 144.70 | 80.65% | other_error | `{"bin_ms": 4.0, "tail_fraction": 0.8}` |
| 9s | RTX5090 | C16 | 28.00 | 144.70 | 80.65% | other_error | `{"bin_ms": 4.0, "tail_fraction": 0.8}` |
| 15s | RTX5090 | C16 | 28.00 | 144.70 | 80.65% | other_error | `{"bin_ms": 4.0, "tail_fraction": 0.8}` |
| 7s | RTX5090 | C07 | 28.00 | 144.45 | 80.62% | other_error | `{"bin_ms": 4.0, "tail_fraction": 0.8}` |
| 12s | RTX5090 | C16 | 30.00 | 144.70 | 79.27% | other_error | `{"bin_ms": 2.0, "tail_fraction": 0.8}` |
| 12s | RTX5090 | C10 | 28.00 | 134.91 | 79.24% | other_error | `{"bin_ms": 4.0, "tail_fraction": 0.8}` |
| 15s | RTX5090 | C10 | 28.00 | 134.91 | 79.24% | other_error | `{"bin_ms": 4.0, "tail_fraction": 0.8}` |
| 5s | RTX5090 | C16 | 32.00 | 144.70 | 77.89% | other_error | `{"bin_ms": 4.0, "tail_fraction": 0.5}` |