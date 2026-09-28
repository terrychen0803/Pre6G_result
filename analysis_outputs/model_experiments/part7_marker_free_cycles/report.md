# Part 7 checkpoint: marker-free cycle detection

The detector reads only cached GPU metric time series. It never queries marker ranges. This evaluator reads marker ranges only after detection to create hidden period labels.

Each outer fold holds out the same workload on both GPUs. Detector bin width and tail fraction are selected from the other 23 workloads (46 profiles).

| Horizon | Device | N | Coverage | Mean period APE | Median | P90 | Max | <=5% | <=10% | Half | Double |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5s | RTX4090 | 24 | 100.0% | 93.92% | 31.88% | 269.79% | 286.56% | 50.0% | 50.0% | 0 | 1 |
| 5s | RTX5090 | 24 | 100.0% | 94.97% | 5.55% | 252.18% | 274.35% | 50.0% | 54.2% | 0 | 1 |
| 5s | both_devices | 48 | 100.0% | 94.44% | 5.75% | 262.54% | 286.56% | 50.0% | 52.1% | 0 | 2 |
| 7s | RTX4090 | 24 | 100.0% | 119.23% | 52.90% | 288.25% | 325.21% | 50.0% | 50.0% | 0 | 1 |
| 7s | RTX5090 | 24 | 100.0% | 6.29% | 2.74% | 11.11% | 51.67% | 54.2% | 87.5% | 1 | 0 |
| 7s | both_devices | 48 | 100.0% | 62.76% | 2.74% | 247.65% | 325.21% | 52.1% | 68.8% | 1 | 1 |
| 9s | RTX4090 | 24 | 100.0% | 25.40% | 3.07% | 70.79% | 212.49% | 54.2% | 66.7% | 1 | 1 |
| 9s | RTX5090 | 24 | 100.0% | 1.02% | 0.79% | 1.94% | 3.63% | 100.0% | 100.0% | 0 | 0 |
| 9s | both_devices | 48 | 100.0% | 13.21% | 1.13% | 32.79% | 212.49% | 77.1% | 83.3% | 1 | 1 |
| 12s | RTX4090 | 15 | 100.0% | 1.44% | 1.14% | 3.00% | 4.32% | 100.0% | 100.0% | 0 | 0 |
| 12s | RTX5090 | 14 | 100.0% | 0.60% | 0.45% | 1.40% | 1.58% | 100.0% | 100.0% | 0 | 0 |
| 12s | both_devices | 29 | 100.0% | 1.04% | 0.83% | 2.43% | 4.32% | 100.0% | 100.0% | 0 | 0 |
| 15s | RTX4090 | 13 | 100.0% | 1.21% | 0.83% | 2.69% | 3.15% | 100.0% | 100.0% | 0 | 0 |
| 15s | RTX5090 | 7 | 100.0% | 0.71% | 0.60% | 1.28% | 1.49% | 100.0% | 100.0% | 0 | 0 |
| 15s | both_devices | 20 | 100.0% | 1.03% | 0.69% | 2.39% | 3.15% | 100.0% | 100.0% | 0 | 0 |

## Adaptive marker-free policy (nested threshold selection)

Each held-out workload uses a confidence threshold selected only from the other 23 workloads. The detector tries 9s, then continues to 12s and 15s only if confidence is insufficient.

| Device | Profiles | Accepted | Coverage | Mean APE | Median | P90 | Max | <=10% | Mean emission |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RTX4090 | 24 | 24 | 100.0% | 1.17% | 0.99% | 2.65% | 5.61% | 100.0% | 10.62s |
| RTX5090 | 24 | 24 | 100.0% | 0.86% | 0.77% | 1.56% | 2.25% | 100.0% | 9.50s |
| both_devices | 48 | 48 | 100.0% | 1.01% | 0.81% | 1.86% | 5.61% | 100.0% | 10.06s |

## Checkpoint decision

A forced period at 5s or 7s is unsafe, and a forced 9s period still has large outliers. The adaptive confidence-gated policy is suitable as a preliminary marker-free segmentation front end on this dataset: all 48 held-out profiles were accepted by 15s and all were within 10% of the hidden marker period.

This is not yet evidence of generalization to arbitrary user workloads. The dataset contains 24 workload configurations on only two GPUs, and the source traces were collected from instrumented runs even though marker/callback fields are never exposed to the detector. The next experiment should build per-cycle features from these inferred periods and evaluate cross-device runtime prediction with the same workload-held-out split.

## Worst out-of-fold detections

| Horizon | Device | Workload | Detected ms | Oracle ms | APE | Class | Configuration |
|---:|---|---|---:|---:|---:|---|---|
| 7s | RTX4090 | C18 | 220.00 | 51.74 | 325.21% | other_error | `{"bin_ms": 1.0, "tail_fraction": 0.5}` |
| 7s | RTX4090 | C17 | 220.00 | 52.26 | 321.00% | other_error | `{"bin_ms": 1.0, "tail_fraction": 0.5}` |
| 7s | RTX4090 | C21 | 220.00 | 54.78 | 301.63% | other_error | `{"bin_ms": 1.0, "tail_fraction": 0.5}` |
| 5s | RTX4090 | C18 | 200.00 | 51.74 | 286.56% | other_error | `{"bin_ms": 1.0, "tail_fraction": 0.5}` |
| 5s | RTX4090 | C17 | 200.00 | 52.26 | 282.73% | other_error | `{"bin_ms": 1.0, "tail_fraction": 0.5}` |
| 5s | RTX4090 | C13 | 191.00 | 50.42 | 278.84% | other_error | `{"bin_ms": 1.0, "tail_fraction": 0.5}` |
| 5s | RTX5090 | C21 | 180.00 | 48.08 | 274.35% | other_error | `{"bin_ms": 1.0, "tail_fraction": 0.5}` |
| 5s | RTX5090 | C18 | 180.00 | 48.77 | 269.07% | other_error | `{"bin_ms": 1.0, "tail_fraction": 0.5}` |
| 5s | RTX5090 | C17 | 167.00 | 46.42 | 259.74% | other_error | `{"bin_ms": 1.0, "tail_fraction": 0.5}` |
| 7s | RTX4090 | C13 | 180.00 | 50.42 | 257.02% | other_error | `{"bin_ms": 1.0, "tail_fraction": 0.65}` |
| 5s | RTX4090 | C14 | 180.00 | 51.62 | 248.67% | other_error | `{"bin_ms": 1.0, "tail_fraction": 0.5}` |
| 7s | RTX4090 | C14 | 180.00 | 51.62 | 248.67% | other_error | `{"bin_ms": 1.0, "tail_fraction": 0.65}` |