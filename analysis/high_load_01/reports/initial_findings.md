# RTX5090 high-load GPU-metrics initial findings

## Scope

This analysis includes unprofiled baselines and `nsys_gpu_metrics_01` only. It excludes every `nsys_trace_01` run and the legacy `C01/nsys_gpu_metrics_01_nsys2024` directory. Raw collection files are read-only; all exports, caches, and results are under `analysis/high_load_01/`.

## Data alignment and QC

| Check | Result |
|---|---:|
| Workload configurations aligned | 24/24 |
| Clean/high collection `experiments.csv` SHA-256 match | yes |
| High-load baselines | 120 (5 per workload) |
| High-load GPU-metrics profiles | 24 |
| Official profiles using Nsight 2026.4.1 | 24/24 |
| Runs with 128 total / 108 steady iterations | 144/144 |
| Missing required files | 0 |
| Netdata timestamp alignment failures | 0 |
| Netdata gaps over 2 seconds | 0 |
| Strong failure/OOM strings | 0 |
| Pre-run mean GPU utilization below 50% | 0 |
| Exact clean/high GPU metric descriptor matches | 24/24 |

The lowest observed five-second pre-run GPU-utilization mean is 80.6%. The high-load condition did not drop to idle in the checked runs.

Clean has three baseline repeats per workload; high-load has five. Workload IDs and configuration fields can be paired, but execution repeat numbers cannot. Clean iteration CSVs have no Unix timestamps and clean runs have no paired Netdata, so load-state matching is available only within the high-load condition.

## Baseline variability

| Condition | Median workload CV | P90 workload CV | Maximum workload CV |
|---|---:|---:|---:|
| Clean | 0.89% | 2.54% | 2.81% |
| High load | 5.96% | 10.54% | 13.84% |

High-load baseline variability is therefore part of the result. No repeat is removed solely because it is faster or slower than the other repeats.

## Original Part 7 period detector

The detector receives only the five GPU-metric time series. NVTX and iteration timestamps are joined afterward as hidden evaluation labels.

| Evaluation | Coverage | Mean APE | Median APE | P90 APE | Maximum APE | Within 10% |
|---|---:|---:|---:|---:|---:|---:|
| Clean RTX5090 OOF adaptive policy | 100% | 0.86% | 0.77% | 1.56% | 2.25% | 100% |
| Clean-trained policy → high load | 4.2% | 8.61%* | 8.61%* | 8.61%* | 8.61%* | 100%* |
| Clean-trained policy → high load, forced 15 s diagnostic | 100% | 35.46% | 29.99% | 81.85% | 90.23% | 29.2% |
| High-load LOOW retuning, forced acceptance | 100% | 24.47% | 17.34% | 52.02% | 80.65% | 25.0% |

`*` Only one of 24 high-load profiles passed the clean-trained confidence gate. Its conditional error must not be interpreted as overall accuracy.

For the forced clean-trained diagnostic, period MAE is 52.15 ms, RMSE is 69.42 ms, and signed bias is -5.95 ms. NVTX median spacing and `iterations.csv` start-to-start median agree extremely closely: median discrepancy 0.0005%, maximum 0.0021%. The error is therefore not caused by a ground-truth alignment problem.

The likely mechanism is device-wide signal contamination: under the high-load condition, the five GPU metric channels contain activity from the two background GPU processes as well as YOLO. Their dominant autocorrelation peaks no longer reliably represent the YOLO iteration period.

## Observed GPU-metrics profiling overhead

| Condition/reference | Mean overhead | Median overhead | P10 | P90 | Negative observations |
|---|---:|---:|---:|---:|---:|
| Clean / baseline median | 32.80% | 40.71% | — | — | 1/24 |
| High load / baseline median | 9.87% | 9.96% | — | — | 3/24 |
| High load / closest pre-run GPU-state baseline | 12.63% | 9.91% | -4.55% | 27.46% | 5/24 |

Nine of 24 high-load profiled runtimes fall inside the min–max range of the five unprofiled baselines. Negative observed overhead does not imply profiler acceleration; it shows that asynchronous contention variation can be larger than the profiling effect.

## Overhead model domain shift

The existing RTX5090 production model is an 83-feature behavior-only Ridge model trained on clean GPU-metrics profiles.

| Experiment | Detector coverage | Overhead MAE | Overhead RMSE | Estimated dry-runtime MAPE |
|---|---:|---:|---:|---:|
| A: clean → clean OOF | 100% | 5.95 pp | 7.43 pp | 4.86% |
| B: clean model → high load, forced diagnostic | 75% | 7962.45 pp | 14374.00 pp | 94.19% |
| C: clean + high → held-out high, original features | 75% | 3.77 pp | 4.98 pp | 29.97% |
| C ablation: original + pre-run Netdata | 75% | 5.32 pp | 6.25 pp | 31.15% |

Experiment B is catastrophic extrapolation, not a plausible overhead estimate. Across high-load profiles, the median fraction of production features outside the clean training range is 28.9%, and the maximum is 33.7%. In particular, background activity shifts SM-active, SM-issue, Tensor-active, and bandwidth statistics far beyond the clean distribution. The existing Ridge model has no OOD guard or upper bound on `exp(model_output)`.

Experiment C holds one workload out from both clean and high-load training data and tests only the held-out high-load profile. Adding high-load examples recalibrates overhead-ratio prediction successfully, but end-to-end dry-runtime error remains high because the detected period is wrong. This separates the two failures: the overhead mapping can be relearned, while period detection remains the principal bottleneck.

The first pre-run Netdata ablation does not improve results. This is not evidence that load features are intrinsically useless: there are only 24 high-load workloads, Netdata is sampled at 1 Hz, GPU utilization is near saturation, and clean runs have no corresponding Netdata values. Clean missing values are imputed inside each training fold with missing-value indicators.

## Comparability cautions

1. The collection-level workload configs, software stack, Nsight version, sampling frequency, and exported GPU metric descriptors match. Clean explicitly requested metric set `gb20x`, while high-load used automatic selection; the exported descriptors nevertheless match exactly.
2. Dataset paths differ and metadata does not record a dataset semantic hash. Configuration and synthetic dataset naming match, but dataset identity cannot be cryptographically proven from metadata alone.
3. Profiling and baseline runs were collected at different times. Baseline mean/median and pre-run load-matched comparisons should both be retained.
4. A five-second Netdata window contains only five observations. It is suitable for coarse state matching, not fine-grained contention reconstruction.
5. GPU metrics are device-wide under this experiment. Background-process activity is part of the signal, which breaks the clean detector's assumption that the dominant period belongs to the target workload.
6. The high-load sample contains 24 workload configurations from one model family and one GPU. Complex-model claims are not justified.

## Initial conclusion

The clean and high-load collections are structurally alignable at the workload level. The high-load data passes the initial integrity and timestamp checks. However, high GPU contention creates a strong domain shift in the device-wide GPU metric time series: the original period detector loses coverage/accuracy, and the clean-trained overhead model extrapolates catastrophically. Retraining with high-load examples repairs overhead-ratio prediction, but cannot repair end-to-end runtime prediction until period detection is made target-aware or otherwise robust to background-process activity.
