# Original clean-trained period detector on high-load RTX5090

The detector and its adaptive policy are selected only from the existing clean GPU-metrics data. High-load markers/timestamps are used only after detection as evaluation labels.

- Clean-selected confidence threshold: 0.40
- High-load accepted profiles: 1/24 (4.2%)
- Mean APE versus iteration-start median: 8.61%
- Median APE: 8.61%
- P90 APE: 8.61%
- Maximum APE: 8.61%
- Within 10%: 100.0%
- Mean emission horizon: 12.00s

## Forced terminal output (diagnostic only)

The following treats each rejected profile's last 15-second candidate as if it had been emitted. This measures period error across all workloads, but is not the confidence-gated deployment behavior.

- Profiles: 24
- Mean APE: 35.46%
- Median APE: 29.99%
- P90 APE: 81.85%
- Maximum APE: 90.23%
- MAE: 52.15 ms
- RMSE: 69.42 ms
- Bias: -5.95 ms
- Within 10%: 29.2%

## Oracle consistency

- Median NVTX-versus-iterations median APE: 0.0005%
- Maximum NVTX-versus-iterations median APE: 0.0021%

## Important interpretation

A large error here is a clean-to-high domain-shift result, not evidence that the reports are misaligned: workload configs, software versions, metric descriptors, and iteration labels are checked separately.
