# Experiment C: clean + high-load training to held-out high-load workload

Each outer fold holds out one workload from both clean and high-load data, then tests only its high-load profile. Ridge alpha is selected inside the remaining workload groups using high-load validation error.

## Forced full-coverage diagnostic

- Original behavior features: overhead MAE 3.77 pp; dry-runtime MAPE 29.97%
- Original + pre-run Netdata: overhead MAE 5.32 pp; dry-runtime MAPE 31.15%
- Overhead MAE change from Netdata: +1.56 pp

Clean rows do not have Netdata timestamps. Their pre-run fields are missing and imputed inside each training fold; missing-value indicators allow the model to distinguish absent historical telemetry. This makes the ablation usable but not equivalent to having true clean-condition load measurements.
