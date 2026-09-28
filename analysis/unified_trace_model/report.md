# Unified clean/high-load RTX5090 trace model

The same frozen marker-free detector and trace feature schema are used for both conditions. Mixed evaluation excludes both clean and high-load rows of the held-out workload.

| Experiment | Model | Test | MAPE | Median APE | P90 APE | <=10% |
|---|---|---|---:|---:|---:|---:|
| clean_to_clean_loow | trace_only | clean | 3.89% | 2.92% | 7.16% | 95.8% |
| clean_to_high_strict | trace_only | high_load_01 | 68.70% | 76.68% | 86.48% | 0.0% |
| high_to_clean_strict | trace_only | clean | 208.71% | 234.13% | 280.08% | 0.0% |
| mixed_grouped_loow | trace_only | clean | 6.18% | 4.40% | 12.36% | 79.2% |
| mixed_grouped_loow | trace_only | high_load_01 | 14.68% | 11.62% | 32.05% | 41.7% |
| mixed_grouped_loow | trace_oracle_regime_moe | clean | 3.89% | 2.92% | 7.16% | 95.8% |
| mixed_grouped_loow | trace_oracle_regime_moe | high_load_01 | 7.42% | 6.87% | 12.31% | 79.2% |
| mixed_grouped_loow | trace_plus_load_flag | clean | 8.12% | 7.08% | 16.93% | 70.8% |
| mixed_grouped_loow | trace_plus_load_flag | high_load_01 | 10.83% | 8.12% | 25.30% | 58.3% |
| mixed_grouped_loow | trace_plus_load_interactions | clean | 5.79% | 4.91% | 10.88% | 83.3% |
| mixed_grouped_loow | trace_plus_load_interactions | high_load_01 | 6.76% | 5.33% | 13.95% | 75.0% |
| mixed_grouped_loow | trace_plus_pre_netdata | clean | 9.88% | 9.23% | 20.63% | 66.7% |
| mixed_grouped_loow | trace_plus_pre_netdata | high_load_01 | 9.07% | 7.09% | 18.80% | 66.7% |
| mixed_grouped_loow | trace_regime_moe | clean | 7.65% | 2.98% | 9.07% | 91.7% |
| mixed_grouped_loow | trace_regime_moe | high_load_01 | 7.42% | 6.87% | 12.31% | 79.2% |
| mixed_grouped_loow | trace_soft_regime_interactions | clean | 6.73% | 5.29% | 11.26% | 79.2% |
| mixed_grouped_loow | trace_soft_regime_interactions | high_load_01 | 6.76% | 5.33% | 13.95% | 75.0% |
| mixed_grouped_loow | trace_only | both | 10.43% | 7.86% | 19.07% | 60.4% |
| mixed_grouped_loow | trace_plus_load_flag | both | 9.47% | 7.76% | 19.75% | 64.6% |
| mixed_grouped_loow | trace_plus_load_interactions | both | 6.27% | 5.00% | 11.94% | 79.2% |
| mixed_grouped_loow | trace_plus_pre_netdata | both | 9.48% | 8.47% | 20.13% | 66.7% |
| mixed_grouped_loow | trace_regime_moe | both | 7.53% | 5.38% | 11.94% | 85.4% |
| mixed_grouped_loow | trace_oracle_regime_moe | both | 5.65% | 5.27% | 11.56% | 87.5% |
| mixed_grouped_loow | trace_soft_regime_interactions | both | 6.75% | 5.29% | 12.24% | 77.1% |

## Headline results

- Clean-only LOOW MAPE: 3.89%.
- Strict clean-to-high MAPE: 68.70%.
- Strict high-to-clean MAPE: 208.71%.
- Unified trace-only grouped LOOW MAPE: 10.43%.
- Unified trace + load flag MAPE: 9.47%.
- Unified trace + load interactions MAPE: 6.27%.
- Unified trace + pre-run telemetry MAPE: 9.48%.
- Trace-inferred regime mixture MAPE: 7.53% (classification accuracy 97.9%).
- Oracle-regime mixture upper-bound MAPE: 5.65%.
- Trace-only soft-regime interaction MAPE: 6.75% (classification accuracy 97.9%).

Clean runs contain no Netdata files. The telemetry ablation therefore uses fold-local imputation and missingness indicators; it is informative but confounded with condition identity.
