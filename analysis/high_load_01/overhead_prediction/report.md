# Clean-trained RTX5090 GPU-metrics overhead model on high load

- Production model family: behavior_only_ridge
- Model feature count: 83
- High-load multiscale detector coverage: 18/24 (75.0%)
- Median fraction of model features outside the clean training range: 28.9%
- Maximum per-profile fraction outside the clean range: 33.7%

## Forced full-coverage diagnostic

Rejected detections are forced through feature extraction only to diagnose domain shift; this is not valid deployment behavior.

- Overhead MAE: 7962.45 pp
- Overhead RMSE: 14374.00 pp
- Overhead P90 absolute error: 18372.44 pp
- Overhead bias: +7962.45 pp
- Estimated dry-runtime MAPE: 94.19%
- Estimated dry-runtime P90 APE: 99.19%

## Confidence-accepted subset

- Profiles: 18
- Overhead MAE: 6656.13 pp
- Estimated dry-runtime MAPE: 93.57%

The production model was trained only on clean RTX5090 profiles. High-load baseline median is used only as the post-hoc label.
