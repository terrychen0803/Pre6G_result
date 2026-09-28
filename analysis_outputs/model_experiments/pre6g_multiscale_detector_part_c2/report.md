# Part C2: multiscale marker-free period detector

## Detector contract

The detector reads only the five timestamped GPU metric channels. It uses 15-1000 ms dynamic search, four bin widths, three tail fractions, cross-configuration clustering, harmonic reconciliation, phase-fold repeatability, and cross-horizon stability. No marker, callback, baseline runtime, target runtime, or workload metadata is read by the detector.

The old Part 7 and Part C programs and outputs are unchanged.

The deployment policy emits early at 9 seconds only when confidence is at least 0.90. Otherwise it waits for a 12/15-second confirmation at confidence 0.65. If a legacy capture ends before 12 seconds, the last available detection may be emitted at the confirmed threshold.

## YOLO same-run marker evaluation

- Profiles: 48
- Accepted: 48 (100.0%)
- Mean period APE: 1.26%
- Median period APE: 0.93%
- P90 period APE: 2.88%
- Max period APE: 7.23%
- Within 10%: 100.0%

## Gemma profile-period versus separate unprofiled baseline

- Profiles: 18
- Accepted: 18 (100.0%)
- Runtime-eligible accepted: 16/16 (100.0%)
- Feature table: 18 rows x 93 columns
- Mean absolute gap: 12.87%
- Median absolute gap: 4.45%
- P90 absolute gap: 32.80%
- Max absolute gap: 33.47%
- Within 10%: 62.5%

This Gemma gap is not a pure detector error: the detected period is measured during Nsight GPU-metric profiling, while the baseline is from a separate unprofiled run.

## Gemma audit

| Workload | Accepted | Horizon | Confidence | Profile period ms | Unprofiled baseline ms | Ratio | Runtime eligible |
|---|---:|---:|---:|---:|---:|---:|---:|
| G01 | True | 9 | 0.959 | 184.50 | 139.33 | 1.324 | True |
| G02 | True | 12 | 0.863 | 187.00 | 140.11 | 1.335 | True |
| G03 | True | 9 | 0.953 | 196.00 | 184.07 | 1.065 | True |
| G04 | True | 9 | 0.954 | 190.00 | 146.36 | 1.298 | True |
| G05 | True | 9 | 0.953 | 205.50 | 195.84 | 1.049 | True |
| G06 | True | 9 | 0.985 | 340.00 | 335.93 | 1.012 | True |
| G07 | True | 9 | 0.958 | 214.00 | 211.03 | 1.014 | True |
| G08 | True | 9 | 0.977 | 375.50 | 369.38 | 1.017 | True |
| G09 | True | 12 | 1.000 | 712.00 | nan | nan | False |
| G10 | True | 9 | 0.964 | 184.00 | 138.15 | 1.332 | True |
| G11 | True | 9 | 0.957 | 185.50 | 141.60 | 1.310 | True |
| G12 | True | 9 | 0.954 | 206.00 | 198.14 | 1.040 | True |
| G13 | True | 9 | 0.954 | 190.00 | 152.53 | 1.246 | True |
| G14 | True | 9 | 0.964 | 223.00 | 221.36 | 1.007 | True |
| G15 | True | 9 | 0.967 | 378.50 | 378.84 | 0.999 | True |
| G16 | True | 9 | 0.960 | 250.00 | 251.06 | 0.996 | True |
| G17 | True | 9 | 1.000 | 432.00 | 434.63 | 0.994 | True |
| G18 | True | 12 | 0.721 | 777.00 | nan | nan | False |