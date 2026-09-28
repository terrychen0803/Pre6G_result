# Pre6G Gemma 4 Part C: frozen marker-free period transfer

## Experimental boundary

The detector configuration and confidence threshold are trained only on the YOLO Part 7 dataset, then frozen before Gemma is evaluated. Gemma baseline runtimes are joined only after detection for held-out family evaluation; they never enter period detection or feature extraction.

## Frozen YOLO-trained detector

| Horizon | Configuration | YOLO profiles | Mean APE | Median APE | Max APE |
|---:|---|---:|---:|---:|---:|
| 9s | `{"bin_ms": 1.0, "tail_fraction": 0.65}` | 48 | 10.14% | 1.13% | 104.44% |
| 12s | `{"bin_ms": 2.0, "tail_fraction": 0.5}` | 29 | 1.04% | 0.83% | 4.32% |
| 15s | `{"bin_ms": 1.0, "tail_fraction": 0.8}` | 20 | 0.93% | 0.55% | 3.15% |

- Frozen confidence threshold: 0.40
- YOLO training-policy coverage: 48/48 (100.0%)
- YOLO accepted mean period APE: 1.01%
- YOLO accepted max period APE: 5.61%

## Gemma external-family result

- Profiles: 18
- Accepted: 11 (61.1%)
- Runtime-regression-eligible acceptance: 11/16 (68.8%)
- Feature rows: 11 rows x 93 columns
- Emitted at 9s: 11
- Profiles with held-out source baseline evaluation: 11
- Mean period APE: 18.22%
- Median period APE: 23.91%
- P90 period APE: 33.47%
- Max period APE: 33.91%
- Within 10%: 45.5%

## Per-workload audit

| Workload | Accepted | Horizon | Confidence | Detected ms | Hidden source baseline ms | APE | Runtime eligible |
|---|---:|---:|---:|---:|---:|---:|---:|
| G01 | True | 9 | 0.709 | 185.00 | 139.33 | 32.78% | True |
| G02 | True | 9 | 0.649 | 187.00 | 140.11 | 33.47% | True |
| G03 | True | 9 | 0.654 | 196.00 | 184.07 | 6.48% | True |
| G04 | True | 9 | 0.625 | 190.00 | 146.36 | 29.82% | True |
| G05 | True | 9 | 0.642 | 205.00 | 195.84 | 4.68% | True |
| G06 | False | nan | 0.097 | 32.00 | 335.93 | 90.47% | True |
| G07 | True | 9 | 0.632 | 213.00 | 211.03 | 0.93% | True |
| G08 | False | nan | 0.072 | 36.00 | 369.38 | 90.25% | True |
| G09 | False | nan | 0.218 | 70.00 | nan | nan% | False |
| G10 | True | 9 | 0.589 | 185.00 | 138.15 | 33.91% | True |
| G11 | True | 9 | 0.778 | 185.00 | 141.60 | 30.65% | True |
| G12 | True | 9 | 0.681 | 205.00 | 198.14 | 3.46% | True |
| G13 | True | 9 | 0.690 | 189.00 | 152.53 | 23.91% | True |
| G14 | True | 9 | 0.596 | 222.00 | 221.36 | 0.29% | True |
| G15 | False | nan | 0.100 | 37.00 | 378.84 | 90.23% | True |
| G16 | False | nan | 0.061 | 245.00 | 251.06 | 2.41% | True |
| G17 | False | nan | 0.223 | 36.00 | 434.63 | 91.72% | True |
| G18 | False | nan | 0.289 | 63.00 | nan | nan% | False |