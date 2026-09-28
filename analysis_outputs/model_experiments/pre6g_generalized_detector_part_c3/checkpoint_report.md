# Part C3 checkpoint: generalized dual-resolution detector

## Scope and data split

- Development families: YOLO26 and Gemma 4.
- First frozen external family: dense MLP (DNN).
- Untouched final family: Vision Transformer (ViT).
- Source profiling device: RTX5090.
- Target runtime-label device: RTX4090.
- The blind detector files are written before baseline/oracle labels are joined.

## Detector

The new detector is independent of Part 7, Part C, and Part C2. It has two branches:

1. A 1 ms five-channel GPU-metric branch for periods from 15 to 1000 ms.
2. A 0.05 ms CUDA-kernel-event branch for periods from 0.2 to 15 ms.

The event branch hashes kernel short-name identifiers into eight activity channels and requires ACF prominence, cross-bucket agreement, and 3/5-second stability. Routing uses signal evidence only; it does not use model name, workload configuration, marker, callback, baseline, or target runtime.

## Development checks

| Family | Profiles | Event branch | Metric branch | Evaluation |
|---|---:|---:|---:|---|
| YOLO26 | 48 | 0 | 48 | Same-run marker mean APE 1.26%, max 7.23%, 100% within 10% |
| Gemma 4 | 18 | 0 | 18 | Separate-run profile-vs-baseline median gap 4.45% |

The fine event branch did not replace the correct long-period branch on either development family.

## Frozen DNN external-family result

| Metric | Result |
|---|---:|
| Profiles | 18 |
| Routed to CUDA-event branch | 18/18 |
| Routed to GPU-metric branch | 0/18 |
| Mean top-1 gap vs separate unprofiled source baseline | 44.97% |
| Median top-1 gap | 23.04% |
| P90 top-1 gap | 113.30% |
| Top-1 within 10% | 38.89% |
| Best-of-top-3 median gap (oracle diagnostic only) | 10.00% |
| Best-of-top-3 within 10% (oracle diagnostic only) | 50.00% |

D09, D11, and D15 contain a baseline-scale candidate at rank 2, rank 2, and rank 3 respectively, showing a harmonic-ranking problem. Several other short workloads have a stable detected profile period that is longer than the separately collected unprofiled baseline. Without same-run markers, that difference cannot be separated into detection error and profiler overhead.

## Decision

Part C3 demonstrates routing generalization: long-period YOLO/Gemma profiles remain on GPU metrics while all sub-15-ms DNN profiles move to fine CUDA events. It does not yet demonstrate sufficiently accurate top-1 sub-millisecond iteration recovery.

ViT remains untouched. The next scientifically clean choice is either:

1. preserve the frozen detector and run ViT as a second external test, or
2. use DNN as new development data to train a harmonic ranker, then use ViT as the sole untouched final test.

Because DNN labels have now been inspected, any DNN-informed change must classify DNN as development data rather than external test data.
