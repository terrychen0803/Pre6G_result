# DNN cross-hardware training runtime collection

A deterministic dense MLP classification workload family for marker-free
cross-device training-step runtime prediction. It complements YOLO (CNN
Detection), Gemma (decoder LLM), and ViT (encoder Transformer).

## Workload design

Nine structural configurations are paired with FP16 and BF16 autocast for 18
workloads. The representative design varies input dimension, hidden width,
depth, and batch size without creating a full Cartesian explosion.

- Input dimensions: 512, 2048, 8192
- Hidden widths: 512, 2048, 4096
- Hidden blocks: 2, 4, 8
- Batch sizes: 64, 256, 1024
- Class count: 1000
- Block: Linear → LayerNorm → GELU
- Optimizer: AdamW
- Parameters/master optimizer state: FP32
- Compute autocast: FP16 or BF16
- Data: deterministic pre-generated pinned CPU tensors

This is a controlled GEMM/activation-dominant GPU compute workload, not an
end-to-end data-loading benchmark.

## Files

- `configs/experiments.csv`: workload matrix.
- `configs/devices.json`: current cross-node hardware audit record; fill null
  fields after measuring each node.
- `scripts/generate_dataset.py`: deterministic synthetic data.
- `scripts/train_dnn.py`: low-instrumentation training workload.
- `scripts/run_baseline.py`: source/target labels.
- `scripts/run_nsys.py`: deployment/oracle profiles.
- `scripts/extract_nsys_features.py`: 0.25 ms marker-free cycle features.
- `scripts/run_hpctoolkit.py`: optional auxiliary collection.
- `scripts/build_prediction_table.py`: leakage-safe paired table.
- `PROFILING_CONTRACT.md`: mode and schema rules.

## Collection

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_dataset.py

# Run separately on each node.
python scripts/run_baseline.py --device-id RTX5090 --workloads all --repeats 5
python scripts/run_baseline.py --device-id RTX4090 --workloads all --repeats 5

# Source feature node only.
python scripts/run_nsys.py \\
  --device-id RTX5090 \\
  --workloads all \\
  --collection-mode deployment \\
  --gpu-metrics-frequency 1000 \\
  --delay 5 --duration 15 --deployment-iters 30000

python scripts/extract_nsys_features.py \\
  --device-id RTX5090 \\
  --workloads all \\
  --collection-mode deployment \\
  --sample-interval-ms 0.25

python scripts/build_prediction_table.py \\
  --source-device RTX5090 \\
  --target-device RTX4090 \\
  --baseline-repeats 5
```

> **Change note (2026-08-26):** deployment profiling keeps
> `--kill=sigterm` so every capture has a fixed external duration. Nsight may
> return `143` or `-15` when it terminates the target after a successful
> capture. The runner accepts only those two codes in deployment mode and only
> when `profile.nsys-rep` exists and is non-empty. Oracle runs and all other
> non-zero return codes remain failures.

To resume a previous `--workloads all` run that stopped on return code 143,
rerun the same command without `--force`; completed reports are skipped.

DNN steps may be much shorter than Gemma/ViT steps. Start with D01, D05, and D09
and verify that 0.25 ms trace bins resolve at least three cycles. GPU Metrics
remain sampled at the configured Nsight frequency; sub-millisecond resolution
mainly comes from CUDA kernel intervals.

The current 5090/4090 comparison is cross-node, not GPU-only: the hosts use
different CPUs. Preserve the detected `environment.json` files and report this
as a confounder until same-host or additional-device experiments are available.

## Oracle validation

```bash
python scripts/run_nsys.py --device-id RTX5090 --workloads D01,D05,D09 --collection-mode oracle
python scripts/extract_nsys_features.py --device-id RTX5090 --workloads D01,D05,D09 --collection-mode oracle --sample-interval-ms 0.25
```

Oracle artifacts are validation-only and cannot enter the prediction table.
