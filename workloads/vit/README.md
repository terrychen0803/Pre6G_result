# ViT cross-hardware training runtime collection

A deterministic Vision Transformer image-classification workload family for
marker-free cross-device training-step runtime prediction. It provides an
encoder-attention family distinct from YOLO detection and Gemma decoder LLM
fine-tuning.

## Workload design

Nine structural configurations are paired with FP16 and BF16 autocast for 18
workloads. The matrix varies image/token count, patch projection, embedding
width, encoder depth, attention heads, and batch size.

- Image sizes: 128, 224, 384
- Patch sizes: 16, 32
- Embedding dimensions: 192, 384, 768
- Encoder depths: 4, 6, 8
- Attention heads: 3, 6, 12
- Batch sizes: 8, 16, 32
- MLP ratio: 4
- Optimizer: AdamW
- Parameters/master optimizer state: FP32
- Compute autocast: FP16 or BF16
- Data: deterministic pre-generated pinned synthetic images

The model is implemented with native PyTorch Conv2d patch embedding and
independently initialized TransformerEncoderLayer blocks; no model or dataset
download is required. This isolates GPU training compute from disk/tokenization
pipelines.

## Files

- `configs/experiments.csv`: workload matrix.
- `configs/devices.json`: current cross-node hardware audit record; fill null
  fields after measuring each node.
- `scripts/generate_dataset.py`: deterministic synthetic images.
- `scripts/train_vit.py`: native PyTorch ViT workload.
- `scripts/run_baseline.py`: source/target labels.
- `scripts/run_nsys.py`: deployment/oracle profiles.
- `scripts/extract_nsys_features.py`: 1 ms marker-free cycle features.
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
  --delay 5 --duration 15 --deployment-iters 2000

python scripts/extract_nsys_features.py \\
  --device-id RTX5090 \\
  --workloads all \\
  --collection-mode deployment \\
  --sample-interval-ms 1

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

Before running all workloads, execute V01, V05, V07, and V09 to check runtime,
VRAM support, capture delay, and period confidence on both devices. Do not
reduce an OOM workload in place; mark it unsupported or create a new workload
ID so the experimental contract remains auditable.

The current 5090/4090 comparison is cross-node, not GPU-only: the hosts use
different CPUs. Preserve the detected `environment.json` files and report this
as a confounder until same-host or additional-device experiments are available.

## Oracle validation

```bash
python scripts/run_nsys.py --device-id RTX5090 --workloads V01,V05,V07,V09 --collection-mode oracle
python scripts/extract_nsys_features.py --device-id RTX5090 --workloads V01,V05,V07,V09 --collection-mode oracle
```

Oracle artifacts are validation-only and cannot enter the prediction table.
