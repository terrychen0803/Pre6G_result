# Environment setup

## Analysis environment

Python 3.10 or newer is required. Create an isolated environment from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Linux/macOS activation:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

GPU workload collection has workload-specific dependencies. Install the relevant file instead of forcing every model stack into one environment:

- `workloads/dnn/requirements.txt`
- `workloads/gemma4/requirements.txt`
- `workloads/vit/requirements.txt`
- YOLO26 additionally requires the compatible PyTorch, Ultralytics and CUDA stack recorded in `configs/yolo26/software_stack.json`.

## External raw data

Raw collections are intentionally not committed. Point the analysis tools to their parent directory with `PRE6G_DATA_ROOT`, or pass `--runs-root` to commands that support it.

```powershell
$env:PRE6G_DATA_ROOT = "D:\research-data\pre6g"
```

The default is `data/raw`. Expected layouts include:

```text
data/raw/
├─ RTX4090_yolo26_collection_20260817/runs/
├─ RTX5090_yolo26_collection_20260817/runs/
└─ high_load_01_RTX5090_20260923/runs_loaded/
```

If `nsys` is not available on `PATH`, set `PRE6G_NSYS` to the Nsight Systems CLI executable.

## Optional CJK font

Figure scripts use `PRE6G_CJK_FONT` when set, then try the Windows Microsoft JhengHei font, and finally fall back to Matplotlib's sans-serif font.

```powershell
$env:PRE6G_CJK_FONT = "C:\Windows\Fonts\msjh.ttc"
```

## Publication validation

The repository-only validation does not require third-party packages:

```powershell
python scripts/validate_publication.py
```

It checks Python syntax, Notebook JSON, required entrypoints, nested Git directories, oversized files, forbidden runtime artifacts and local workspace paths.
