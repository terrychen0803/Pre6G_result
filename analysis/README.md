# Analysis pipeline

The scripts preserve raw collection data and write derived tables under
`analysis_outputs/`. Large Nsight SQLite exports are cached under
`analysis_cache/nsys/`.

## Portable Nsight Systems CLI

The workspace-local CLI is NVIDIA Nsight Systems 2026.4.1, matching the report
producer version exactly. It is not added to the system `PATH`.

```powershell
.\analysis\nsys.cmd --version
```

The verified installer retained in `analysis_cache/installers/` has SHA-256:

```text
3D28E92B23FC28508E38E9DC7268168F4083F9EDA613D58252BE1DBE6B49E103
```

Its Authenticode signature was valid and its MSI product metadata identifies
NVIDIA Nsight Systems 2026.4.1. The CLI was reconstructed from the MSI's
`Target-windows-x64` component because machine-wide installation requires
administrator privileges.

The extracted `nsys.exe` also has a valid NVIDIA Authenticode signature and
SHA-256 `E3C19D7C2F96A218491D5363B2D21748A874E9B9FCBEE718AAB60ACC8188E380`.

## Reproduce the current derived data

```powershell
python analysis/scan_runs.py
python analysis/aggregate_ground_truth.py
python analysis/analyze_iterations.py
python analysis/analyze_profiler_overhead.py
python analysis/export_nsys.py --mode nsys_trace --mode nsys_gpu_metrics
python analysis/extract_nsys_trace.py
python analysis/extract_nsys_gpu_metrics.py
python analysis/inspect_hpctoolkit_postprocess.py
python analysis/build_workload_features.py
python analysis/build_cross_device_dataset.py
python analysis/analyze_correlations.py
python analysis/generate_figures.py
python analysis/generate_report.py
```

Missing profiler tables or metrics remain missing. In particular, the current
CPU-metrics report does not expose PMU counters, so IPC and miss-rate features
are not synthesized.

## YOLO single-device profiling-overhead model

Train and evaluate one marker-free Nsight GPU Metrics overhead model per known
device. Cycle features are rebuilt from the current multiscale YOLO detector:

```powershell
python analysis/run_single_device_overhead_models.py
```

Results and deployable model bundles are written under
`analysis_outputs/model_experiments/yolo_single_device_overhead/`. The outer
evaluation leaves one YOLO workload out within each device; feature-family and
Ridge-alpha selection use only inner grouped folds. The model predicts the
profile-to-dry runtime ratio and constrains deployment overhead to be
non-negative.

Run inference with a generated example:

```powershell
python analysis/predict_single_device_overhead.py `
  --model analysis_outputs/model_experiments/yolo_single_device_overhead/models/RTX5090_nsys_gpu_metrics_overhead_ridge.joblib `
  --input-csv analysis_outputs/model_experiments/yolo_single_device_overhead/examples/RTX5090_nsys_gpu_metrics_example.csv `
  --output-csv analysis_outputs/model_experiments/yolo_single_device_overhead/examples/RTX5090_example_prediction.csv
```

## YOLO single-device direct dry-runtime model

The direct-target variant predicts `log(dry_runtime_ms)` from the same current
multiscale marker-free features, then derives overhead from the detected
profiling period:

```powershell
python analysis/run_single_device_dry_runtime_models.py
```

Results and deployable bundles are written under
`analysis_outputs/model_experiments/yolo_single_device_dry_runtime/`.

```powershell
python analysis/predict_single_device_dry_runtime.py `
  --model analysis_outputs/model_experiments/yolo_single_device_dry_runtime/models/RTX5090_nsys_gpu_metrics_dry_runtime_ridge.joblib `
  --input-csv analysis_outputs/model_experiments/yolo_single_device_dry_runtime/examples/RTX5090_nsys_gpu_metrics_example.csv `
  --output-csv analysis_outputs/model_experiments/yolo_single_device_dry_runtime/examples/RTX5090_example_prediction.csv
```
