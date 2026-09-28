# Profiling and data contract

## Collection modes

- **baseline** runs on every source/target device and is the only runtime/VRAM
  label source. It creates no per-iteration CUDA event and synchronizes only at
  the two edges of the complete measured window.
- **deployment** runs on the source feature device. Nsight controls the capture
  externally through delay/duration. The training process receives no NVTX,
callback, iteration marker, or CUDA profiler-range request.
  The runner explicitly sends SIGTERM when the external duration ends; a
  deployment training summary is therefore not expected or used.
- **oracle** uses NVTX, CUDA events, and cudaProfilerApi only to validate the
  marker-free period detector. Oracle values are rejected by the table builder.

## Schema boundary

- `x_profile.*`: marker-free semantic low-level time-series features.
- `x_hardware.*`: numeric source/target hardware descriptors.
- `y_target_*`: profiler-free target labels.
- `audit_*`: workload configuration, names, source baseline, fingerprints, and
  diagnostics. Never use these columns as predictor inputs.

The primary target is `y_target_iteration_median_ms`. Peak VRAM is retained as
`y_target_peak_vram_bytes` for a future task-specific capacity predictor.

## Semantic signals

The extractor maps compatible Nsight metric fields to canonical names such as:

- `profile.gpu.sm_active`
- `profile.gpu.tensor_active`
- `profile.gpu.dram_read`
- `profile.gpu.dram_write`
- `profile.gpu.memory_active`

It always derives portable CUDA-trace fallback signals:

- `profile.gpu.kernel_active`
- `profile.gpu.kernel_launch_rate`
- `profile.gpu.memcpy_active`
- `profile.gpu.memcpy_bytes`

Raw profiler table/column names are kept out of model-input column names.

## Reproducibility rules

Use the same experiment CSV, model config, generated dataset manifest, PyTorch,
CUDA, and code revision on both nodes. Archive old results before changing any
of these controlled inputs. Unsupported/OOM configurations must be recorded as
unsupported; do not silently shrink them.

A deployment capture must contain at least three complete cycles. Tune external
capture delay/duration and the deployment-only iteration override before the
full run. Validate period estimates against oracle runs on a representative
subset, then perform capture-offset and sampling-frequency sensitivity checks.
