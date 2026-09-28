from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from common import (
    DEFAULT_EXPERIMENTS,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_RUNS_ROOT,
    load_json,
    load_workload,
    sanitize_name,
)

ModelFactory = Callable[[Any, Any, dict[str, Any], dict[str, Any]], Any]
DatasetLoader = Callable[[Any, Path, dict[str, Any]], tuple[Any, Any, dict[str, Any]]]


def create_parser(description: str, default_dataset_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--workload-id", required=True)
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--run-type",
        choices=["baseline", "nsys_oracle", "nsys_deployment", "hpctoolkit_cuda", "hpctoolkit_cuda_trace"],
        default="baseline",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--iteration-override", type=int, default=None)
    parser.add_argument("--nvtx", action="store_true")
    parser.add_argument("--cuda-profiler-range", action="store_true")
    parser.add_argument("--experiments", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    return parser


def select_rows(tensor, start: int, count: int):
    total = tensor.shape[0]
    if start + count <= total:
        return tensor.narrow(0, start, count)
    import torch
    first = tensor.narrow(0, start, total - start)
    second = tensor.narrow(0, 0, count - first.shape[0])
    return torch.cat((first, second), dim=0).pin_memory()


def create_scaler(torch_module, enabled: bool):
    try:
        return torch_module.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch_module.cuda.amp.GradScaler(enabled=enabled)


def execute_training(
    args: argparse.Namespace,
    model_factory: ModelFactory,
    dataset_loader: DatasetLoader,
) -> None:
    workload = load_workload(args.experiments.resolve(), args.workload_id)
    model_config = load_json(args.model_config.resolve())
    oracle_mode = args.run_type == "nsys_oracle"
    if (args.nvtx or args.cuda_profiler_range) and not oracle_mode:
        raise ValueError("NVTX/cudaProfilerApi are allowed only for nsys_oracle")
    if args.iteration_override is not None:
        if args.run_type != "nsys_deployment":
            raise ValueError("--iteration-override is allowed only for nsys_deployment")
        if args.iteration_override <= int(workload["warmup_iters"]):
            raise ValueError("--iteration-override must exceed warmup_iters")
    execution_total = args.iteration_override or int(workload["total_iters"])

    import torch
    import torch.nn as nn

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    seed = int(model_config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dtype_name = str(workload["dtype"])
    if dtype_name == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Native BF16 is unavailable on this GPU")
        autocast_dtype = torch.bfloat16
    elif dtype_name == "fp16":
        autocast_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported dtype: {dtype_name}")

    inputs, labels, dataset_manifest = dataset_loader(
        torch, args.dataset_root.resolve(), workload
    )
    inputs = inputs.pin_memory()
    labels = labels.pin_memory()
    if int(workload["batch_size"]) > inputs.shape[0]:
        raise ValueError("Dataset must contain at least one complete batch")

    model = model_factory(torch, nn, workload, model_config).to(device)
    model.train()
    optimizer_config = model_config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        eps=float(optimizer_config["eps"]),
        fused=bool(optimizer_config["fused"]),
    )
    optimizer.zero_grad(set_to_none=True)
    scaler = create_scaler(torch, enabled=dtype_name == "fp16")
    criterion = nn.CrossEntropyLoss()

    device_label = sanitize_name(args.device_id)
    output_dir = (
        args.runs_root.resolve()
        / device_label
        / str(workload["workload_id"])
        / f"{args.run_type}_{args.repeat:02d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    properties = torch.cuda.get_device_properties(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    metadata = {
        "schema_version": 1,
        "task_class": model_config["task_class"],
        "workload_family": model_config["workload_family"],
        "workload_name": model_config["workload_name"],
        "framework": model_config["framework"],
        "mode": model_config["mode"],
        "workload": workload,
        "model_config": model_config,
        "dataset_manifest": dataset_manifest,
        "device_id": device_label,
        "device_argument": args.device,
        "run_type": args.run_type,
        "repeat": args.repeat,
        "execution_total_iters": execution_total,
        "instrumentation_policy": "oracle_per_iteration" if oracle_mode else "window_only" if args.run_type == "baseline" else "none",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": properties.name,
        "gpu_total_memory_bytes": properties.total_memory,
        "gpu_multiprocessor_count": properties.multi_processor_count,
        "gpu_compute_capability": list(torch.cuda.get_device_capability(device)),
        "parameter_count": parameter_count,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    warmup_iters = int(workload["warmup_iters"])
    batch_size = int(workload["batch_size"])
    window_start_ns: int | None = None
    window_end_ns: int | None = None
    profiler_active = False
    oracle_records: list[tuple[int, Any, Any]] = []
    final_loss = None
    torch.cuda.reset_peak_memory_stats(device)

    try:
        for step in range(1, execution_total + 1):
            if step == warmup_iters + 1 and args.run_type in {"baseline", "nsys_oracle"}:
                torch.cuda.synchronize()
                window_start_ns = time.perf_counter_ns()
                if args.cuda_profiler_range:
                    torch.cuda.cudart().cudaProfilerStart()
                    profiler_active = True

            batch_start = ((step - 1) * batch_size) % inputs.shape[0]
            start_event = torch.cuda.Event(enable_timing=True) if oracle_mode and step > warmup_iters else None
            end_event = torch.cuda.Event(enable_timing=True) if oracle_mode and step > warmup_iters else None
            if args.nvtx:
                torch.cuda.nvtx.range_push(f"TRAIN_ITER_{step:05d}")
            if start_event is not None:
                start_event.record()

            batch_inputs = select_rows(inputs, batch_start, batch_size).to(device, non_blocking=True)
            batch_labels = select_rows(labels, batch_start, batch_size).to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                logits = model(batch_inputs)
                loss = criterion(logits, batch_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if end_event is not None:
                end_event.record()
                oracle_records.append((step, start_event, end_event))
            if args.nvtx:
                torch.cuda.nvtx.range_pop()
            final_loss = loss.detach()

            if step == execution_total and args.run_type in {"baseline", "nsys_oracle"}:
                torch.cuda.synchronize()
                window_end_ns = time.perf_counter_ns()
                if profiler_active:
                    torch.cuda.cudart().cudaProfilerStop()
                    profiler_active = False

        if final_loss is None:
            raise RuntimeError("Training loop did not execute")
        final_loss_value = float(final_loss.float().cpu())
        if not math.isfinite(final_loss_value):
            raise FloatingPointError("Non-finite final loss")

        measurement_iters = execution_total - warmup_iters
        window_total_ms = (
            (window_end_ns - window_start_ns) / 1e6
            if window_start_ns is not None and window_end_ns is not None
            else None
        )
        mean_ms = window_total_ms / measurement_iters if window_total_ms else None
        instrumentation = "oracle_per_iteration" if oracle_mode else "window_only" if args.run_type == "baseline" else "none"
        summary = {
            "schema_version": 1,
            "workload_id": workload["workload_id"],
            "device_id": device_label,
            "run_type": args.run_type,
            "repeat": args.repeat,
            "instrumentation": instrumentation,
            "measurement_iters": measurement_iters,
            "steady_window_total_ms": window_total_ms,
            "steady_window_mean_iter_ms": mean_ms,
            "final_loss": final_loss_value,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
            "parameter_count": parameter_count,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        if oracle_records:
            values = []
            for step, start_event, end_event in oracle_records:
                values.append({"iteration_id": step, "gpu_total_ms": start_event.elapsed_time(end_event)})
            with (output_dir / "iterations.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["iteration_id", "gpu_total_ms"])
                writer.writeheader()
                writer.writerows(values)
            summary["oracle_gpu_median_ms"] = statistics.median(item["gpu_total_ms"] for item in values)
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except Exception as exception:
        if profiler_active:
            try:
                torch.cuda.cudart().cudaProfilerStop()
            except Exception:
                pass
        (output_dir / "failure.json").write_text(
            json.dumps({"type": type(exception).__name__, "message": str(exception)}, indent=2),
            encoding="utf-8",
        )
        raise

    print(f"Experiment complete: {output_dir}")
