from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from common import DEFAULT_EXPERIMENTS, DEFAULT_RUNS_ROOT, PROJECT_ROOT, flatten_numeric, load_json, load_workloads, parse_workload_ids, sanitize_name

DEFAULT_OUTPUT = PROJECT_ROOT / "prediction" / "rtx5090_to_rtx4090_vit.csv"


def load_baselines(runs_root: Path, device: str, workload_id: str, repeats: int):
    summaries, metadata = [], []
    for repeat in range(1, repeats + 1):
        run_dir = runs_root / device / workload_id / f"baseline_{repeat:02d}"
        summary_path, metadata_path = run_dir / "summary.json", run_dir / "metadata.json"
        if not summary_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Incomplete baseline: {run_dir}")
        summary = load_json(summary_path)
        if summary.get("run_type") != "baseline" or summary.get("instrumentation") != "window_only":
            raise ValueError(f"Invalid low-instrumentation baseline: {summary_path}")
        summaries.append(summary)
        metadata.append(load_json(metadata_path))
    return summaries, metadata


def fingerprint(metadata: dict[str, Any]) -> str:
    controlled = {
        "model_config": metadata.get("model_config"),
        "dataset_manifest": metadata.get("dataset_manifest"),
        "torch_version": metadata.get("torch_version"),
        "torch_cuda_version": metadata.get("torch_cuda_version"),
    }
    payload = json.dumps(controlled, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def aggregate(summaries):
    runtime = [float(item["steady_window_mean_iter_ms"]) for item in summaries]
    vram = [float(item["peak_vram_bytes"]) for item in summaries]
    return {
        "runtime_mean": statistics.mean(runtime),
        "runtime_median": statistics.median(runtime),
        "runtime_std": statistics.stdev(runtime) if len(runtime) > 1 else 0.0,
        "peak_vram_median": statistics.median(vram),
    }


def profile_inputs(features: dict[str, Any]) -> dict[str, float]:
    if features.get("collection_mode") != "deployment":
        raise ValueError("Only deployment profiles are legal prediction inputs")
    selected = features.get("deployment_features")
    if not isinstance(selected, dict):
        raise ValueError("Missing deployment_features")
    return flatten_numeric(selected, "x_profile")


def hardware_inputs(prefix: str, metadata: dict[str, Any]) -> dict[str, float]:
    result = {
        f"x_hardware.{prefix}.total_memory_bytes": float(metadata["gpu_total_memory_bytes"]),
        f"x_hardware.{prefix}.multiprocessor_count": float(metadata["gpu_multiprocessor_count"]),
    }
    capability = metadata.get("gpu_compute_capability") or []
    if len(capability) == 2:
        result[f"x_hardware.{prefix}.compute_capability_major"] = float(capability[0])
        result[f"x_hardware.{prefix}.compute_capability_minor"] = float(capability[1])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a leakage-safe cross-device prediction table")
    parser.add_argument("--source-device", default="RTX5090")
    parser.add_argument("--target-device", default="RTX4090")
    parser.add_argument("--workloads", default="all")
    parser.add_argument("--baseline-repeats", type=int, default=5)
    parser.add_argument("--profile-repeat", type=int, default=1)
    parser.add_argument("--min-period-confidence", type=float, default=0.5)
    parser.add_argument("--experiments", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-stack-mismatch", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.baseline_repeats < 1 or not 0 <= args.min_period_confidence <= 1:
        raise ValueError("Invalid repeats or confidence")

    source, target = sanitize_name(args.source_device), sanitize_name(args.target_device)
    runs_root = args.runs_root.resolve()
    workloads = load_workloads(args.experiments.resolve())
    workload_ids = parse_workload_ids(args.workloads, set(workloads))
    rows, skipped = [], []
    for workload_id in workload_ids:
        try:
            feature_path = runs_root / source / workload_id / f"nsys_deployment_{args.profile_repeat:02d}" / "profile_features.json"
            if not feature_path.exists():
                raise FileNotFoundError(feature_path)
            features = load_json(feature_path)
            confidence = float(features.get("deployment_features", {}).get("period_detection", {}).get("confidence", 0.0))
            if confidence < args.min_period_confidence:
                raise ValueError(f"Period confidence {confidence:.3f} is below threshold")
            source_summaries, source_metadata = load_baselines(runs_root, source, workload_id, args.baseline_repeats)
            target_summaries, target_metadata = load_baselines(runs_root, target, workload_id, args.baseline_repeats)
            source_fingerprints = {fingerprint(item) for item in source_metadata}
            target_fingerprints = {fingerprint(item) for item in target_metadata}
            if len(source_fingerprints) != 1 or len(target_fingerprints) != 1:
                raise ValueError("Stack changed between repeats")
            if source_fingerprints != target_fingerprints and not args.allow_stack_mismatch:
                raise ValueError("Source and target controlled stacks differ")

            source_values, target_values = aggregate(source_summaries), aggregate(target_summaries)
            workload = workloads[workload_id]
            row: dict[str, Any] = {
                "audit_workload_id": workload_id,
                "audit_source_device_id": source,
                "audit_target_device_id": target,
                "audit_source_gpu_name": source_metadata[0]["gpu_name"],
                "audit_target_gpu_name": target_metadata[0]["gpu_name"],
                "audit_controlled_stack_fingerprint": next(iter(source_fingerprints)),
                "audit_source_runtime_median_ms": source_values["runtime_median"],
                "audit_source_runtime_std_ms": source_values["runtime_std"],
                "y_target_iteration_mean_ms": target_values["runtime_mean"],
                "y_target_iteration_median_ms": target_values["runtime_median"],
                "y_target_peak_vram_bytes": target_values["peak_vram_median"],
            }
            for key, value in workload.items():
                row[f"audit_config_{key}"] = value
            row.update(hardware_inputs("source", source_metadata[0]))
            row.update(hardware_inputs("target", target_metadata[0]))
            row.update(profile_inputs(features))
            rows.append(row)
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exception:
            if not args.allow_incomplete:
                raise
            skipped.append({"workload_id": workload_id, "reason": str(exception)})
    if not rows:
        raise SystemExit("No complete rows")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": 1,
        "source_device": source,
        "target_device": target,
        "rows": len(rows),
        "skipped": skipped,
        "profile_contract": "deployment_marker_free_semantic",
        "model_input_prefix": "x_",
        "audit_prefix": "audit_",
        "primary_target": "y_target_iteration_median_ms",
        "secondary_target": "y_target_peak_vram_bytes",
        "output_csv": str(output),
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
