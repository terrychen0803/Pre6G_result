from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_EXPERIMENTS,
    DEFAULT_OUTPUT_DIR,
    PROJECT_ROOT,
    discover_runs_roots,
    first_nested_value,
    iter_run_directories,
    load_experiments,
    load_json,
    parse_bool,
    relative_path,
    write_csv,
)


INVENTORY_FIELDS = [
    "collection_root",
    "device_id",
    "workload_id",
    "run_type",
    "repeat",
    "run_directory",
    "summary_json_exists",
    "summary_json_valid",
    "metadata_json_exists",
    "metadata_json_valid",
    "iterations_csv_exists",
    "run_log_exists",
    "profiler_log_exists",
    "nsys_report_exists",
    "nsys_report_bytes",
    "hpctoolkit_measurement_exists",
    "hpctoolkit_database_exists",
    "model",
    "batch",
    "imgsz",
    "amp",
    "nbs",
    "max_iters",
    "warmup_iters",
    "device",
    "gpu_name",
    "cuda_version",
    "pytorch_version",
    "ultralytics_version",
    "nsight_version",
    "dataset_hash",
    "metadata_run_type",
    "requested_iterations",
    "actual_iterations",
    "valid_iterations",
    "steady_window_mean_iter_ms",
    "config_matches_experiment",
    "parse_errors",
]


REQUIRED_MAIN_TYPES = {
    "baseline": 3,
    "nsys_trace": 1,
    "nsys_gpu_metrics": 1,
    "nsys_cpu_metrics": 1,
}


def count_csv_rows(path: Path) -> tuple[int | None, str | None]:
    if not path.exists():
        return None, None
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            return sum(1 for _ in reader), None
    except Exception as exc:
        return None, f"iterations.csv: {type(exc).__name__}: {exc}"


def run_files(run_dir: Path) -> dict[str, Any]:
    measurement_dirs = [
        path
        for path in run_dir.iterdir()
        if path.is_dir()
        and (path.name == "hpctoolkit-measurements" or path.name.endswith("-measurements"))
    ]
    database_dirs = [
        path
        for path in run_dir.iterdir()
        if path.is_dir()
        and (path.name == "hpctoolkit-database" or path.name.endswith("-database"))
    ]
    report = run_dir / "profile.nsys-rep"
    return {
        "summary_json_exists": (run_dir / "summary.json").exists(),
        "metadata_json_exists": (run_dir / "metadata.json").exists(),
        "iterations_csv_exists": (run_dir / "iterations.csv").exists(),
        "run_log_exists": (run_dir / "run.log").exists(),
        "profiler_log_exists": any(
            (run_dir / name).exists() for name in ("nsys.log", "hpcrun.log")
        ),
        "nsys_report_exists": report.exists(),
        "nsys_report_bytes": report.stat().st_size if report.exists() else None,
        "hpctoolkit_measurement_exists": bool(measurement_dirs),
        "hpctoolkit_database_exists": bool(database_dirs),
    }


def config_matches(summary: dict, metadata: dict, expected: dict | None) -> bool:
    if expected is None:
        return False
    workload = metadata.get("workload", {}) if isinstance(metadata, dict) else {}
    actual = {
        key: workload.get(key, summary.get(key))
        for key in ("model", "batch", "imgsz", "amp", "nbs", "max_iters", "warmup_iters")
    }
    try:
        return (
            actual["model"] == expected["model"]
            and int(actual["batch"]) == expected["batch"]
            and int(actual["imgsz"]) == expected["imgsz"]
            and parse_bool(actual["amp"]) == expected["amp"]
            and int(actual["nbs"]) == expected["nbs"]
            and int(actual["max_iters"]) == expected["max_iters"]
            and int(actual["warmup_iters"]) == expected["warmup_iters"]
        )
    except (TypeError, ValueError):
        return False


def scan(runs_roots: list[Path], experiments_path: Path) -> list[dict[str, Any]]:
    experiments = load_experiments(experiments_path)
    rows: list[dict[str, Any]] = []
    for location in iter_run_directories(runs_roots):
        run_dir = location["run_directory"]
        summary, summary_error = load_json(run_dir / "summary.json")
        metadata, metadata_error = load_json(run_dir / "metadata.json")
        profile_config, profile_error = load_json(run_dir / "profiling_config.json")
        summary = summary or {}
        metadata = metadata or {}
        profile_config = profile_config or {}
        workload = metadata.get("workload", {})
        actual_iterations, iteration_error = count_csv_rows(run_dir / "iterations.csv")
        errors = [
            value
            for value in (summary_error, metadata_error, profile_error, iteration_error)
            if value
        ]
        files = run_files(run_dir)
        row = {
            "collection_root": relative_path(location["runs_root"].parent),
            "device_id": location["device_id"],
            "workload_id": location["workload_id"],
            "run_type": location["run_type"],
            "repeat": location["repeat"],
            "run_directory": relative_path(run_dir),
            **files,
            "summary_json_valid": bool(summary) and summary_error is None,
            "metadata_json_valid": bool(metadata) and metadata_error is None,
            "model": workload.get("model", summary.get("model")),
            "batch": workload.get("batch", summary.get("batch")),
            "imgsz": workload.get("imgsz", summary.get("imgsz")),
            "amp": workload.get("amp", summary.get("amp")),
            "nbs": workload.get("nbs", summary.get("nbs")),
            "max_iters": workload.get("max_iters", summary.get("max_iters")),
            "warmup_iters": workload.get("warmup_iters", summary.get("warmup_iters")),
            "device": metadata.get("device_argument"),
            "gpu_name": metadata.get("gpu_name"),
            "cuda_version": metadata.get("torch_cuda_version"),
            "pytorch_version": metadata.get("torch_version"),
            "ultralytics_version": metadata.get("ultralytics_version"),
            "nsight_version": profile_config.get("nsys_version"),
            "dataset_hash": first_nested_value(
                metadata, ("dataset_sha256", "dataset_hash", "semantic_hash")
            ),
            "metadata_run_type": metadata.get("run_type", summary.get("run_type")),
            "requested_iterations": workload.get("max_iters", summary.get("max_iters")),
            "actual_iterations": actual_iterations,
            "valid_iterations": summary.get("valid_iters"),
            "steady_window_mean_iter_ms": summary.get("steady_window_mean_iter_ms"),
            "config_matches_experiment": config_matches(
                summary, metadata, experiments.get(location["workload_id"])
            ),
            "parse_errors": " | ".join(errors),
        }
        rows.append(row)
    return rows


def build_report(rows: list[dict[str, Any]], experiments_path: Path) -> str:
    experiments = load_experiments(experiments_path)
    expected_workloads = sorted(experiments)
    devices = sorted({row["device_id"] for row in rows})
    counts = Counter((row["device_id"], row["run_type"]) for row in rows)
    logical = defaultdict(list)
    for row in rows:
        logical[(row["device_id"], row["workload_id"], row["run_type"])].append(row)

    lines = [
        "# Collection validation report",
        "",
        "## Data inventory",
        "",
        f"- Discovered devices: {', '.join(devices) if devices else 'none'}",
        f"- Discovered run directories: {len(rows)}",
        f"- Expected workloads from config: {len(expected_workloads)}",
        "",
        "| Device | Workloads | Baseline | Nsight trace | GPU metrics | CPU metrics | HPCToolkit CUDA |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for device in devices:
        workloads = {row["workload_id"] for row in rows if row["device_id"] == device}
        lines.append(
            f"| {device} | {len(workloads)} | {counts[(device, 'baseline')]} | "
            f"{counts[(device, 'nsys_trace')]} | {counts[(device, 'nsys_gpu_metrics')]} | "
            f"{counts[(device, 'nsys_cpu_metrics')]} | {counts[(device, 'hpctoolkit_cuda')]} |"
        )

    lines += ["", "## Completeness", ""]
    missing: list[str] = []
    for device in devices:
        for workload in expected_workloads:
            for run_type, expected_count in REQUIRED_MAIN_TYPES.items():
                actual = len(logical[(device, workload, run_type)])
                if actual != expected_count:
                    missing.append(
                        f"- {device}/{workload}: {run_type} expected {expected_count}, found {actual}"
                    )
    lines.extend(missing or ["All required baseline and Nsight run counts are complete."])

    malformed = [row for row in rows if row["parse_errors"]]
    config_bad = [row for row in rows if not row["config_matches_experiment"]]
    dataset_hashes = sorted({str(row["dataset_hash"]) for row in rows if row["dataset_hash"]})
    nsys_missing = [
        row for row in rows if row["run_type"].startswith("nsys_") and not row["nsys_report_exists"]
    ]
    lines += [
        "",
        "## File and configuration validation",
        "",
        f"- Malformed/unreadable runs: {len(malformed)}",
        f"- Runs inconsistent with experiments.csv: {len(config_bad)}",
        f"- Nsight runs missing profile.nsys-rep: {len(nsys_missing)}",
        f"- Distinct recorded dataset hashes: {len(dataset_hashes)}",
    ]
    if dataset_hashes:
        lines.extend(f"  - `{value}`" for value in dataset_hashes)
    else:
        lines.append(
            "- Warning: no run metadata records a dataset semantic hash; dataset identity cannot be cryptographically verified."
        )
    lines += [
        "",
        "HPCToolkit is optional and is not included in the main-data validity gate.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan all downloaded experiment runs.")
    parser.add_argument("--runs-root", action="append", type=Path)
    parser.add_argument("--experiments", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    runs_roots = [path.resolve() for path in args.runs_root] if args.runs_root else discover_runs_roots()
    if not runs_roots:
        raise SystemExit("No runs directories found.")
    rows = scan(runs_roots, args.experiments.resolve())
    output_dir = args.output_dir.resolve()
    write_csv(output_dir / "collection_inventory.csv", rows, INVENTORY_FIELDS)
    (output_dir / "collection_validation_report.md").write_text(
        build_report(rows, args.experiments.resolve()), encoding="utf-8"
    )
    print(f"Scanned {len(rows)} runs from {len(runs_roots)} collection root(s).")


if __name__ == "__main__":
    main()
