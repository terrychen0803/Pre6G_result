from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

from process_pre6g_gemma_part_ab import (
    BASE_BIN_MS,
    CHANNEL_SLUGS,
    DEFAULT_NSYS,
    HORIZONS_SECONDS,
    baseline_summary,
    export_sqlite,
    extract_timeseries,
    load_experiments,
    sha256,
    validate_sqlite,
    write_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("PRE6G_DATA_ROOT", PROJECT_ROOT / "data" / "raw")).expanduser()
FAMILIES = {
    "dnn": {
        "source_root": DATA_ROOT
        / "Pre6G_5090_dnn_gemma_vit"
        / "dnn_runtime_prediction",
        "target_root": DATA_ROOT
        / "Pre6G_4090_dnn_gemma_vit"
        / "Pre6G"
        / "dnn_runtime_prediction",
        "workload_family": "dense_mlp",
        "prefix": "D",
    },
    "vit": {
        "source_root": DATA_ROOT
        / "Pre6G_5090_dnn_gemma_vit"
        / "vit_runtime_prediction",
        "target_root": DATA_ROOT
        / "Pre6G_4090_dnn_gemma_vit"
        / "Pre6G"
        / "vit_runtime_prediction",
        "workload_family": "vision_transformer",
        "prefix": "V",
    },
}


def save_npz(
    path: Path,
    time_ms: np.ndarray,
    values: np.ndarray,
    workload: str,
    workload_family: str,
    horizon: int | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        time_ms=time_ms,
        values=values,
        channels=np.asarray(CHANNEL_SLUGS),
        base_bin_ms=np.asarray(BASE_BIN_MS),
        device_id=np.asarray("RTX5090"),
        workload_id=np.asarray(workload),
        workload_family=np.asarray(workload_family),
        horizon_seconds=np.asarray(horizon if horizon is not None else -1),
        marker_free=np.asarray(True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare one untouched Pre6G external workload family."
    )
    parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
    parser.add_argument("--nsys", type=Path, default=DEFAULT_NSYS)
    parser.add_argument("--force-export", action="store_true")
    args = parser.parse_args()
    spec = FAMILIES[args.family]
    source_root = Path(spec["source_root"]).resolve()
    target_root = Path(spec["target_root"]).resolve()
    cache_root = (
        PROJECT_ROOT
        / "analysis_cache"
        / f"pre6g_{args.family}_5090_to_4090"
    ).resolve()
    output_root = (
        PROJECT_ROOT
        / "analysis_outputs"
        / f"pre6g_{args.family}_5090_to_4090"
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    experiments = load_experiments(source_root / "configs" / "experiments.csv")

    manifest_rows: list[dict] = []
    sqlite_rows: list[dict] = []
    timeseries_rows: list[dict] = []
    for index, experiment in enumerate(experiments, start=1):
        workload = experiment["workload_id"]
        if not workload.startswith(str(spec["prefix"])):
            raise ValueError(f"Unexpected workload id {workload}")
        print(f"[{index:02d}/{len(experiments):02d}] {workload}")
        run_dir = (
            source_root
            / "runs"
            / "RTX5090"
            / workload
            / "nsys_deployment_01"
        )
        raw_report = run_dir / "profile.nsys-rep"
        raw_sqlite = run_dir / "profile.sqlite"
        cached_sqlite = cache_root / "sqlite" / f"{workload}.sqlite"
        database = raw_sqlite if raw_sqlite.exists() and not args.force_export else cached_sqlite
        if not raw_report.exists() or raw_report.stat().st_size == 0:
            raise FileNotFoundError(raw_report)
        if database == cached_sqlite:
            export_sqlite(args.nsys.expanduser(), raw_report, database, args.force_export)

        source_baseline = baseline_summary(source_root, "RTX5090", workload)
        target_baseline = baseline_summary(target_root, "RTX4090", workload)
        runtime_eligible = (
            source_baseline["count"] == 5 and target_baseline["count"] == 5
        )
        metadata_path = run_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        row = {
            "workload_id": workload,
            "workload_family": spec["workload_family"],
            "source_device": "RTX5090",
            "target_device": "RTX4090",
            "profile_report_path": str(raw_report),
            "profile_report_bytes": raw_report.stat().st_size,
            "profile_report_sha256": sha256(raw_report),
            "profile_sqlite_path": str(database),
            "uses_workload_markers": False,
            "instrumentation_policy": metadata.get("instrumentation_policy"),
            "execution_total_iters": metadata.get("execution_total_iters"),
            "source_baseline_count": source_baseline["count"],
            "source_baseline_median_ms": source_baseline["median_ms"],
            "source_baseline_std_ms": source_baseline["std_ms"],
            "target_baseline_count": target_baseline["count"],
            "target_baseline_median_ms": target_baseline["median_ms"],
            "target_baseline_std_ms": target_baseline["std_ms"],
            "runtime_regression_eligible": runtime_eligible,
        }
        row.update({f"audit_config_{key}": value for key, value in experiment.items()})
        manifest_rows.append(row)

        sqlite_diagnostics, descriptors = validate_sqlite(database)
        sqlite_diagnostics["workload_id"] = workload
        sqlite_rows.append(sqlite_diagnostics)
        if not sqlite_diagnostics["schema_valid"]:
            raise ValueError(f"Invalid SQLite schema for {workload}")
        time_ms, values, diagnostics = extract_timeseries(database, descriptors)
        full_path = cache_root / "timeseries_1ms" / "full" / f"{workload}.npz"
        save_npz(
            full_path,
            time_ms,
            values,
            workload,
            str(spec["workload_family"]),
            None,
        )
        diagnostics.update(
            {
                "workload_id": workload,
                "full_timeseries_path": str(full_path),
            }
        )
        all_prefixes_valid = True
        for horizon in HORIZONS_SECONDS:
            count = int(horizon * 1000 / BASE_BIN_MS) + 1
            valid = values.shape[1] >= count
            prefix_path = (
                cache_root
                / "timeseries_1ms"
                / f"prefix_{horizon}s"
                / f"{workload}.npz"
            )
            diagnostics[f"prefix_{horizon}s_valid"] = valid
            diagnostics[f"prefix_{horizon}s_samples"] = count if valid else 0
            diagnostics[f"prefix_{horizon}s_path"] = str(prefix_path) if valid else ""
            all_prefixes_valid = all_prefixes_valid and valid
            if valid:
                save_npz(
                    prefix_path,
                    time_ms[:count],
                    values[:, :count],
                    workload,
                    str(spec["workload_family"]),
                    horizon,
                )
        diagnostics["timeseries_valid"] = (
            diagnostics["nan_count"] == 0
            and diagnostics["infinite_count"] == 0
            and not diagnostics["constant_channels"]
            and all_prefixes_valid
        )
        timeseries_rows.append(diagnostics)

    write_csv(output_root / "manifest.csv", manifest_rows)
    write_csv(output_root / "sqlite_validation.csv", sqlite_rows)
    write_csv(output_root / "timeseries_validation.csv", timeseries_rows)
    valid_sqlite = sum(bool(row["schema_valid"]) for row in sqlite_rows)
    valid_timeseries = sum(bool(row["timeseries_valid"]) for row in timeseries_rows)
    eligible = sum(bool(row["runtime_regression_eligible"]) for row in manifest_rows)
    minimum_capture = min(float(row["capture_seconds"]) for row in timeseries_rows)
    maximum_capture = max(float(row["capture_seconds"]) for row in timeseries_rows)
    report_lines = [
        f"# Pre6G {args.family.upper()} external-family preparation",
        "",
        f"- Workloads: {len(manifest_rows)}",
        f"- Valid SQLite: {valid_sqlite}",
        f"- Valid 1 ms time series with all prefixes: {valid_timeseries}",
        f"- Runtime-regression-eligible pairs: {eligible}",
        f"- Capture range: {minimum_capture:.3f} to {maximum_capture:.3f} seconds",
        f"- NVTX rows: {sum(int(row['nvtx_rows']) for row in sqlite_rows)}",
        f"- Workload family: {spec['workload_family']}",
        "- Baselines and workload metadata are audit-only and are not stored in NPZ detector inputs.",
    ]
    (output_root / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "family": args.family,
                "workloads": len(manifest_rows),
                "valid_sqlite": valid_sqlite,
                "valid_timeseries": valid_timeseries,
                "runtime_eligible": eligible,
                "output_root": str(output_root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
