from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("PRE6G_DATA_ROOT", PROJECT_ROOT / "data" / "raw")).expanduser()
CLEAN_ROOT = DATA_ROOT / "RTX5090_yolo26_collection_20260817" / "runs" / "RTX5090"
CLEAN_CONFIG = DATA_ROOT / "RTX5090_yolo26_collection_20260817" / "configs" / "experiments.csv"
HIGH_ROOT = (
    DATA_ROOT
    / "high_load_01_RTX5090_20260923"
    / "runs_loaded"
    / "high_load_01"
    / "RTX5090"
)
HIGH_CONFIG = DATA_ROOT / "high_load_01_RTX5090_20260923" / "configs" / "experiments.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "analysis" / "high_load_01" / "qc"

WORKLOAD_FIELDS = ("model", "batch", "imgsz", "amp", "max_iters", "warmup_iters", "nbs")
SOFTWARE_FIELDS = (
    "python_version",
    "torch_version",
    "torch_cuda_version",
    "ultralytics_version",
    "gpu_name",
)
NETDATA_FIELDS = {
    "gpu_util": "GPU Util%",
    "gpu_mem_used": "GPU Mem Used(MB)",
    "gpu_power": "GPU Power(W)",
    "cpu_user": "CPU User%",
    "cpu_system": "CPU System%",
    "cpu_iowait": "CPU IOWait%",
    "load1": "Load 1min",
    "mem_used": "Mem Used(MB)",
}
FAILURE_PATTERN = re.compile(
    r"cuda out of memory|outofmemoryerror|traceback \(most recent call last\)|"
    r"segmentation fault|collection failed|application returned non-zero|fatal error",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return (Path("data/raw") / resolved.relative_to(DATA_ROOT.resolve())).as_posix()
    except ValueError:
        try:
            return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
        except ValueError:
            return resolved.as_posix()


def normalize_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_experiments(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["amp"] = data["amp"].map(normalize_bool)
    return data.sort_values("workload_id").reset_index(drop=True)


def workload_from_metadata(metadata: dict) -> dict:
    source = metadata.get("workload", {})
    result = {key: source.get(key) for key in WORKLOAD_FIELDS}
    result["amp"] = normalize_bool(result["amp"])
    for key in ("batch", "imgsz", "max_iters", "warmup_iters", "nbs"):
        result[key] = int(result[key])
    return result


def workload_matches(metadata: dict, expected: pd.Series) -> bool:
    actual = workload_from_metadata(metadata)
    return all(actual[key] == expected[key] for key in WORKLOAD_FIELDS)


def log_failure(run_dir: Path) -> str:
    matches: list[str] = []
    for name in ("run.log", "nsys.log", "netdata.log"):
        path = run_dir / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        found = FAILURE_PATTERN.search(text)
        if found:
            matches.append(f"{name}:{found.group(0)}")
    return " | ".join(matches)


def numeric_stats(frame: pd.DataFrame, column: str, prefix: str) -> dict:
    if column not in frame:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return {
        f"{prefix}_mean": float(values.mean()) if len(values) else np.nan,
        f"{prefix}_std": float(values.std(ddof=1)) if len(values) >= 2 else np.nan,
        f"{prefix}_min": float(values.min()) if len(values) else np.nan,
        f"{prefix}_max": float(values.max()) if len(values) else np.nan,
    }


def inspect_iterations(run_dir: Path) -> tuple[dict, pd.DataFrame]:
    path = run_dir / "iterations.csv"
    data = pd.read_csv(path)
    warmup = data["warmup"].map(normalize_bool)
    steady = data[~warmup]
    result = {
        "iteration_count": len(data),
        "warmup_count": int(warmup.sum()),
        "steady_count": len(steady),
        "has_unix_timestamps": {"start_unix_ns", "end_unix_ns"} <= set(data.columns),
        "iteration_ids_contiguous": data["iteration_id"].tolist() == list(range(1, len(data) + 1)),
    }
    if result["has_unix_timestamps"]:
        starts = pd.to_numeric(data["start_unix_ns"], errors="coerce")
        ends = pd.to_numeric(data["end_unix_ns"], errors="coerce")
        result.update(
            timestamps_valid=bool(starts.notna().all() and ends.notna().all() and (ends > starts).all()),
            first_iteration_start_ns=int(starts.min()),
            last_iteration_end_ns=int(ends.max()),
        )
    else:
        result.update(
            timestamps_valid=False,
            first_iteration_start_ns=np.nan,
            last_iteration_end_ns=np.nan,
        )
    return result, data


def inspect_netdata(run_dir: Path, iterations: pd.DataFrame, pre_window_s: float) -> dict:
    path = run_dir / "netdata.csv"
    if not path.exists() or not {"start_unix_ns", "end_unix_ns"} <= set(iterations.columns):
        return {"netdata_exists": path.exists(), "netdata_timestamp_alignment": False}
    data = pd.read_csv(path)
    timestamps = pd.to_numeric(data["TimestampUnixNs"], errors="coerce")
    valid = timestamps.notna()
    data = data.loc[valid].copy()
    data["_timestamp_ns"] = timestamps.loc[valid].astype("int64")
    data = data.sort_values("_timestamp_ns")
    first_start = int(pd.to_numeric(iterations["start_unix_ns"]).min())
    last_end = int(pd.to_numeric(iterations["end_unix_ns"]).max())
    pre_start = first_start - int(pre_window_s * 1e9)
    pre = data[(data._timestamp_ns >= pre_start) & (data._timestamp_ns < first_start)]
    runtime = data[(data._timestamp_ns >= first_start) & (data._timestamp_ns <= last_end)]
    gaps = data._timestamp_ns.diff().dropna() / 1e9
    result = {
        "netdata_exists": True,
        "netdata_rows": len(data),
        "netdata_max_gap_s": float(gaps.max()) if len(gaps) else np.nan,
        "netdata_gap_over_2s": bool(len(gaps) and (gaps > 2.0).any()),
        "pre_samples": len(pre),
        "runtime_samples": len(runtime),
        "netdata_timestamp_alignment": bool(len(pre) and len(runtime)),
        "pre_window_coverage_s": (
            float((pre._timestamp_ns.max() - pre._timestamp_ns.min()) / 1e9)
            if len(pre) >= 2
            else 0.0
        ),
    }
    for name, column in NETDATA_FIELDS.items():
        result.update(numeric_stats(pre, column, f"pre_{name}"))
    pre_gpu = result.get("pre_gpu_util_mean", np.nan)
    result["background_load_idle_warning"] = bool(math.isfinite(pre_gpu) and pre_gpu < 50.0)
    return result


def inspect_run(
    condition: str,
    workload_id: str,
    run_type: str,
    repeat: int,
    run_dir: Path,
    expected: pd.Series,
    pre_window_s: float,
) -> dict:
    required = ["summary.json", "metadata.json", "iterations.csv"]
    if condition == "high_load_01":
        required.append("netdata.csv")
    if run_type == "nsys_gpu_metrics":
        required.extend(["profiling_config.json", "profile.nsys-rep", "nsys.log"])
    missing = [name for name in required if not (run_dir / name).exists()]
    row = {
        "condition": condition,
        "device_id": "RTX5090",
        "workload_id": workload_id,
        "run_type": run_type,
        "repeat": repeat,
        "run_directory": portable_path(run_dir),
        "missing_files": " | ".join(missing),
    }
    if missing and any(name in missing for name in ("summary.json", "metadata.json", "iterations.csv")):
        return row
    summary = load_json(run_dir / "summary.json")
    metadata = load_json(run_dir / "metadata.json")
    iteration_qc, iterations = inspect_iterations(run_dir)
    row.update(iteration_qc)
    row.update(
        config_matches_experiment=workload_matches(metadata, expected),
        steady_window_total_ms=float(summary["steady_window_total_ms"]),
        steady_window_mean_iter_ms=float(summary["steady_window_mean_iter_ms"]),
        host_callback_mean_ms=float(summary["host_callback_mean_ms"]),
        gpu_mean_ms=float(summary["gpu_mean_ms"]),
        software_signature="|".join(str(metadata.get(key)) for key in SOFTWARE_FIELDS),
        log_failure=log_failure(run_dir),
    )
    row.update(inspect_netdata(run_dir, iterations, pre_window_s))
    if run_type == "nsys_gpu_metrics" and (run_dir / "profiling_config.json").exists():
        profile = load_json(run_dir / "profiling_config.json")
        row.update(
            nsys_version=profile.get("nsys_version"),
            gpu_metrics_frequency_hz=profile.get("gpu_metrics_frequency_hz"),
            gpu_metrics_set=profile.get("gpu_metrics_set"),
            trace="|".join(profile.get("trace") or []),
            sample=profile.get("sample"),
            cpuctxsw=profile.get("cpuctxsw"),
            profile_report_bytes=(run_dir / "profile.nsys-rep").stat().st_size
            if (run_dir / "profile.nsys-rep").exists()
            else np.nan,
        )
    return row


def baseline_variability(runs: pd.DataFrame) -> pd.DataFrame:
    baseline = runs[runs.run_type == "baseline"].copy()
    records = []
    for (condition, workload), group in baseline.groupby(["condition", "workload_id"], sort=True):
        values = group.steady_window_total_ms.astype(float)
        mean = float(values.mean())
        records.append(
            {
                "condition": condition,
                "device_id": "RTX5090",
                "workload_id": workload,
                "n": len(values),
                "mean_ms": mean,
                "median_ms": float(values.median()),
                "std_ms": float(values.std(ddof=1)),
                "cv_percent": float(values.std(ddof=1) / mean * 100),
                "min_ms": float(values.min()),
                "max_ms": float(values.max()),
            }
        )
    return pd.DataFrame(records)


def build_alignment(
    clean_experiments: pd.DataFrame,
    high_experiments: pd.DataFrame,
    runs: pd.DataFrame,
) -> pd.DataFrame:
    clean = clean_experiments.set_index("workload_id")
    high = high_experiments.set_index("workload_id")
    rows = []
    for workload in sorted(set(clean.index) | set(high.index)):
        row = {"workload_id": workload}
        for field in WORKLOAD_FIELDS:
            row[f"clean_{field}"] = clean.at[workload, field] if workload in clean.index else np.nan
            row[f"high_{field}"] = high.at[workload, field] if workload in high.index else np.nan
        row["config_fields_match"] = bool(
            workload in clean.index
            and workload in high.index
            and all(clean.at[workload, field] == high.at[workload, field] for field in WORKLOAD_FIELDS)
        )
        for condition, expected_baselines in (("clean", 3), ("high_load_01", 5)):
            subset = runs[(runs.condition == condition) & (runs.workload_id == workload)]
            row[f"{condition}_baseline_count"] = int((subset.run_type == "baseline").sum())
            row[f"{condition}_gpu_profile_count"] = int((subset.run_type == "nsys_gpu_metrics").sum())
            row[f"{condition}_run_config_all_match"] = bool(subset.config_matches_experiment.fillna(False).all())
            row[f"{condition}_expected_baselines"] = expected_baselines
        rows.append(row)
    return pd.DataFrame(rows)


def report_text(
    runs: pd.DataFrame,
    alignment: pd.DataFrame,
    variability: pd.DataFrame,
    config_hash_match: bool,
) -> str:
    high = runs[runs.condition == "high_load_01"]
    profiles = high[high.run_type == "nsys_gpu_metrics"]
    low_pre = high[high.background_load_idle_warning.fillna(False)]
    bad_gaps = high[high.netdata_gap_over_2s.fillna(False)]
    lines = [
        "# High-load GPU-metrics QC and clean alignment",
        "",
        "## Scope",
        "",
        "Only baseline and `nsys_gpu_metrics_01` runs are included. Trace runs and the C01 2024 legacy profile are excluded by exact directory name.",
        "",
        "## Alignment",
        "",
        f"- Clean/high collection config SHA-256 match: {config_hash_match}",
        f"- Workloads with matching configuration fields: {int(alignment.config_fields_match.sum())}/{len(alignment)}",
        f"- Clean runs scanned: {int((runs.condition == 'clean').sum())}",
        f"- High-load runs scanned: {len(high)}",
        f"- Official high-load GPU-metrics profiles: {len(profiles)}",
        f"- Official profiles recorded with Nsight 2026.4.1: {int(profiles.nsys_version.fillna('').str.contains('2026.4.1').sum())}/{len(profiles)}",
        "",
        "Clean has three baseline repeats per workload and high-load has five. Comparisons therefore use per-workload distributions/medians, not repeat-number pairing.",
        "",
        "## High-load quality gates",
        "",
        f"- Missing required files: {int((high.missing_files.fillna('') != '').sum())}",
        f"- Incorrect iteration count: {int((high.iteration_count != 128).sum())}",
        f"- Incorrect steady iteration count: {int((high.steady_count != 108).sum())}",
        f"- Workload config mismatches: {int((~high.config_matches_experiment.fillna(False)).sum())}",
        f"- Timestamp alignment failures: {int((~high.netdata_timestamp_alignment.fillna(False)).sum())}",
        f"- Netdata gaps over 2 seconds: {len(bad_gaps)}",
        f"- Pre-run mean GPU utilization below 50%: {len(low_pre)}",
        f"- Strong failure strings in logs: {int((high.log_failure.fillna('') != '').sum())}",
        "",
        "## Known comparability caveats",
        "",
        "- Clean iteration CSVs do not contain Unix timestamps and clean runs have no paired Netdata, so current-load matching is available only among high-load runs.",
        "- Clean explicitly requested GPU metric set `gb20x`; high-load left the metric set on automatic selection. SQLite metric descriptors must match before applying the old detector.",
        "- Background contention makes five high-load baselines a runtime distribution, not five interchangeable point estimates. No run is removed solely for being fast or slow.",
        "- GPU-metrics profiling and baseline executions occurred at different times. Overhead is reported against baseline mean/median and, separately, against the closest high-load baseline in pre-run load state.",
        "",
        "## Baseline variability overview",
        "",
    ]
    for condition, group in variability.groupby("condition", sort=True):
        lines.append(
            f"- {condition}: median workload CV {group.cv_percent.median():.2f}%, "
            f"P90 {group.cv_percent.quantile(0.9):.2f}%, max {group.cv_percent.max():.2f}%"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="QC and align clean/high-load RTX5090 GPU-metrics data.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pre-window-s", type=float, default=5.0)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    clean_experiments = load_experiments(CLEAN_CONFIG)
    high_experiments = load_experiments(HIGH_CONFIG)
    run_rows: list[dict] = []
    for condition, root, experiments, baseline_count in (
        ("clean", CLEAN_ROOT, clean_experiments, 3),
        ("high_load_01", HIGH_ROOT, high_experiments, 5),
    ):
        expected = experiments.set_index("workload_id")
        for workload in expected.index:
            workload_root = root / workload
            for repeat in range(1, baseline_count + 1):
                run_rows.append(
                    inspect_run(
                        condition,
                        workload,
                        "baseline",
                        repeat,
                        workload_root / f"baseline_{repeat:02d}",
                        expected.loc[workload],
                        args.pre_window_s,
                    )
                )
            run_rows.append(
                inspect_run(
                    condition,
                    workload,
                    "nsys_gpu_metrics",
                    1,
                    workload_root / "nsys_gpu_metrics_01",
                    expected.loc[workload],
                    args.pre_window_s,
                )
            )
    runs = pd.DataFrame(run_rows).sort_values(
        ["condition", "workload_id", "run_type", "repeat"]
    )
    variability = baseline_variability(runs)
    alignment = build_alignment(clean_experiments, high_experiments, runs)
    config_hash_match = sha256(CLEAN_CONFIG) == sha256(HIGH_CONFIG)

    runs.to_csv(output / "run_qc.csv", index=False)
    variability.to_csv(output / "baseline_variability.csv", index=False)
    alignment.to_csv(output / "workload_alignment.csv", index=False)
    (output / "qc_summary.md").write_text(
        report_text(runs, alignment, variability, config_hash_match), encoding="utf-8"
    )
    (output / "qc_contract.json").write_text(
        json.dumps(
            {
                "clean_root": portable_path(CLEAN_ROOT),
                "high_root": portable_path(HIGH_ROOT),
                "included_run_types": ["baseline", "nsys_gpu_metrics_01"],
                "excluded_run_types": ["nsys_trace_01", "nsys_gpu_metrics_01_nsys2024"],
                "pre_window_seconds": args.pre_window_s,
                "raw_data_modified": False,
                "config_sha256": {
                    "clean": sha256(CLEAN_CONFIG),
                    "high_load_01": sha256(HIGH_CONFIG),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote QC outputs to {output}")


if __name__ == "__main__":
    main()
