from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QC = PROJECT_ROOT / "analysis" / "high_load_01" / "qc" / "run_qc.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "analysis" / "high_load_01" / "overhead"
MATCH_FEATURES = [
    "pre_gpu_util_mean",
    "pre_gpu_util_std",
    "pre_gpu_util_min",
    "pre_gpu_util_max",
    "pre_gpu_mem_used_mean",
    "pre_gpu_power_mean",
]


def robust_scales(data: pd.DataFrame, features: list[str]) -> dict[str, float]:
    result = {}
    for feature in features:
        values = pd.to_numeric(data[feature], errors="coerce").dropna()
        scale = float(values.quantile(0.75) - values.quantile(0.25))
        if not np.isfinite(scale) or scale <= 0:
            scale = float(values.std(ddof=1))
        result[feature] = scale if np.isfinite(scale) and scale > 0 else 1.0
    return result


def closest_baseline(
    profile: pd.Series, baselines: pd.DataFrame, scales: dict[str, float]
) -> tuple[pd.Series, float]:
    distances = np.zeros(len(baselines), dtype=float)
    for feature, scale in scales.items():
        profile_value = float(profile[feature])
        values = pd.to_numeric(baselines[feature], errors="coerce").to_numpy(dtype=float)
        distances += np.square((values - profile_value) / scale)
    distances = np.sqrt(distances / len(scales))
    position = int(np.nanargmin(distances))
    return baselines.iloc[position], float(distances[position])


def overhead(profile_ms: float, reference_ms: float) -> tuple[float, float, float]:
    absolute = profile_ms - reference_ms
    ratio = absolute / reference_ms
    return absolute, ratio, ratio * 100


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze clean/high GPU-metrics profiling overhead.")
    parser.add_argument("--run-qc", type=Path, default=DEFAULT_QC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(args.run_qc)
    high = runs[runs.condition == "high_load_01"]
    scales = robust_scales(high, MATCH_FEATURES)
    records = []
    for (condition, workload), group in runs.groupby(["condition", "workload_id"], sort=True):
        baselines = group[group.run_type == "baseline"].copy()
        profiles = group[group.run_type == "nsys_gpu_metrics"]
        if len(profiles) != 1:
            raise ValueError(f"Expected one GPU-metrics profile for {condition}/{workload}")
        profile = profiles.iloc[0]
        baseline_values = baselines.steady_window_total_ms.astype(float)
        profile_ms = float(profile.steady_window_total_ms)
        baseline_mean = float(baseline_values.mean())
        baseline_median = float(baseline_values.median())
        mean_abs, mean_ratio, mean_pct = overhead(profile_ms, baseline_mean)
        median_abs, median_ratio, median_pct = overhead(profile_ms, baseline_median)
        row = {
            "condition": condition,
            "device_id": "RTX5090",
            "workload_id": workload,
            "profile_runtime_ms": profile_ms,
            "baseline_n": len(baselines),
            "baseline_mean_ms": baseline_mean,
            "baseline_median_ms": baseline_median,
            "baseline_std_ms": float(baseline_values.std(ddof=1)),
            "baseline_cv_percent": float(baseline_values.std(ddof=1) / baseline_mean * 100),
            "baseline_min_ms": float(baseline_values.min()),
            "baseline_max_ms": float(baseline_values.max()),
            "overhead_vs_mean_absolute_ms": mean_abs,
            "overhead_vs_mean_ratio": mean_ratio,
            "overhead_vs_mean_percent": mean_pct,
            "overhead_vs_median_absolute_ms": median_abs,
            "overhead_vs_median_ratio": median_ratio,
            "overhead_vs_median_percent": median_pct,
            "profile_within_baseline_range": bool(
                baseline_values.min() <= profile_ms <= baseline_values.max()
            ),
        }
        if condition == "high_load_01":
            matched, distance = closest_baseline(profile, baselines, scales)
            matched_ms = float(matched.steady_window_total_ms)
            matched_abs, matched_ratio, matched_pct = overhead(profile_ms, matched_ms)
            row.update(
                matched_baseline_repeat=int(matched["repeat"]),
                matched_load_distance=distance,
                matched_baseline_runtime_ms=matched_ms,
                overhead_vs_matched_absolute_ms=matched_abs,
                overhead_vs_matched_ratio=matched_ratio,
                overhead_vs_matched_percent=matched_pct,
            )
            for feature in MATCH_FEATURES:
                row[f"profile_{feature}"] = float(profile[feature])
                row[f"matched_{feature}"] = float(matched[feature])
                row[f"profile_minus_matched_{feature}"] = float(profile[feature]) - float(matched[feature])
        records.append(row)
    data = pd.DataFrame(records).sort_values(["condition", "workload_id"])
    data.to_csv(output / "gpu_metrics_overhead_by_workload.csv", index=False)

    summaries = []
    for condition, group in data.groupby("condition", sort=True):
        for reference in ("mean", "median"):
            values = group[f"overhead_vs_{reference}_percent"]
            summaries.append(
                {
                    "condition": condition,
                    "reference": f"baseline_{reference}",
                    "n_workloads": len(group),
                    "mean_overhead_percent": float(values.mean()),
                    "median_overhead_percent": float(values.median()),
                    "std_overhead_percent": float(values.std(ddof=1)),
                    "p10_overhead_percent": float(values.quantile(0.1)),
                    "p90_overhead_percent": float(values.quantile(0.9)),
                    "min_overhead_percent": float(values.min()),
                    "max_overhead_percent": float(values.max()),
                    "negative_overhead_count": int((values < 0).sum()),
                    "profile_within_baseline_range_count": int(group.profile_within_baseline_range.sum()),
                }
            )
        if condition == "high_load_01":
            values = group.overhead_vs_matched_percent
            summaries.append(
                {
                    "condition": condition,
                    "reference": "load_matched_baseline",
                    "n_workloads": len(group),
                    "mean_overhead_percent": float(values.mean()),
                    "median_overhead_percent": float(values.median()),
                    "std_overhead_percent": float(values.std(ddof=1)),
                    "p10_overhead_percent": float(values.quantile(0.1)),
                    "p90_overhead_percent": float(values.quantile(0.9)),
                    "min_overhead_percent": float(values.min()),
                    "max_overhead_percent": float(values.max()),
                    "negative_overhead_count": int((values < 0).sum()),
                    "profile_within_baseline_range_count": int(group.profile_within_baseline_range.sum()),
                }
            )
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "gpu_metrics_overhead_summary.csv", index=False)

    clean = data[data.condition == "clean"].set_index("workload_id")
    loaded = data[data.condition == "high_load_01"].set_index("workload_id")
    paired = pd.DataFrame(index=sorted(set(clean.index) & set(loaded.index)))
    paired.index.name = "workload_id"
    for column in (
        "profile_runtime_ms",
        "baseline_median_ms",
        "baseline_cv_percent",
        "overhead_vs_median_percent",
    ):
        paired[f"clean_{column}"] = clean.loc[paired.index, column]
        paired[f"high_{column}"] = loaded.loc[paired.index, column]
        paired[f"high_minus_clean_{column}"] = paired[f"high_{column}"] - paired[f"clean_{column}"]
    paired.reset_index().to_csv(output / "clean_high_overhead_paired.csv", index=False)

    clean_summary = summary[
        (summary.condition == "clean") & (summary.reference == "baseline_median")
    ].iloc[0]
    high_summary = summary[
        (summary.condition == "high_load_01") & (summary.reference == "baseline_median")
    ].iloc[0]
    matched_summary = summary[
        (summary.condition == "high_load_01") & (summary.reference == "load_matched_baseline")
    ].iloc[0]
    lines = [
        "# GPU-metrics profiling overhead: clean versus high load",
        "",
        "Only `nsys_gpu_metrics_01` is analyzed. Trace profiles are excluded.",
        "",
        "## Baseline-median comparison",
        "",
        f"- Clean median overhead across workloads: {clean_summary.median_overhead_percent:.2f}% (mean {clean_summary.mean_overhead_percent:.2f}%)",
        f"- High-load median overhead across workloads: {high_summary.median_overhead_percent:.2f}% (mean {high_summary.mean_overhead_percent:.2f}%)",
        f"- Clean negative observed overheads: {int(clean_summary.negative_overhead_count)}/24",
        f"- High-load negative observed overheads: {int(high_summary.negative_overhead_count)}/24",
        f"- High-load profiled runtimes inside the five-baseline range: {int(high_summary.profile_within_baseline_range_count)}/24",
        "",
        "## High-load pre-run GPU-state matched comparison",
        "",
        f"- Matched-baseline median overhead: {matched_summary.median_overhead_percent:.2f}% (mean {matched_summary.mean_overhead_percent:.2f}%)",
        f"- P10/P90: {matched_summary.p10_overhead_percent:.2f}% / {matched_summary.p90_overhead_percent:.2f}%",
        f"- Negative observed overheads: {int(matched_summary.negative_overhead_count)}/24",
        "",
        "A negative observed value does not imply that profiling accelerates training. It means run-to-run contention variation is larger than the profiler effect for that workload/reference run.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "experiment_contract.json").write_text(
        json.dumps(
            {
                "profile_mode": "nsys_gpu_metrics_01",
                "trace_profiles_used": False,
                "runtime_target": "steady_window_total_ms",
                "baseline_repeats": {"clean": 3, "high_load_01": 5},
                "load_match_features": MATCH_FEATURES,
                "load_match_scaling": "global high-load IQR, fallback to standard deviation",
                "load_match_distance": "root mean squared standardized distance",
                "raw_data_modified": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote GPU-metrics overhead analysis to {output}")


if __name__ == "__main__":
    main()
