from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import DEFAULT_CACHE_DIR, DEFAULT_OUTPUT_DIR
from detect_marker_free_cycles import BIN_FACTORS, TAIL_FRACTIONS, detect
from evaluate_marker_free_cycles import (
    ADAPTIVE_HORIZONS,
    apply_adaptive_policy,
    choose_confidence_threshold,
    ground_truth,
)
from extract_marker_free_cycle_features import extract_profile


def build_candidates(timeseries_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    configurations = [
        (factor, tail_fraction)
        for factor in BIN_FACTORS
        for tail_fraction in TAIL_FRACTIONS
    ]
    for cache_path in sorted(timeseries_dir.glob("G*.npz")):
        cached = np.load(cache_path)
        values = cached["values"].astype(float)
        base_bin_ms = float(cached["base_bin_ms"])
        workload_id = str(cached["workload_id"])
        device_id = str(cached["device_id"])
        for horizon in ADAPTIVE_HORIZONS:
            horizon_samples = int(horizon * 1000 / base_bin_ms)
            if values.shape[1] < horizon_samples:
                continue
            prefix = values[:, :horizon_samples]
            for factor, tail_fraction in configurations:
                usable = (prefix.shape[1] // factor) * factor
                rebinned = prefix[:, :usable].reshape(
                    prefix.shape[0], -1, factor
                ).mean(axis=2)
                result = detect(
                    rebinned,
                    bin_ms=base_bin_ms * factor,
                    tail_fraction=tail_fraction,
                )
                rows.append(
                    {
                        "device_id": device_id,
                        "workload_id": workload_id,
                        "horizon_seconds": horizon,
                        "configuration": json.dumps(
                            {
                                "bin_ms": base_bin_ms * factor,
                                "tail_fraction": tail_fraction,
                            },
                            sort_keys=True,
                        ),
                        "bin_ms": base_bin_ms * factor,
                        "tail_fraction": tail_fraction,
                        "segment_start_seconds": horizon * (1 - tail_fraction),
                        **result,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["horizon_seconds", "device_id", "workload_id", "configuration"]
    )


def train_frozen_yolo_policy(
    yolo_candidates_path: Path, yolo_cache_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, float, pd.DataFrame, pd.DataFrame]:
    candidates = pd.read_csv(yolo_candidates_path)
    candidates = candidates[candidates.horizon_seconds.isin(ADAPTIVE_HORIZONS)].copy()
    oracle = ground_truth(yolo_cache_dir)
    training = candidates.merge(
        oracle, on=["device_id", "workload_id"], validate="many_to_one"
    )
    training["period_ape_percent"] = (
        (training.detected_period_ms - training.oracle_period_ms).abs()
        / training.oracle_period_ms
        * 100
    )
    score_rows: list[dict] = []
    selected_rows: list[pd.DataFrame] = []
    policy_rows: list[dict] = []
    for horizon in ADAPTIVE_HORIZONS:
        horizon_data = training[training.horizon_seconds == horizon]
        scores = (
            horizon_data.groupby("configuration")
            .period_ape_percent.agg(["mean", "median", "max", "count"])
            .reset_index()
            .sort_values(["mean", "median", "max", "configuration"])
        )
        scores.insert(0, "horizon_seconds", horizon)
        scores["selected"] = False
        scores.loc[scores.index[0], "selected"] = True
        score_rows.extend(scores.to_dict("records"))
        winner = str(scores.iloc[0].configuration)
        chosen = horizon_data[horizon_data.configuration == winner].copy()
        selected_rows.append(chosen)
        policy_rows.append(
            {
                "horizon_seconds": horizon,
                "configuration": winner,
                "training_profiles": len(chosen),
                "training_mean_period_ape_percent": float(
                    chosen.period_ape_percent.mean()
                ),
                "training_median_period_ape_percent": float(
                    chosen.period_ape_percent.median()
                ),
                "training_max_period_ape_percent": float(
                    chosen.period_ape_percent.max()
                ),
            }
        )
    selected = pd.concat(selected_rows, ignore_index=True)
    threshold, threshold_scores = choose_confidence_threshold(selected)
    frozen_training_deployment = apply_adaptive_policy(selected, threshold)
    return (
        pd.DataFrame(policy_rows),
        pd.DataFrame(score_rows),
        threshold,
        threshold_scores,
        frozen_training_deployment,
    )


def apply_frozen_policy(
    candidates: pd.DataFrame, policy: pd.DataFrame, threshold: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows: list[pd.DataFrame] = []
    for item in policy.itertuples(index=False):
        selected_rows.append(
            candidates[
                (candidates.horizon_seconds == item.horizon_seconds)
                & (candidates.configuration == item.configuration)
            ].copy()
        )
    selected = pd.concat(selected_rows, ignore_index=True).sort_values(
        ["device_id", "workload_id", "horizon_seconds"]
    )
    deployment = apply_adaptive_policy(selected, threshold).sort_values(
        ["device_id", "workload_id"]
    )
    deployment["selected_confidence_threshold"] = threshold
    return selected, deployment


def extract_features(
    deployment: pd.DataFrame, timeseries_dir: Path
) -> pd.DataFrame:
    rows: list[dict] = []
    for detection in deployment[deployment.accepted].itertuples(index=False):
        cache_path = timeseries_dir / f"{detection.workload_id}.npz"
        rows.append(extract_profile(pd.Series(detection._asdict()), cache_path))
    return pd.DataFrame(rows).sort_values(["device_id", "workload_id"])


def fmt(value: float, digits: int = 2) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.{digits}f}"


def write_report(
    output_path: Path,
    policy: pd.DataFrame,
    threshold: float,
    yolo_deployment: pd.DataFrame,
    gemma_deployment: pd.DataFrame,
    features: pd.DataFrame,
) -> None:
    accepted = gemma_deployment[gemma_deployment.accepted]
    evaluated = accepted[accepted.source_baseline_median_ms.notna()]
    eligible = gemma_deployment[gemma_deployment.runtime_regression_eligible]
    eligible_accepted = eligible[eligible.accepted]
    lines = [
        "# Pre6G Gemma 4 Part C: frozen marker-free period transfer",
        "",
        "## Experimental boundary",
        "",
        "The detector configuration and confidence threshold are trained only on the YOLO Part 7 dataset, then frozen before Gemma is evaluated. Gemma baseline runtimes are joined only after detection for held-out family evaluation; they never enter period detection or feature extraction.",
        "",
        "## Frozen YOLO-trained detector",
        "",
        "| Horizon | Configuration | YOLO profiles | Mean APE | Median APE | Max APE |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in policy.itertuples(index=False):
        lines.append(
            f"| {row.horizon_seconds:.0f}s | `{row.configuration}` | {row.training_profiles} | {row.training_mean_period_ape_percent:.2f}% | {row.training_median_period_ape_percent:.2f}% | {row.training_max_period_ape_percent:.2f}% |"
        )
    yolo_accepted = yolo_deployment[yolo_deployment.accepted]
    lines += [
        "",
        f"- Frozen confidence threshold: {threshold:.2f}",
        f"- YOLO training-policy coverage: {len(yolo_accepted)}/{len(yolo_deployment)} ({len(yolo_accepted) / len(yolo_deployment):.1%})",
        f"- YOLO accepted mean period APE: {yolo_accepted.period_ape_percent.mean():.2f}%",
        f"- YOLO accepted max period APE: {yolo_accepted.period_ape_percent.max():.2f}%",
        "",
        "## Gemma external-family result",
        "",
        f"- Profiles: {len(gemma_deployment)}",
        f"- Accepted: {len(accepted)} ({len(accepted) / len(gemma_deployment):.1%})",
        f"- Runtime-regression-eligible acceptance: {len(eligible_accepted)}/{len(eligible)} ({len(eligible_accepted) / len(eligible):.1%})",
        f"- Feature rows: {len(features)} rows x {len(features.columns)} columns",
    ]
    for horizon, count in accepted.emission_horizon_seconds.value_counts().sort_index().items():
        lines.append(f"- Emitted at {horizon:.0f}s: {count}")
    if len(evaluated):
        lines += [
            f"- Profiles with held-out source baseline evaluation: {len(evaluated)}",
            f"- Mean period APE: {fmt(float(evaluated.period_ape_percent.mean()))}%",
            f"- Median period APE: {fmt(float(evaluated.period_ape_percent.median()))}%",
            f"- P90 period APE: {fmt(float(evaluated.period_ape_percent.quantile(0.9)))}%",
            f"- Max period APE: {fmt(float(evaluated.period_ape_percent.max()))}%",
            f"- Within 10%: {(evaluated.period_ape_percent <= 10).mean():.1%}",
        ]
    lines += [
        "",
        "## Per-workload audit",
        "",
        "| Workload | Accepted | Horizon | Confidence | Detected ms | Hidden source baseline ms | APE | Runtime eligible |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in gemma_deployment.itertuples(index=False):
        lines.append(
            f"| {row.workload_id} | {bool(row.accepted)} | {fmt(float(row.emission_horizon_seconds), 0)} | {fmt(float(row.confidence), 3)} | {fmt(float(row.detected_period_ms))} | {fmt(float(row.source_baseline_median_ms))} | {fmt(float(row.period_ape_percent))}% | {bool(row.runtime_regression_eligible)} |"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transfer the frozen YOLO Part 7 detector to marker-free Gemma profiles."
    )
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--gemma-cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR
        / "pre6g_gemma4_5090_to_4090"
        / "timeseries_1ms"
        / "full",
    )
    parser.add_argument(
        "--gemma-manifest",
        type=Path,
        default=DEFAULT_OUTPUT_DIR
        / "pre6g_gemma4_5090_to_4090"
        / "gemma_manifest.csv",
    )
    parser.add_argument(
        "--yolo-candidates",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "marker_free_cycle_candidates.csv",
    )
    parser.add_argument(
        "--yolo-cache-dir", type=Path, default=DEFAULT_CACHE_DIR / "nsys"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR
        / "model_experiments"
        / "pre6g_gemma4_part_c",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    policy, config_scores, threshold, threshold_scores, yolo_deployment = (
        train_frozen_yolo_policy(args.yolo_candidates, args.yolo_cache_dir)
    )
    candidates = build_candidates(args.gemma_cache_dir.resolve())
    selected, deployment = apply_frozen_policy(candidates, policy, threshold)

    # Ground truth is deliberately joined only after the frozen detector emits.
    manifest = pd.read_csv(args.gemma_manifest)
    audit_columns = [
        "workload_id",
        "source_baseline_median_ms",
        "target_baseline_median_ms",
        "runtime_regression_eligible",
        "target_supported",
    ]
    deployment = deployment.merge(
        manifest[audit_columns], on="workload_id", validate="one_to_one"
    )
    deployment["period_ape_percent"] = (
        (deployment.detected_period_ms - deployment.source_baseline_median_ms).abs()
        / deployment.source_baseline_median_ms
        * 100
    )
    features = extract_features(deployment, args.gemma_cache_dir.resolve())

    candidates.to_csv(output_dir / "gemma_cycle_candidates.csv", index=False)
    selected.to_csv(output_dir / "gemma_frozen_config_detections.csv", index=False)
    deployment.to_csv(output_dir / "gemma_deployment_detections.csv", index=False)
    features.to_csv(output_dir / "gemma_cycle_features.csv", index=False)
    policy.to_csv(output_dir / "frozen_yolo_detector_policy.csv", index=False)
    config_scores.to_csv(output_dir / "yolo_configuration_selection_audit.csv", index=False)
    threshold_scores.to_csv(output_dir / "yolo_threshold_selection_audit.csv", index=False)
    yolo_deployment.to_csv(output_dir / "yolo_frozen_policy_training_audit.csv", index=False)
    contract = {
        "detector_training_family": "YOLO26",
        "external_evaluation_family": "Gemma 4 LoRA SFT",
        "source_device": "RTX5090",
        "adaptive_horizons_seconds": list(ADAPTIVE_HORIZONS),
        "confidence_threshold": threshold,
        "configurations": policy[
            ["horizon_seconds", "configuration"]
        ].to_dict("records"),
        "detector_inputs": ["five timestamped GPU metric channels", "capture horizon"],
        "forbidden_inputs": [
            "NVTX markers",
            "callbacks",
            "source baseline runtime",
            "target baseline runtime",
            "workload configuration metadata",
        ],
        "baseline_usage": "post-detection evaluation only",
    }
    (output_dir / "experiment_contract.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )
    write_report(
        output_dir / "report.md",
        policy,
        threshold,
        yolo_deployment,
        deployment,
        features,
    )
    print(
        json.dumps(
            {
                "gemma_candidates": len(candidates),
                "gemma_profiles": len(deployment),
                "gemma_accepted": int(deployment.accepted.sum()),
                "gemma_feature_rows": len(features),
                "runtime_eligible": int(deployment.runtime_regression_eligible.sum()),
                "model_ready_runtime_rows": int(
                    (deployment.accepted & deployment.runtime_regression_eligible).sum()
                ),
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
