from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "analysis"))

from detect_multiscale_marker_free_cycles import add_yolo_oracle, run_collection  # noqa: E402
from extract_marker_free_cycle_features import extract_profile  # noqa: E402


DEFAULT_MANIFEST = (
    PROJECT_ROOT / "analysis" / "high_load_01" / "period_detection" / "marker_free_timeseries_manifest.csv"
)
DEFAULT_ORACLE_CACHE = PROJECT_ROOT / "analysis" / "high_load_01" / "cache" / "nsys"
DEFAULT_OVERHEAD = PROJECT_ROOT / "analysis" / "high_load_01" / "overhead" / "gpu_metrics_overhead_by_workload.csv"
DEFAULT_MODEL = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "model_experiments"
    / "yolo_single_device_overhead"
    / "models"
    / "RTX5090_nsys_gpu_metrics_overhead_ridge.joblib"
)
DEFAULT_CLEAN_SAMPLES = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "model_experiments"
    / "yolo_single_device_overhead"
    / "samples.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "analysis" / "high_load_01" / "overhead_prediction"


def metrics(data: pd.DataFrame, name: str) -> dict:
    overhead_error = data.predicted_overhead_percent - data.actual_overhead_percent
    dry_ape = (
        (data.estimated_dry_runtime_ms - data.actual_dry_runtime_ms).abs()
        / data.actual_dry_runtime_ms
        * 100
    )
    return {
        "evaluation": name,
        "n": len(data),
        "overhead_mae_pp": float(overhead_error.abs().mean()),
        "overhead_median_ae_pp": float(overhead_error.abs().median()),
        "overhead_p90_ae_pp": float(overhead_error.abs().quantile(0.9)),
        "overhead_rmse_pp": float(np.sqrt(np.mean(np.square(overhead_error)))),
        "overhead_bias_pp": float(overhead_error.mean()),
        "dry_runtime_mape_percent": float(dry_ape.mean()),
        "dry_runtime_median_ape_percent": float(dry_ape.median()),
        "dry_runtime_p90_ape_percent": float(dry_ape.quantile(0.9)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the existing clean-trained RTX5090 overhead model on high load.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--oracle-cache", type=Path, default=DEFAULT_ORACLE_CACHE)
    parser.add_argument("--overhead", type=Path, default=DEFAULT_OVERHEAD)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--clean-samples", type=Path, default=DEFAULT_CLEAN_SAMPLES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    cache_paths = [Path(value) for value in manifest.cache_path]
    horizons, candidates, deployment = run_collection(cache_paths)
    deployment = add_yolo_oracle(deployment, args.oracle_cache)
    deployment.to_csv(output / "high_load_multiscale_deployment.csv", index=False)
    horizons.to_csv(output / "high_load_multiscale_horizons.csv", index=False)
    candidates.to_csv(output / "high_load_multiscale_candidates.csv", index=False)

    feature_rows = []
    path_lookup = {
        (row.device_id, row.workload_id): Path(row.cache_path)
        for row in manifest.itertuples(index=False)
    }
    for item in deployment.itertuples(index=False):
        detection = pd.Series(item._asdict())
        if not np.isfinite(float(detection.emission_horizon_seconds)):
            detection.emission_horizon_seconds = float(detection.horizon_seconds)
        features = extract_profile(
            detection, path_lookup[(item.device_id, item.workload_id)]
        )
        features["detector_accepted"] = bool(item.accepted)
        features["selection_reason"] = item.selection_reason
        features["oracle_period_ms"] = float(item.oracle_period_ms)
        features["period_ape_percent"] = float(item.period_ape_percent)
        feature_rows.append(features)
    features = pd.DataFrame(feature_rows).sort_values(["device_id", "workload_id"])
    features.to_csv(output / "high_load_marker_free_cycle_features.csv", index=False)

    bundle = joblib.load(args.model)
    model_features = list(bundle["features"])
    missing = [feature for feature in model_features if feature not in features]
    if missing:
        raise ValueError(f"Missing production model features: {missing}")
    numeric = features[model_features].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("High-load production inputs contain missing values.")
    raw_log_ratio = bundle["pipeline"].predict(numeric)
    predicted_ratio = np.maximum(1.0, np.exp(raw_log_ratio))

    clean_samples = pd.read_csv(args.clean_samples)
    clean_samples = clean_samples[clean_samples.device_id == "RTX5090"]
    clean_numeric = clean_samples[model_features].apply(pd.to_numeric, errors="coerce")
    train_min = clean_numeric.min()
    train_max = clean_numeric.max()
    train_median = clean_numeric.median()
    train_iqr = clean_numeric.quantile(0.75) - clean_numeric.quantile(0.25)
    train_std = clean_numeric.std(ddof=1)
    scales = train_iqr.where(train_iqr > 0, train_std).where(lambda value: value > 0, 1.0)
    outside = numeric.lt(train_min, axis=1) | numeric.gt(train_max, axis=1)
    robust_z = numeric.sub(train_median, axis=1).div(scales, axis=1).abs()
    ood_profiles = features[["device_id", "workload_id", "detector_accepted"]].copy()
    ood_profiles["feature_outside_clean_range_count"] = outside.sum(axis=1).to_numpy()
    ood_profiles["feature_outside_clean_range_fraction"] = outside.mean(axis=1).to_numpy()
    ood_profiles["median_absolute_robust_z"] = robust_z.median(axis=1).to_numpy()
    ood_profiles["max_absolute_robust_z"] = robust_z.max(axis=1).to_numpy()
    ood_profiles.to_csv(output / "feature_ood_by_profile.csv", index=False)
    ood_features = pd.DataFrame(
        {
            "feature": model_features,
            "clean_min": train_min.to_numpy(),
            "clean_max": train_max.to_numpy(),
            "clean_median": train_median.to_numpy(),
            "clean_iqr_or_std_scale": scales.to_numpy(),
            "high_min": numeric.min().to_numpy(),
            "high_max": numeric.max().to_numpy(),
            "high_median": numeric.median().to_numpy(),
            "high_outside_clean_range_fraction": outside.mean().to_numpy(),
            "median_shift_in_clean_robust_scales": (
                (numeric.median() - train_median) / scales
            ).to_numpy(),
        }
    )
    ood_features.to_csv(output / "feature_ood_by_feature.csv", index=False)

    labels = pd.read_csv(args.overhead)
    labels = labels[labels.condition == "high_load_01"].copy()
    label_columns = [
        "device_id",
        "workload_id",
        "profile_runtime_ms",
        "baseline_median_ms",
        "overhead_vs_median_ratio",
        "overhead_vs_median_percent",
        "baseline_cv_percent",
    ]
    predictions = features.merge(
        labels[label_columns], on=["device_id", "workload_id"], validate="one_to_one"
    )
    predictions["predicted_overhead_ratio"] = predicted_ratio
    predictions["raw_predicted_log_overhead_ratio"] = raw_log_ratio
    predictions["predicted_overhead_percent"] = (predicted_ratio - 1.0) * 100
    predictions["actual_profile_runtime_ms"] = predictions.profile_runtime_ms / 108.0
    predictions["actual_dry_runtime_ms"] = predictions.baseline_median_ms / 108.0
    predictions["actual_overhead_ratio"] = predictions.overhead_vs_median_ratio + 1.0
    predictions["actual_overhead_percent"] = predictions.overhead_vs_median_percent
    predictions["estimated_dry_runtime_ms"] = (
        predictions.detected_period_ms / predictions.predicted_overhead_ratio
    )
    predictions["overhead_error_pp"] = (
        predictions.predicted_overhead_percent - predictions.actual_overhead_percent
    )
    predictions["dry_runtime_ape_percent"] = (
        (predictions.estimated_dry_runtime_ms - predictions.actual_dry_runtime_ms).abs()
        / predictions.actual_dry_runtime_ms
        * 100
    )
    predictions.to_csv(output / "clean_model_high_load_predictions.csv", index=False)

    accepted = predictions[predictions.detector_accepted]
    summary_rows = [metrics(predictions, "forced_all_profiles")]
    if len(accepted):
        summary_rows.append(metrics(accepted, "confidence_accepted_only"))
    summary = pd.DataFrame(summary_rows)
    summary.insert(1, "detector_coverage", float(deployment.accepted.mean()))
    summary.to_csv(output / "transfer_metrics.csv", index=False)

    all_metrics = summary[summary.evaluation == "forced_all_profiles"].iloc[0]
    lines = [
        "# Clean-trained RTX5090 GPU-metrics overhead model on high load",
        "",
        "- Production model family: behavior_only_ridge",
        f"- Model feature count: {len(model_features)}",
        f"- High-load multiscale detector coverage: {int(deployment.accepted.sum())}/{len(deployment)} ({deployment.accepted.mean():.1%})",
        f"- Median fraction of model features outside the clean training range: {ood_profiles.feature_outside_clean_range_fraction.median():.1%}",
        f"- Maximum per-profile fraction outside the clean range: {ood_profiles.feature_outside_clean_range_fraction.max():.1%}",
        "",
        "## Forced full-coverage diagnostic",
        "",
        "Rejected detections are forced through feature extraction only to diagnose domain shift; this is not valid deployment behavior.",
        "",
        f"- Overhead MAE: {all_metrics.overhead_mae_pp:.2f} pp",
        f"- Overhead RMSE: {all_metrics.overhead_rmse_pp:.2f} pp",
        f"- Overhead P90 absolute error: {all_metrics.overhead_p90_ae_pp:.2f} pp",
        f"- Overhead bias: {all_metrics.overhead_bias_pp:+.2f} pp",
        f"- Estimated dry-runtime MAPE: {all_metrics.dry_runtime_mape_percent:.2f}%",
        f"- Estimated dry-runtime P90 APE: {all_metrics.dry_runtime_p90_ape_percent:.2f}%",
        "",
    ]
    if len(accepted):
        accepted_metrics = summary[summary.evaluation == "confidence_accepted_only"].iloc[0]
        lines += [
            "## Confidence-accepted subset",
            "",
            f"- Profiles: {int(accepted_metrics.n)}",
            f"- Overhead MAE: {accepted_metrics.overhead_mae_pp:.2f} pp",
            f"- Estimated dry-runtime MAPE: {accepted_metrics.dry_runtime_mape_percent:.2f}%",
            "",
        ]
    lines += [
        "The production model was trained only on clean RTX5090 profiles. High-load baseline median is used only as the post-hoc label.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "experiment_contract.json").write_text(
        json.dumps(
            {
                "model_path": str(args.model.resolve()),
                "training_domain": "clean RTX5090",
                "test_domain": "high_load_01 RTX5090",
                "profile_mode": "nsys_gpu_metrics",
                "trace_profiles_used": False,
                "label_reference": "median of five high-load baselines",
                "rejected_profile_policy": "excluded for deployment; forced only in separate diagnostic",
                "raw_data_modified": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote overhead transfer evaluation to {output}")


if __name__ == "__main__":
    main()
