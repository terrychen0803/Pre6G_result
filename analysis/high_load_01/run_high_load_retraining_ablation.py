from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLEAN = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "model_experiments"
    / "yolo_single_device_overhead"
    / "samples.csv"
)
DEFAULT_HIGH = (
    PROJECT_ROOT
    / "analysis"
    / "high_load_01"
    / "overhead_prediction"
    / "clean_model_high_load_predictions.csv"
)
DEFAULT_QC = PROJECT_ROOT / "analysis" / "high_load_01" / "qc" / "run_qc.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "analysis" / "high_load_01" / "overhead_prediction" / "retraining_ablation"
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
PRE_FEATURES = [
    "pre_gpu_util_mean",
    "pre_gpu_util_std",
    "pre_gpu_util_min",
    "pre_gpu_util_max",
    "pre_gpu_mem_used_mean",
    "pre_gpu_power_mean",
]


def pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def choose_alpha(training: pd.DataFrame, features: list[str]) -> tuple[float, pd.DataFrame]:
    groups = training.workload_id.astype(str)
    splitter = GroupKFold(n_splits=5)
    rows = []
    for alpha in ALPHAS:
        errors = []
        for train_index, validation_index in splitter.split(training, groups=groups):
            train = training.iloc[train_index]
            validation = training.iloc[validation_index]
            model = pipeline(alpha)
            model.fit(train[features], np.log(train.actual_overhead_ratio.astype(float)))
            validation_high = validation[validation.condition == "high_load_01"]
            prediction = np.maximum(
                1.0, np.exp(model.predict(validation_high[features]))
            )
            actual = validation_high.actual_overhead_ratio.to_numpy(float)
            errors.extend(np.abs((prediction - actual) * 100).tolist())
        rows.append({"alpha": alpha, "inner_high_overhead_mae_pp": float(np.mean(errors))})
    scores = pd.DataFrame(rows).sort_values(["inner_high_overhead_mae_pp", "alpha"])
    return float(scores.iloc[0].alpha), scores


def outer_predictions(data: pd.DataFrame, features: list[str], model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = []
    audits = []
    for workload in sorted(data.workload_id.unique()):
        training = data[data.workload_id != workload].copy()
        test = data[(data.workload_id == workload) & (data.condition == "high_load_01")].copy()
        if len(test) != 1:
            raise ValueError(f"Expected one high-load test row for {workload}")
        alpha, scores = choose_alpha(training, features)
        scores["held_out_workload"] = workload
        scores["model"] = model_name
        audits.append(scores)
        model = pipeline(alpha)
        model.fit(training[features], np.log(training.actual_overhead_ratio.astype(float)))
        predicted_ratio = float(max(1.0, np.exp(model.predict(test[features])[0])))
        row = test.iloc[0]
        estimated_dry = float(row.detected_period_ms) / predicted_ratio
        predictions.append(
            {
                "model": model_name,
                "workload_id": workload,
                "detector_accepted": bool(row.detector_accepted),
                "selected_alpha": alpha,
                "actual_overhead_ratio": float(row.actual_overhead_ratio),
                "actual_overhead_percent": float(row.actual_overhead_percent),
                "predicted_overhead_ratio": predicted_ratio,
                "predicted_overhead_percent": (predicted_ratio - 1.0) * 100,
                "overhead_error_pp": (predicted_ratio - float(row.actual_overhead_ratio)) * 100,
                "detected_period_ms": float(row.detected_period_ms),
                "period_ape_percent": float(row.period_ape_percent),
                "actual_dry_runtime_ms": float(row.actual_dry_runtime_ms),
                "estimated_dry_runtime_ms": estimated_dry,
                "dry_runtime_ape_percent": abs(estimated_dry - float(row.actual_dry_runtime_ms))
                / float(row.actual_dry_runtime_ms)
                * 100,
            }
        )
    return pd.DataFrame(predictions), pd.concat(audits, ignore_index=True)


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in predictions.groupby("model", sort=True):
        for subset_name, subset in (
            ("forced_all_profiles", group),
            ("confidence_accepted_only", group[group.detector_accepted]),
        ):
            if not len(subset):
                continue
            error = subset.overhead_error_pp
            rows.append(
                {
                    "experiment": "clean_plus_high_train_to_held_out_high",
                    "model": model,
                    "subset": subset_name,
                    "n": len(subset),
                    "detector_coverage": float(group.detector_accepted.mean()),
                    "overhead_mae_pp": float(error.abs().mean()),
                    "overhead_median_ae_pp": float(error.abs().median()),
                    "overhead_p90_ae_pp": float(error.abs().quantile(0.9)),
                    "overhead_rmse_pp": float(np.sqrt(np.mean(np.square(error)))),
                    "overhead_bias_pp": float(error.mean()),
                    "dry_runtime_mape_percent": float(subset.dry_runtime_ape_percent.mean()),
                    "dry_runtime_p90_ape_percent": float(subset.dry_runtime_ape_percent.quantile(0.9)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment C and pre-run Netdata ablation.")
    parser.add_argument("--clean-samples", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--high-predictions", type=Path, default=DEFAULT_HIGH)
    parser.add_argument("--run-qc", type=Path, default=DEFAULT_QC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    clean = pd.read_csv(args.clean_samples)
    clean = clean[clean.device_id == "RTX5090"].copy()
    clean["condition"] = "clean"
    clean["detector_accepted"] = True
    if "period_ape_percent" not in clean:
        clean["period_ape_percent"] = np.nan
    high = pd.read_csv(args.high_predictions).copy()
    high["condition"] = "high_load_01"
    qc = pd.read_csv(args.run_qc)
    profile_qc = qc[
        (qc.condition == "high_load_01") & (qc.run_type == "nsys_gpu_metrics")
    ][["workload_id", *PRE_FEATURES]]
    high = high.merge(profile_qc, on="workload_id", validate="one_to_one")
    for feature in PRE_FEATURES:
        clean[feature] = np.nan

    behavior = sorted(column for column in clean.columns if column.startswith("mf_"))
    columns = sorted(
        set(
            [
                "condition",
                "device_id",
                "workload_id",
                "detector_accepted",
                "detected_period_ms",
                "period_ape_percent",
                "actual_overhead_ratio",
                "actual_overhead_percent",
                "actual_dry_runtime_ms",
            ]
            + behavior
            + PRE_FEATURES
        )
    )
    data = pd.concat([clean[columns], high[columns]], ignore_index=True)
    if len(data) != 48:
        raise ValueError(f"Expected 48 combined RTX5090 rows, got {len(data)}")

    prediction_frames = []
    audit_frames = []
    for name, features in (
        ("original_behavior_only", behavior),
        ("original_plus_pre_netdata", behavior + PRE_FEATURES),
    ):
        predictions, audit = outer_predictions(data, features, name)
        prediction_frames.append(predictions)
        audit_frames.append(audit)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    audit = pd.concat(audit_frames, ignore_index=True)
    summary = summarize(predictions)
    predictions.to_csv(output / "predictions.csv", index=False)
    audit.to_csv(output / "inner_alpha_selection.csv", index=False)
    summary.to_csv(output / "metrics.csv", index=False)

    forced = summary[summary.subset == "forced_all_profiles"].set_index("model")
    original = forced.loc["original_behavior_only"]
    augmented = forced.loc["original_plus_pre_netdata"]
    lines = [
        "# Experiment C: clean + high-load training to held-out high-load workload",
        "",
        "Each outer fold holds out one workload from both clean and high-load data, then tests only its high-load profile. Ridge alpha is selected inside the remaining workload groups using high-load validation error.",
        "",
        "## Forced full-coverage diagnostic",
        "",
        f"- Original behavior features: overhead MAE {original.overhead_mae_pp:.2f} pp; dry-runtime MAPE {original.dry_runtime_mape_percent:.2f}%",
        f"- Original + pre-run Netdata: overhead MAE {augmented.overhead_mae_pp:.2f} pp; dry-runtime MAPE {augmented.dry_runtime_mape_percent:.2f}%",
        f"- Overhead MAE change from Netdata: {augmented.overhead_mae_pp - original.overhead_mae_pp:+.2f} pp",
        "",
        "Clean rows do not have Netdata timestamps. Their pre-run fields are missing and imputed inside each training fold; missing-value indicators allow the model to distinguish absent historical telemetry. This makes the ablation usable but not equivalent to having true clean-condition load measurements.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "experiment_contract.json").write_text(
        json.dumps(
            {
                "outer_split": "leave one workload out; remove both clean and high rows for held-out workload",
                "test_domain": "held-out high_load_01 RTX5090",
                "inner_selection": "five-fold GroupKFold by workload; choose Ridge alpha on high-load validation MAE",
                "target": "log(profile runtime / five-baseline median runtime)",
                "prediction_constraint": "overhead ratio >= 1.0",
                "pre_run_features": PRE_FEATURES,
                "clean_pre_run_features": "unavailable; fold-local median imputation plus missing indicators",
                "trace_profiles_used": False,
                "raw_data_modified": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote retraining ablation to {output}")


if __name__ == "__main__":
    main()
