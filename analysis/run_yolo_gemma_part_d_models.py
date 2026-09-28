from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import DEFAULT_OUTPUT_DIR


PART_C = DEFAULT_OUTPUT_DIR / "model_experiments" / "pre6g_yolo_gemma_part_c"
OUTPUT = DEFAULT_OUTPUT_DIR / "model_experiments" / "pre6g_yolo_gemma_part_d"
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def feature_sets(data: pd.DataFrame) -> dict[str, list[str]]:
    hardware = sorted(c for c in data if c.startswith("hw_"))
    behavior = sorted(c for c in data if c.startswith("mf_"))
    compact_suffixes = (
        "_mean",
        "_p90",
        "_active_fraction",
        "_cycle_mean_cv",
        "_harmonic1_fraction",
        "_cycle_repeatability",
    )
    compact = sorted(
        c
        for c in behavior
        if c.startswith("mf_summary_")
        or c.startswith("mf_cross_corr_")
        or c.endswith(compact_suffixes)
    )
    quality = [
        "detector_confidence",
        "emission_horizon_seconds",
        "feature_cycle_count",
    ]
    period = ["log_primary_period_ms"]
    return {
        "period_quality_ridge": hardware + period + quality,
        "behavior_only_ridge": hardware + behavior,
        "period_compact_ridge": hardware + period + compact + quality,
        "period_full_ridge": hardware + period + behavior + quality,
    }


def pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold()),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def target(frame: pd.DataFrame) -> np.ndarray:
    return np.log(frame.target_iteration_ms.to_numpy(float))


def choose_alpha(train: pd.DataFrame, features: list[str]) -> tuple[float, float]:
    splits = min(5, train.workload_id.nunique())
    splitter = GroupKFold(n_splits=splits)
    trials = []
    for order, alpha in enumerate(ALPHAS):
        errors = []
        for fit_index, validation_index in splitter.split(
            train, groups=train.workload_id
        ):
            fit = train.iloc[fit_index]
            validation = train.iloc[validation_index]
            model = pipeline(alpha)
            model.fit(fit[features], target(fit))
            prediction = np.exp(model.predict(validation[features]))
            actual = validation.target_iteration_ms.to_numpy(float)
            errors.extend(np.abs(prediction - actual) / actual * 100)
        trials.append((float(np.mean(errors)), order, alpha))
    best = min(trials, key=lambda item: (item[0], item[1]))
    return float(best[2]), float(best[0])


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    alpha: float,
) -> np.ndarray:
    model = pipeline(alpha)
    model.fit(train[features], target(train))
    return np.exp(model.predict(test[features]))


def records(
    protocol: str,
    model_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> list[dict]:
    alpha, inner_mape = choose_alpha(train, features)
    predictions = fit_predict(train, test, features, alpha)
    rows = []
    for item, prediction in zip(test.to_dict("records"), predictions):
        actual = float(item["target_iteration_ms"])
        rows.append(
            {
                "protocol": protocol,
                "model": model_name,
                "workload_family": item["workload_family"],
                "workload_id": item["workload_id"],
                "actual_ms": actual,
                "predicted_ms": float(prediction),
                "ape_percent": abs(float(prediction) - actual) / actual * 100,
                "selected_alpha": alpha,
                "inner_mape_percent": inner_mape,
                "input_feature_count": len(features),
            }
        )
    return rows


def loo_predictions(
    data: pd.DataFrame, protocol: str, specs: dict[str, list[str]]
) -> list[dict]:
    rows = []
    for workload in sorted(data.workload_id.unique()):
        train = data[data.workload_id != workload]
        test = data[data.workload_id == workload]
        for name, features in specs.items():
            rows.extend(records(protocol, name, train, test, features))
        family_trials = []
        for order, (name, features) in enumerate(specs.items()):
            alpha, score = choose_alpha(train, features)
            family_trials.append((score, order, name, features, alpha))
        _, _, selected_name, selected_features, selected_alpha = min(
            family_trials, key=lambda item: (item[0], item[1])
        )
        prediction = fit_predict(
            train, test, selected_features, selected_alpha
        )[0]
        item = test.iloc[0]
        actual = float(item.target_iteration_ms)
        rows.append(
            {
                "protocol": protocol,
                "model": "nested_selected_ridge",
                "selected_feature_family": selected_name,
                "workload_family": item.workload_family,
                "workload_id": item.workload_id,
                "actual_ms": actual,
                "predicted_ms": float(prediction),
                "ape_percent": abs(float(prediction) - actual) / actual * 100,
                "selected_alpha": selected_alpha,
                "inner_mape_percent": min(family_trials)[0],
                "input_feature_count": len(selected_features),
            }
        )
    return rows


def period_ratio_predictions(
    protocol: str, train: pd.DataFrame, test: pd.DataFrame
) -> list[dict]:
    ratio = np.median(
        train.target_iteration_ms / np.exp(train.log_primary_period_ms)
    )
    predictions = np.exp(test.log_primary_period_ms) * ratio
    rows = []
    for item, prediction in zip(test.to_dict("records"), predictions):
        actual = float(item["target_iteration_ms"])
        rows.append(
            {
                "protocol": protocol,
                "model": "period_median_ratio",
                "workload_family": item["workload_family"],
                "workload_id": item["workload_id"],
                "actual_ms": actual,
                "predicted_ms": float(prediction),
                "ape_percent": abs(float(prediction) - actual) / actual * 100,
                "selected_alpha": np.nan,
                "inner_mape_percent": np.nan,
                "input_feature_count": 1,
            }
        )
    return rows


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = list(
        predictions.groupby(["protocol", "model", "workload_family"], sort=True)
    )
    grouped += [
        ((protocol, model, "all"), group)
        for (protocol, model), group in predictions.groupby(
            ["protocol", "model"], sort=True
        )
        if group.workload_family.nunique() > 1
    ]
    for (protocol, model, family), group in grouped:
        actual = group.actual_ms.to_numpy(float)
        predicted = group.predicted_ms.to_numpy(float)
        ape = group.ape_percent.to_numpy(float)
        rows.append(
            {
                "protocol": protocol,
                "model": model,
                "evaluation_family": family,
                "n": len(group),
                "mae_ms": float(np.mean(np.abs(predicted - actual))),
                "mape_percent": float(np.mean(ape)),
                "median_ape_percent": float(np.median(ape)),
                "p90_ape_percent": float(np.quantile(ape, 0.9)),
                "max_ape_percent": float(np.max(ape)),
                "r2": float(r2_score(actual, predicted)) if len(group) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(PART_C / "directed_samples.csv")
    specs = feature_sets(data)
    yolo = data[data.workload_family == "yolo26"]
    gemma = data[data.workload_family == "gemma4"]
    rows = []
    rows.extend(loo_predictions(yolo, "yolo_only_loo", specs))
    rows.extend(loo_predictions(data, "combined_yolo_gemma_loo", specs))
    for name, features in specs.items():
        rows.extend(records("train_yolo_test_gemma", name, yolo, gemma, features))
        rows.extend(records("train_gemma_test_yolo", name, gemma, yolo, features))
    rows.extend(period_ratio_predictions("train_yolo_test_gemma", yolo, gemma))
    rows.extend(period_ratio_predictions("train_gemma_test_yolo", gemma, yolo))
    predictions = pd.DataFrame(rows)
    if "selected_feature_family" not in predictions:
        predictions["selected_feature_family"] = ""
    else:
        predictions["selected_feature_family"] = predictions[
            "selected_feature_family"
        ].fillna("")
    predictions.to_csv(OUTPUT / "predictions.csv", index=False)
    metrics = summarize(predictions)
    metrics.to_csv(OUTPUT / "metrics.csv", index=False)

    production_trials = []
    for order, (name, features) in enumerate(specs.items()):
        alpha, cv_mape = choose_alpha(data, features)
        production_trials.append((cv_mape, order, name, features, alpha))
    cv_mape, _, best_name, best_features, best_alpha = min(
        production_trials, key=lambda item: (item[0], item[1])
    )
    final_model = pipeline(best_alpha)
    final_model.fit(data[best_features], target(data))
    joblib.dump(final_model, OUTPUT / "yolo_gemma_5090_to_4090_ridge.joblib")
    production = {
        "training_rows": len(data),
        "training_families": {"yolo26": len(yolo), "gemma4": len(gemma)},
        "selected_feature_family": best_name,
        "selected_alpha": best_alpha,
        "grouped_cv_mape_percent": cv_mape,
        "selection_metric_note": (
            "This grouped-CV score is used to choose the final fit. The outer "
            "leave-one-workload-out rows in metrics.csv are the primary unbiased "
            "evaluation of generalization to an unseen workload."
        ),
        "features": best_features,
        "prediction_target": "log RTX4090 unprofiled iteration runtime",
        "excluded": ["DNN", "ViT", "markers", "callbacks", "source baseline"],
    }
    (OUTPUT / "production_model_contract.json").write_text(
        json.dumps(production, indent=2), encoding="utf-8"
    )

    lines = [
        "# Part D: YOLO + Gemma marker-free runtime prediction",
        "",
        "All experiments use RTX5090 marker-free profiling features to predict RTX4090 unprofiled per-iteration runtime. DNN and ViT are excluded.",
        "",
        "## Results",
        "",
        "| Protocol | Evaluation family | Model | N | MAPE | Median APE | P90 | Max | R2 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.sort_values(
        ["protocol", "evaluation_family", "mape_percent"]
    ).itertuples(index=False):
        lines.append(
            f"| {row.protocol} | {row.evaluation_family} | {row.model} | {row.n} | {row.mape_percent:.2f}% | {row.median_ape_percent:.2f}% | {row.p90_ape_percent:.2f}% | {row.max_ape_percent:.2f}% | {row.r2:.4f} |"
        )
    lines += [
        "",
        "## Production fit",
        "",
        f"- Selected family: {best_name}",
        f"- Ridge alpha: {best_alpha}",
        f"- Grouped CV MAPE used for selection: {cv_mape:.2f}%",
        f"- Training rows: {len(data)}",
        "- Primary evaluation: combined_yolo_gemma_loo / all / nested_selected",
        "- The grouped-CV selection score is not a replacement for the outer LOO estimate.",
    ]
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(production, indent=2))


if __name__ == "__main__":
    main()
