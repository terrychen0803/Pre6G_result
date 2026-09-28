from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import DEFAULT_OUTPUT_DIR
from run_baseline_models import metric_row


MAIN_FEATURES = ["log_source_runtime", "log_batch", "log_pixels", "amp_numeric"]
INTERACTION_FEATURES = MAIN_FEATURES + [
    "log_batch_x_log_pixels", "amp_x_log_batch", "amp_x_log_pixels"
]


def prepare(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["log_source_runtime"] = np.log(result.source_ground_truth_iter_ms.astype(float))
    result["log_batch"] = np.log(result.batch.astype(float))
    result["log_pixels"] = np.log(result.imgsz.astype(float) ** 2)
    result["amp_numeric"] = result.amp.astype(str).str.lower().map({"true": 1.0, "false": 0.0})
    if result.amp_numeric.isna().any():
        raise ValueError("AMP contains values that cannot be parsed as booleans.")
    result["log_batch_x_log_pixels"] = result.log_batch * result.log_pixels
    result["amp_x_log_batch"] = result.amp_numeric * result.log_batch
    result["amp_x_log_pixels"] = result.amp_numeric * result.log_pixels
    return result


def model_specs() -> dict[str, dict]:
    return {
        "ridge_main_log_target": {
            "features": MAIN_FEATURES, "target": "log_target",
            "estimator": Ridge(), "grid": [{"model__alpha": value} for value in (0.0, 0.01, 0.1, 1.0, 10.0, 100.0)],
        },
        "ridge_interactions_log_target": {
            "features": INTERACTION_FEATURES, "target": "log_target",
            "estimator": Ridge(), "grid": [{"model__alpha": value} for value in (0.0, 0.01, 0.1, 1.0, 10.0, 100.0)],
        },
        "ridge_interactions_log_ratio": {
            "features": INTERACTION_FEATURES, "target": "log_ratio",
            "estimator": Ridge(), "grid": [{"model__alpha": value} for value in (0.0, 0.01, 0.1, 1.0, 10.0, 100.0)],
        },
        "elasticnet_interactions_log_target": {
            "features": INTERACTION_FEATURES, "target": "log_target",
            "estimator": ElasticNet(max_iter=100000, tol=1e-8),
            "grid": [{"model__alpha": alpha, "model__l1_ratio": ratio} for alpha, ratio in itertools.product((0.001, 0.01, 0.1, 1.0), (0.1, 0.5, 0.9))],
        },
        "huber_interactions_log_target": {
            "features": INTERACTION_FEATURES, "target": "log_target",
            "estimator": HuberRegressor(max_iter=5000),
            "grid": [{"model__epsilon": epsilon, "model__alpha": alpha} for epsilon, alpha in itertools.product((1.1, 1.35, 1.75, 2.0), (0.0, 0.01, 0.1, 1.0))],
        },
    }


def target_values(frame: pd.DataFrame, kind: str) -> np.ndarray:
    target = frame.target_ground_truth_iter_ms.to_numpy(float)
    if kind == "log_target":
        return np.log(target)
    if kind == "log_ratio":
        return np.log(target / frame.source_ground_truth_iter_ms.to_numpy(float))
    raise ValueError(kind)


def restore_prediction(prediction: float, row: pd.Series, kind: str) -> float:
    if kind == "log_target":
        return float(np.exp(prediction))
    if kind == "log_ratio":
        return float(row.source_ground_truth_iter_ms * np.exp(prediction))
    raise ValueError(kind)


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", clone(estimator))])


def choose_parameters(train: pd.DataFrame, spec: dict) -> tuple[dict, float]:
    scores = []
    for order, parameters in enumerate(spec["grid"]):
        ape = []
        for workload in sorted(train.workload_id.unique()):
            inner_train = train[train.workload_id != workload]
            inner_test = train[train.workload_id == workload]
            pipeline = make_pipeline(spec["estimator"])
            pipeline.set_params(**parameters)
            pipeline.fit(inner_train[spec["features"]], target_values(inner_train, spec["target"]))
            raw = float(pipeline.predict(inner_test[spec["features"]])[0])
            predicted = restore_prediction(raw, inner_test.iloc[0], spec["target"])
            actual = float(inner_test.iloc[0].target_ground_truth_iter_ms)
            ape.append(abs(predicted - actual) / actual * 100)
        scores.append((float(np.mean(ape)), order, parameters))
    best = min(scores, key=lambda item: (item[0], item[1]))
    return best[2], best[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Part 2: nested-LOOW workload-aware models.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    output = root / "model_experiments" / "part2_workload_models"
    output.mkdir(parents=True, exist_ok=True)
    data = prepare(pd.read_csv(root / "cross_device_samples.csv"))
    specs = model_specs()
    predictions, coefficients = [], []
    for (source_device, target_device), direction_data in data.groupby(["source_device_id", "target_device_id"], sort=True):
        direction = f"{source_device}_to_{target_device}"
        for workload in sorted(direction_data.workload_id.unique()):
            train = direction_data[direction_data.workload_id != workload]
            test = direction_data[direction_data.workload_id == workload].iloc[0]
            for name, spec in specs.items():
                parameters, inner_mape = choose_parameters(train, spec)
                pipeline = make_pipeline(spec["estimator"])
                pipeline.set_params(**parameters)
                pipeline.fit(train[spec["features"]], target_values(train, spec["target"]))
                raw = float(pipeline.predict(test.to_frame().T[spec["features"]])[0])
                predicted = restore_prediction(raw, test, spec["target"])
                actual = float(test.target_ground_truth_iter_ms)
                parameter_text = json.dumps(parameters, sort_keys=True)
                predictions.append({
                    "direction": direction, "source_device_id": source_device, "target_device_id": target_device,
                    "workload_id": workload, "model": name,
                    "source_runtime_ms": float(test.source_ground_truth_iter_ms), "actual_ms": actual,
                    "predicted_ms": predicted, "error_ms": predicted - actual,
                    "absolute_error_ms": abs(predicted - actual), "ape_percent": abs(predicted - actual) / actual * 100,
                    "selected_parameters": parameter_text, "inner_selection_mape_percent": inner_mape,
                })
                model = pipeline.named_steps["model"]
                for feature, coefficient in zip(spec["features"], model.coef_):
                    coefficients.append({"direction": direction, "held_out_workload": workload, "model": name, "feature": feature, "standardized_coefficient": float(coefficient), "selected_parameters": parameter_text})
    prediction_frame = pd.DataFrame(predictions)
    coefficient_frame = pd.DataFrame(coefficients)
    metrics = []
    for (direction, model), group in prediction_frame.groupby(["direction", "model"], sort=True):
        metrics.append(metric_row(direction, model, group))
    for model, group in prediction_frame.groupby("model", sort=True):
        metrics.append(metric_row("both_directions_combined", model, group))
    metric_frame = pd.DataFrame(metrics).sort_values(["direction", "mape_percent", "model"])
    prediction_frame.to_csv(output / "predictions.csv", index=False)
    metric_frame.to_csv(output / "metrics.csv", index=False)
    coefficient_frame.to_csv(output / "coefficients.csv", index=False)

    baseline = pd.read_csv(root / "model_experiments" / "part1_baselines" / "metrics.csv")
    lines = [
        "# Part 2: workload-aware regularized models", "",
        "Outer evaluation is leave-one-workload-out. Scaling and hyperparameter selection use only the outer training fold; selection uses an inner leave-one-workload-out loop.", "",
        "| Direction | Model | MAE ms | RMSE ms | MAPE | Median APE | Max APE | R2 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_frame.itertuples():
        lines.append(f"| {row.direction} | {row.model} | {row.mae_ms:.3f} | {row.rmse_ms:.3f} | {row.mape_percent:.2f}% | {row.median_ape_percent:.2f}% | {row.max_ape_percent:.2f}% | {row.r2:.4f} |")
    lines += ["", "## Change relative to the best Part 1 baseline", "", "| Direction | Part 1 best | Part 1 MAPE | Part 2 best | Part 2 MAPE | Relative change |", "|---|---|---:|---|---:|---:|"]
    for direction in sorted(prediction_frame.direction.unique()):
        old = baseline[baseline.direction == direction].sort_values("mape_percent").iloc[0]
        new = metric_frame[metric_frame.direction == direction].sort_values("mape_percent").iloc[0]
        change = (new.mape_percent / old.mape_percent - 1) * 100
        lines.append(f"| {direction} | {old.model} | {old.mape_percent:.2f}% | {new.model} | {new.mape_percent:.2f}% | {change:+.1f}% |")
    lines += ["", "## Hyperparameter selection frequency", ""]
    for (direction, model), group in prediction_frame.groupby(["direction", "model"], sort=True):
        counts = Counter(group.selected_parameters)
        formatted = "; ".join(f"`{key}`: {value}" for key, value in counts.most_common())
        lines.append(f"- {direction} / {model}: {formatted}")
    lines += ["", "## Worst workloads for the best Part 2 model", ""]
    for direction, group in metric_frame[metric_frame.direction != "both_directions_combined"].groupby("direction"):
        best = group.sort_values("mape_percent").iloc[0].model
        worst = prediction_frame[(prediction_frame.direction == direction) & (prediction_frame.model == best)].nlargest(5, "ape_percent")
        lines += [f"### {direction}: `{best}`", "", "| Workload | Source ms | Actual ms | Predicted ms | APE |", "|---|---:|---:|---:|---:|"]
        for row in worst.itertuples():
            lines.append(f"| {row.workload_id} | {row.source_runtime_ms:.3f} | {row.actual_ms:.3f} | {row.predicted_ms:.3f} | {row.ape_percent:.2f}% |")
        lines.append("")
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(prediction_frame)} nested-LOOW predictions and {len(metric_frame)} metric rows to {output}.")


if __name__ == "__main__":
    main()
