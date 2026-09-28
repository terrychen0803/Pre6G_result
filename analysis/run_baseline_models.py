from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from common import DEFAULT_OUTPUT_DIR


MODELS = [
    "target_train_median",
    "identity_source",
    "global_ratio_median",
    "linear_source_ols",
    "log_source_ols",
]


def predict_one(model: str, train: pd.DataFrame, test: pd.Series) -> float:
    source = train.source_ground_truth_iter_ms.to_numpy(float)
    target = train.target_ground_truth_iter_ms.to_numpy(float)
    source_test = float(test.source_ground_truth_iter_ms)
    if model == "target_train_median":
        return float(np.median(target))
    if model == "identity_source":
        return source_test
    if model == "global_ratio_median":
        return source_test * float(np.median(target / source))
    if model == "linear_source_ols":
        design = np.column_stack([np.ones(len(source)), source])
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
        return float(coefficients[0] + coefficients[1] * source_test)
    if model == "log_source_ols":
        design = np.column_stack([np.ones(len(source)), np.log(source)])
        coefficients = np.linalg.lstsq(design, np.log(target), rcond=None)[0]
        return float(np.exp(coefficients[0] + coefficients[1] * np.log(source_test)))
    raise ValueError(model)


def metric_row(direction: str, model: str, group: pd.DataFrame) -> dict:
    actual = group.actual_ms.to_numpy(float)
    predicted = group.predicted_ms.to_numpy(float)
    error = predicted - actual
    absolute = np.abs(error)
    ape = absolute / actual * 100
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "direction": direction,
        "model": model,
        "n": len(group),
        "mae_ms": float(absolute.mean()),
        "rmse_ms": float(np.sqrt(np.mean(error**2))),
        "mape_percent": float(ape.mean()),
        "median_ape_percent": float(np.median(ape)),
        "max_ape_percent": float(ape.max()),
        "bias_ms": float(error.mean()),
        "r2": float(1 - np.sum(error**2) / denominator) if denominator else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Part 1: leakage-safe source-runtime baselines.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output = args.output_dir.resolve() / "model_experiments" / "part1_baselines"
    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.output_dir.resolve() / "cross_device_samples.csv")
    required = {"source_device_id", "target_device_id", "workload_id", "source_ground_truth_iter_ms", "target_ground_truth_iter_ms"}
    missing = required - set(data.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")
    if (data.source_device_id == data.target_device_id).any():
        raise SystemExit("Source/target leakage: identical devices found.")
    if data.duplicated(["source_device_id", "target_device_id", "workload_id"]).any():
        raise SystemExit("Duplicate directed device/workload rows found.")

    predictions = []
    for (source_device, target_device), direction_data in data.groupby(["source_device_id", "target_device_id"], sort=True):
        direction = f"{source_device}_to_{target_device}"
        if len(direction_data) != direction_data.workload_id.nunique():
            raise SystemExit(f"Direction {direction} does not contain one row per workload.")
        for _, test in direction_data.sort_values("workload_id").iterrows():
            train = direction_data[direction_data.workload_id != test.workload_id]
            if test.workload_id in set(train.workload_id):
                raise RuntimeError("Held-out workload leaked into training data.")
            for model in MODELS:
                predicted = predict_one(model, train, test)
                actual = float(test.target_ground_truth_iter_ms)
                predictions.append({
                    "direction": direction,
                    "source_device_id": source_device,
                    "target_device_id": target_device,
                    "workload_id": test.workload_id,
                    "model": model,
                    "source_runtime_ms": float(test.source_ground_truth_iter_ms),
                    "actual_ms": actual,
                    "predicted_ms": predicted,
                    "error_ms": predicted - actual,
                    "absolute_error_ms": abs(predicted - actual),
                    "ape_percent": abs(predicted - actual) / actual * 100,
                })
    prediction_frame = pd.DataFrame(predictions)
    metrics = []
    for (direction, model), group in prediction_frame.groupby(["direction", "model"], sort=True):
        metrics.append(metric_row(direction, model, group))
    for model, group in prediction_frame.groupby("model", sort=True):
        metrics.append(metric_row("both_directions_macro_rows", model, group))
    metric_frame = pd.DataFrame(metrics).sort_values(["direction", "mape_percent", "model"])
    prediction_frame.to_csv(output / "predictions.csv", index=False)
    metric_frame.to_csv(output / "metrics.csv", index=False)

    lines = [
        "# Part 1: leakage-safe runtime baselines", "",
        "Each direction is trained independently. Every outer fold holds out one workload, and the reverse direction is never used as training data for that fold.", "",
        "| Direction | Model | MAE ms | RMSE ms | MAPE | Median APE | Max APE | R2 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_frame.itertuples():
        lines.append(f"| {row.direction} | {row.model} | {row.mae_ms:.3f} | {row.rmse_ms:.3f} | {row.mape_percent:.2f}% | {row.median_ape_percent:.2f}% | {row.max_ape_percent:.2f}% | {row.r2:.4f} |")
    lines += ["", "## Worst workloads for the best model in each direction", ""]
    for direction, group in metric_frame[metric_frame.direction != "both_directions_macro_rows"].groupby("direction"):
        best = group.sort_values("mape_percent").iloc[0].model
        worst = prediction_frame[(prediction_frame.direction == direction) & (prediction_frame.model == best)].nlargest(5, "ape_percent")
        lines += [f"### {direction}: `{best}`", "", "| Workload | Source ms | Actual ms | Predicted ms | APE |", "|---|---:|---:|---:|---:|"]
        for row in worst.itertuples():
            lines.append(f"| {row.workload_id} | {row.source_runtime_ms:.3f} | {row.actual_ms:.3f} | {row.predicted_ms:.3f} | {row.ape_percent:.2f}% |")
        lines.append("")
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(prediction_frame)} fold predictions and {len(metric_frame)} metric rows to {output}.")


if __name__ == "__main__":
    main()
