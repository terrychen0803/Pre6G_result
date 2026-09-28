from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def prepare(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    if "detected_period_ms" not in result:
        raise ValueError("Input requires detected_period_ms.")
    if (result.detected_period_ms.astype(float) <= 0).any():
        raise ValueError("detected_period_ms must be positive.")
    if "log_detected_period_ms" not in result:
        result["log_detected_period_ms"] = np.log(
            result.detected_period_ms.astype(float)
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict YOLO dry runtime from same-device Nsight GPU Metrics features."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    data = prepare(pd.read_csv(args.input_csv))
    if "device_id" in data:
        observed = set(data.device_id.astype(str).unique())
        expected = {str(bundle["device_id"])}
        if observed != expected:
            raise ValueError(
                f"Input device_id must be {expected}; observed {observed}."
            )
    features = list(bundle["features"])
    missing = [feature for feature in features if feature not in data]
    if missing:
        raise ValueError(f"Missing model inputs: {missing}")
    numeric = data[features].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("Model inputs contain missing or non-numeric values.")

    raw_prediction = np.exp(bundle["pipeline"].predict(numeric))
    detected_period = data.detected_period_ms.to_numpy(float)
    predicted_dry = np.minimum(raw_prediction, detected_period)
    predicted_ratio = detected_period / predicted_dry
    result = pd.DataFrame(
        {
            "device_id": bundle["device_id"],
            "profiler_mode": bundle["profiler_mode"],
            "predicted_dry_runtime_ms": predicted_dry,
            "predicted_overhead_ratio": predicted_ratio,
            "predicted_overhead_percent": (predicted_ratio - 1.0) * 100.0,
        }
    )
    if "workload_id" in data:
        result.insert(1, "workload_id", data.workload_id.to_numpy())
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(result)} predictions to {args.output_csv}.")


if __name__ == "__main__":
    main()
