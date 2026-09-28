from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def prepare(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["log_batch"] = np.log(result.batch.astype(float))
    result["log_pixels"] = np.log(result.imgsz.astype(float) ** 2)
    result["amp_numeric"] = result.amp.astype(str).str.lower().map(
        {"true": 1.0, "false": 0.0}
    )
    result["log_profile_sample_count"] = np.log(
        result.source_gpu_sample_count.astype(float)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict target-GPU iteration runtime.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    data = prepare(pd.read_csv(args.input_csv))
    if "source_device_id" in data:
        observed = set(data.source_device_id.astype(str).unique())
        expected = {str(bundle["source_device_id"])}
        if observed != expected:
            raise ValueError(
                f"Input source_device_id must be {expected}; observed {observed}."
            )
    features = bundle["features"]
    missing = [feature for feature in features if feature not in data]
    if missing:
        raise ValueError(f"Missing model inputs: {missing}")
    if data[features].isna().any().any():
        raise ValueError("Model inputs contain missing or invalid values.")
    predicted = np.exp(bundle["pipeline"].predict(data[features]))
    result = pd.DataFrame(
        {
            "source_device_id": bundle["source_device_id"],
            "target_device_id": bundle["target_device_id"],
            "predicted_target_iter_ms": predicted,
        }
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(result)} predictions to {args.output_csv}.")


if __name__ == "__main__":
    main()
