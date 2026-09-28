from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import DEFAULT_OUTPUT_DIR


WORKLOAD_COLUMNS = ["workload_id", "model", "batch", "imgsz", "amp", "nbs"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build directed source-to-target GPU runtime samples.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    data = pd.read_csv(out / "workload_features.csv")
    schema = pd.read_csv(out / "feature_schema.csv")
    profile_columns = schema.loc[schema.role == "source_profile_predictor", "column"].tolist()
    availability_columns = schema.loc[schema.role == "quality_control", "column"].tolist()
    hardware_columns = [column for column in data.columns if column.startswith("hw_")]
    source_feature_columns = list(dict.fromkeys(profile_columns + availability_columns))
    rows = []
    for workload, group in data.groupby("workload_id", sort=True):
        records = list(group.to_dict("records"))
        for source in records:
            for target in records:
                if source["device_id"] == target["device_id"]:
                    continue
                row = {
                    "source_device_id": source["device_id"],
                    "target_device_id": target["device_id"],
                    **{column: source[column] for column in WORKLOAD_COLUMNS},
                    "source_ground_truth_iter_ms": source["ground_truth_iter_ms"],
                    "target_ground_truth_iter_ms": target["ground_truth_iter_ms"],
                    "target_over_source_runtime_ratio": target["ground_truth_iter_ms"] / source["ground_truth_iter_ms"],
                    "source_over_target_speedup": source["ground_truth_iter_ms"] / target["ground_truth_iter_ms"],
                }
                for column in source_feature_columns:
                    row[f"source_{column}"] = source.get(column)
                for column in hardware_columns:
                    suffix = column.removeprefix("hw_")
                    row[f"source_hw_{suffix}"] = source.get(column)
                    row[f"target_hw_{suffix}"] = target.get(column)
                rows.append(row)
    result = pd.DataFrame(rows).sort_values(["source_device_id", "target_device_id", "workload_id"])
    if (result.source_device_id == result.target_device_id).any():
        raise RuntimeError("Source and target device leakage detected.")
    expected = data.workload_id.nunique() * data.device_id.nunique() * (data.device_id.nunique() - 1)
    if len(result) != expected:
        raise RuntimeError(f"Expected {expected} directed samples, built {len(result)}.")
    result.to_csv(out / "cross_device_samples.csv", index=False)
    roles = []
    for column in result.columns:
        if column in {"source_device_id", "target_device_id", "workload_id"}: role = "identifier"
        elif column in {"target_ground_truth_iter_ms", "target_over_source_runtime_ratio", "source_over_target_speedup"}: role = "label"
        elif column == "source_ground_truth_iter_ms": role = "source_runtime_predictor"
        elif column in WORKLOAD_COLUMNS: role = "workload_predictor"
        elif column.startswith("source_hw_"): role = "source_hardware_predictor"
        elif column.startswith("target_hw_"): role = "target_hardware_predictor"
        elif column.startswith("source_hpct_") or column.endswith("_available"): role = "quality_control"
        else: role = "source_profile_predictor"
        roles.append({"column": column, "role": role, "target_information": column.startswith("target_") and role != "target_hardware_predictor"})
    pd.DataFrame(roles).to_csv(out / "cross_device_feature_schema.csv", index=False)
    print(f"Wrote {len(result)} directed cross-device samples with {len(result.columns)} columns.")


if __name__ == "__main__":
    main()
