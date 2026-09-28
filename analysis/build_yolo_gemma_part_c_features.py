from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import DEFAULT_OUTPUT_DIR, PROJECT_ROOT
from run_marker_free_runtime_models import HARDWARE_FIELDS


OUTPUT = DEFAULT_OUTPUT_DIR / "model_experiments" / "pre6g_yolo_gemma_part_c"


def hardware_features(hardware: pd.DataFrame) -> dict[str, float]:
    source = hardware.loc["RTX5090"]
    target = hardware.loc["RTX4090"]
    result = {}
    for field in HARDWARE_FIELDS:
        source_value = float(source[field])
        target_value = float(target[field])
        if source_value > 0 and target_value > 0:
            result[f"hw_log_ratio_{field}"] = float(np.log(target_value / source_value))
        else:
            result[f"hw_difference_{field}"] = target_value - source_value
    return result


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    yolo = pd.read_csv(DEFAULT_OUTPUT_DIR / "marker_free_cycle_features.csv")
    yolo = yolo[yolo.device_id == "RTX5090"].copy()
    gemma_root = (
        DEFAULT_OUTPUT_DIR
        / "model_experiments"
        / "pre6g_multiscale_detector_part_c2"
    )
    gemma = pd.read_csv(gemma_root / "gemma_cycle_features.csv")
    gemma_manifest = pd.read_csv(
        DEFAULT_OUTPUT_DIR
        / "pre6g_gemma4_5090_to_4090"
        / "gemma_manifest.csv"
    )
    eligible_gemma = set(
        gemma_manifest[gemma_manifest.runtime_regression_eligible].workload_id
    )
    gemma = gemma[gemma.workload_id.isin(eligible_gemma)].copy()

    yolo_schema = list(yolo.columns)
    gemma_schema = list(gemma.columns)
    if yolo_schema != gemma_schema:
        missing_yolo = sorted(set(gemma_schema) - set(yolo_schema))
        missing_gemma = sorted(set(yolo_schema) - set(gemma_schema))
        raise ValueError(
            f"Cycle feature schema mismatch: missing_yolo={missing_yolo}, "
            f"missing_gemma={missing_gemma}"
        )

    yolo_labels = pd.read_csv(DEFAULT_OUTPUT_DIR / "ground_truth.csv")
    yolo_labels = yolo_labels[
        (yolo_labels.device_id == "RTX4090") & yolo_labels.ground_truth_valid
    ][["workload_id", "ground_truth_iter_ms"]].rename(
        columns={"ground_truth_iter_ms": "target_iteration_ms"}
    )
    gemma_labels = gemma_manifest[
        gemma_manifest.runtime_regression_eligible
    ][
        [
            "workload_id",
            "source_baseline_median_ms",
            "target_baseline_median_ms",
        ]
    ].rename(columns={"target_baseline_median_ms": "target_iteration_ms"})

    yolo = yolo.merge(yolo_labels, on="workload_id", validate="one_to_one")
    gemma = gemma.merge(gemma_labels, on="workload_id", validate="one_to_one")
    yolo["workload_family"] = "yolo26"
    yolo["source_baseline_median_ms"] = np.nan
    gemma["workload_family"] = "gemma4"
    data = pd.concat([yolo, gemma], ignore_index=True)
    data["source_device_id"] = "RTX5090"
    data["target_device_id"] = "RTX4090"
    data["direction"] = "RTX5090_to_RTX4090"
    data["log_primary_period_ms"] = np.log(data.detected_period_ms.astype(float))
    hardware = pd.read_csv(
        PROJECT_ROOT / "hardware" / "hardware_descriptors.csv"
    ).set_index("device_id")
    for name, value in hardware_features(hardware).items():
        data[name] = value
    data = data.sort_values(["workload_family", "workload_id"]).reset_index(drop=True)

    feature_columns = [
        column
        for column in data.columns
        if column.startswith("mf_")
        or column.startswith("hw_")
        or column
        in {
            "log_primary_period_ms",
            "detector_confidence",
            "emission_horizon_seconds",
            "feature_cycle_count",
        }
    ]
    forbidden = {
        "source_baseline_median_ms",
        "target_iteration_ms",
        "workload_family",
        "workload_id",
        "device_id",
    }
    if forbidden.intersection(feature_columns):
        raise RuntimeError("A forbidden field entered deployment features")
    numeric = data[feature_columns].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("Non-finite deployment feature value")

    data.to_csv(OUTPUT / "directed_samples.csv", index=False)
    pd.DataFrame(
        {
            "feature": feature_columns,
            "family": [
                "hardware"
                if value.startswith("hw_")
                else "period_quality"
                if value
                in {
                    "log_primary_period_ms",
                    "detector_confidence",
                    "emission_horizon_seconds",
                    "feature_cycle_count",
                }
                else "marker_free_cycle_behavior"
                for value in feature_columns
            ],
            "deployment_valid": True,
        }
    ).to_csv(OUTPUT / "feature_manifest.csv", index=False)
    contract = {
        "direction": "RTX5090_to_RTX4090",
        "families": {"yolo26": len(yolo), "gemma4": len(gemma)},
        "rows": len(data),
        "cycle_schema_columns": len(yolo_schema),
        "deployment_feature_count": len(feature_columns),
        "deployment_inputs": [
            "marker-free source GPU metric time series",
            "marker-free detected source period and quality",
            "source and target hardware descriptors",
        ],
        "label_only": ["RTX4090 unprofiled baseline iteration runtime"],
        "audit_only": ["source baseline iteration runtime", "workload family"],
        "excluded": [
            "NVTX markers",
            "callbacks",
            "workload configuration metadata",
            "target profiling metrics",
            "DNN",
            "ViT",
        ],
    }
    (OUTPUT / "experiment_contract.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
