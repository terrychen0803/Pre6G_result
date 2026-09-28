from __future__ import annotations

import sys
import sqlite3
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_prediction_table import profile_inputs
from common import load_workloads
from marker_free_features import extract_marker_free_features
from extract_nsys_features import metric_signals
from run_nsys import is_expected_duration_stop


class DnnProjectTests(unittest.TestCase):
    def test_matrix_has_balanced_precision_pairs(self) -> None:
        workloads = load_workloads(ROOT / "configs" / "experiments.csv")
        self.assertEqual(len(workloads), 18)
        fp16 = [item for item in workloads.values() if item["dtype"] == "fp16"]
        bf16 = [item for item in workloads.values() if item["dtype"] == "bf16"]
        structural = lambda item: (
            item["input_dim"], item["hidden_dim"], item["depth"], item["batch_size"]
        )
        self.assertEqual({structural(item) for item in fp16}, {structural(item) for item in bf16})

    def test_marker_free_period(self) -> None:
        samples, period = 8000, 40
        timestamp = np.arange(samples) * 0.25e6
        phase = 2 * np.pi * np.arange(samples) / period
        result = extract_marker_free_features(
            timestamp,
            {"profile.gpu.kernel_active": 2 + np.sin(phase), "profile.gpu.kernel_launch_rate": 5 + 2 * np.cos(phase)},
            sample_interval_ms=0.25,
            min_period_ms=1.0,
            max_period_ms=50.0,
        )
        self.assertAlmostEqual(result["period_detection"]["period_ms"], 10.0, delta=0.5)

    def test_oracle_features_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            profile_inputs({"collection_mode": "oracle", "deployment_features": {"x": 1}})

    def test_documented_gpu_metrics_schema_is_mapped_semantically(self) -> None:
        connection = sqlite3.connect(":memory:")
        connection.execute("CREATE TABLE TARGET_INFO_GPU_METRICS(typeId INT, sourceId INT, typeName TEXT, metricId INT, metricName TEXT)")
        connection.execute("CREATE TABLE GPU_METRICS(rawTimestamp INT, timestamp INT, typeId INT, metricId INT, value INT)")
        connection.execute("INSERT INTO TARGET_INFO_GPU_METRICS VALUES(1,1,'gpu',7,'SM Active')")
        connection.executemany("INSERT INTO GPU_METRICS VALUES(?,?,?,?,?)", [(i, i, 1, 7, i * 10) for i in range(4)])
        signals = metric_signals(connection, ["TARGET_INFO_GPU_METRICS", "GPU_METRICS"], np.arange(4))
        self.assertIn("profile.gpu.sm_active", signals)
        connection.close()

    def test_deployment_accepts_duration_sigterm_codes(self) -> None:
        self.assertTrue(is_expected_duration_stop("deployment", 143))
        self.assertTrue(is_expected_duration_stop("deployment", -15))

    def test_other_collection_exits_are_not_expected_stops(self) -> None:
        self.assertFalse(is_expected_duration_stop("deployment", 1))
        self.assertFalse(is_expected_duration_stop("oracle", 143))


if __name__ == "__main__":
    unittest.main()
