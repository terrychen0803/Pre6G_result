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


class VitProjectTests(unittest.TestCase):
    def test_matrix_has_balanced_precision_pairs(self) -> None:
        workloads = load_workloads(ROOT / "configs" / "experiments.csv")
        self.assertEqual(len(workloads), 18)
        fp16 = [item for item in workloads.values() if item["dtype"] == "fp16"]
        bf16 = [item for item in workloads.values() if item["dtype"] == "bf16"]
        structural = lambda item: (
            item["image_size"], item["patch_size"], item["embed_dim"],
            item["depth"], item["num_heads"], item["batch_size"],
        )
        self.assertEqual({structural(item) for item in fp16}, {structural(item) for item in bf16})

    def test_marker_free_period(self) -> None:
        samples, period = 6000, 100
        timestamp = np.arange(samples) * 1e6
        phase = 2 * np.pi * np.arange(samples) / period
        result = extract_marker_free_features(
            timestamp,
            {"profile.gpu.sm_active": 50 + 20 * np.sin(phase), "profile.gpu.memory_active": 30 + 10 * np.cos(phase)},
            sample_interval_ms=1.0,
            min_period_ms=5.0,
            max_period_ms=500.0,
        )
        self.assertAlmostEqual(result["period_detection"]["period_ms"], 100.0, delta=2.0)

    def test_oracle_features_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            profile_inputs({"collection_mode": "oracle", "deployment_features": {"x": 1}})

    def test_generic_event_gpu_metrics_are_mapped_semantically(self) -> None:
        connection = sqlite3.connect(":memory:")
        connection.execute("CREATE TABLE GENERIC_EVENTS(timestamp INT, data TEXT)")
        connection.executemany(
            "INSERT INTO GENERIC_EVENTS VALUES(?,?)",
            [(i, '{\"Tensor Active\": \"%d\"}' % (i * 10)) for i in range(4)],
        )
        signals = metric_signals(connection, ["GENERIC_EVENTS"], np.arange(4))
        self.assertIn("profile.gpu.tensor_active", signals)
        connection.close()

    def test_deployment_accepts_duration_sigterm_codes(self) -> None:
        self.assertTrue(is_expected_duration_stop("deployment", 143))
        self.assertTrue(is_expected_duration_stop("deployment", -15))

    def test_other_collection_exits_are_not_expected_stops(self) -> None:
        self.assertFalse(is_expected_duration_stop("deployment", 1))
        self.assertFalse(is_expected_duration_stop("oracle", 143))


if __name__ == "__main__":
    unittest.main()
