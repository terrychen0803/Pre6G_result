from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from common import DEFAULT_OUTPUT_DIR


# Nine seconds is the shortest horizon that is reliable enough to emit a
# prediction for most profiles. Longer horizons are retained for an adaptive
# "continue profiling when confidence is low" policy. Profiles which ended
# before a horizon are simply omitted at that horizon.
HORIZONS_SECONDS = (5, 7, 9, 12, 15)
BIN_FACTORS = (1, 2, 4)
TAIL_FRACTIONS = (0.5, 0.65, 0.8)
MIN_PERIOD_MS = 15.0
MAX_PERIOD_MS = 250.0


def robust_scale(values: np.ndarray) -> np.ndarray:
    median = np.median(values, axis=1, keepdims=True)
    q25 = np.percentile(values, 25, axis=1, keepdims=True)
    q75 = np.percentile(values, 75, axis=1, keepdims=True)
    return (values - median) / np.maximum(q75 - q25, 1e-9)


def autocorrelation(values: np.ndarray) -> np.ndarray:
    centered = values - np.mean(values)
    n = len(centered)
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centered, size)
    correlation = np.fft.irfft(spectrum * np.conjugate(spectrum), size)[:n]
    denominator = np.arange(n, 0, -1) * max(float(np.var(centered)), 1e-12)
    return correlation / denominator


def detect(values: np.ndarray, bin_ms: float, tail_fraction: float) -> dict:
    segment_count = max(32, int(values.shape[1] * tail_fraction))
    segment = robust_scale(values[:, -segment_count:])
    channel_acf = np.asarray([autocorrelation(channel) for channel in segment])
    consensus = np.median(channel_acf, axis=0)
    lags_ms = np.arange(len(consensus)) * bin_ms
    valid_indices = np.where(
        (lags_ms >= MIN_PERIOD_MS) & (lags_ms <= MAX_PERIOD_MS)
    )[0]
    minimum_distance = max(1, int(8.0 / bin_ms))
    peaks, properties = find_peaks(
        consensus[valid_indices], prominence=0.02, distance=minimum_distance
    )
    if not len(peaks):
        return {
            "detected_period_ms": np.nan,
            "acf_height": np.nan,
            "acf_prominence": np.nan,
            "channel_agreement": 0.0,
            "cycles_observed": 0.0,
            "confidence": 0.0,
            "candidate_count": 0,
        }
    peak_indices = valid_indices[peaks]
    scores = consensus[peak_indices] + properties["prominences"]
    selected_order = int(np.argmax(scores))
    selected_index = int(peak_indices[selected_order])
    period_ms = float(lags_ms[selected_index])
    tolerance_ms = max(2 * bin_ms, period_ms * 0.08)
    agreement = []
    for acf in channel_acf:
        channel_peaks, _ = find_peaks(
            acf[valid_indices], prominence=0.01, distance=minimum_distance
        )
        channel_periods = lags_ms[valid_indices[channel_peaks]]
        agreement.append(
            bool(len(channel_periods) and np.min(np.abs(channel_periods - period_ms)) <= tolerance_ms)
        )
    channel_agreement = float(np.mean(agreement))
    cycles_observed = float(segment.shape[1] * bin_ms / period_ms)
    prominence = float(properties["prominences"][selected_order])
    height = float(consensus[selected_index])
    confidence = float(
        np.clip(prominence, 0.0, 1.0)
        * channel_agreement
        * min(1.0, cycles_observed / 5.0)
    )
    return {
        "detected_period_ms": period_ms,
        "acf_height": height,
        "acf_prominence": prominence,
        "channel_agreement": channel_agreement,
        "cycles_observed": cycles_observed,
        "confidence": confidence,
        "candidate_count": int(len(peaks)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect cycles without marker tables.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    manifest = pd.read_csv(root / "marker_free_timeseries_manifest.csv")
    rows = []
    configurations = [
        (factor, tail)
        for factor in BIN_FACTORS
        for tail in TAIL_FRACTIONS
    ]
    for item in manifest.itertuples():
        cached = np.load(item.cache_path)
        base_values = cached["values"].astype(float)
        base_bin_ms = float(cached["base_bin_ms"])
        for horizon in HORIZONS_SECONDS:
            horizon_samples = int(horizon * 1000 / base_bin_ms)
            if base_values.shape[1] < horizon_samples:
                continue
            prefix = base_values[:, :horizon_samples]
            for factor, tail_fraction in configurations:
                usable = (prefix.shape[1] // factor) * factor
                rebinned = prefix[:, :usable].reshape(
                    prefix.shape[0], -1, factor
                ).mean(axis=2)
                result = detect(
                    rebinned,
                    bin_ms=base_bin_ms * factor,
                    tail_fraction=tail_fraction,
                )
                rows.append(
                    {
                        "device_id": item.device_id,
                        "workload_id": item.workload_id,
                        "horizon_seconds": horizon,
                        "configuration": json.dumps(
                            {"bin_ms": base_bin_ms * factor, "tail_fraction": tail_fraction},
                            sort_keys=True,
                        ),
                        "bin_ms": base_bin_ms * factor,
                        "tail_fraction": tail_fraction,
                        "segment_start_seconds": horizon * (1 - tail_fraction),
                        **result,
                    }
                )
    predictions = pd.DataFrame(rows).sort_values(
        ["horizon_seconds", "device_id", "workload_id", "configuration"]
    )
    output = root / "marker_free_cycle_candidates.csv"
    predictions.to_csv(output, index=False)
    print(f"Wrote {len(predictions)} marker-free cycle candidates to {output}.")


if __name__ == "__main__":
    main()
