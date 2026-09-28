from __future__ import annotations

import math
from typing import Any

import numpy as np


def robust_standardize(matrix: np.ndarray) -> np.ndarray:
    median = np.median(matrix, axis=0)
    scale = np.percentile(matrix, 75, axis=0) - np.percentile(matrix, 25, axis=0)
    fallback = np.std(matrix, axis=0)
    scale = np.where(scale > 1e-12, scale, fallback)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (matrix - median) / scale


def normalized_autocorrelation(signal: np.ndarray) -> np.ndarray:
    size = len(signal)
    fft_size = 1 << (2 * size - 1).bit_length()
    spectrum = np.fft.rfft(signal, fft_size)
    correlation = np.fft.irfft(spectrum * np.conjugate(spectrum), fft_size)[:size]
    correlation /= np.arange(size, 0, -1, dtype=float)
    return correlation / correlation[0] if correlation[0] > 1e-12 else np.zeros(size)


def detect_period(
    matrix: np.ndarray,
    sample_interval_ms: float,
    min_period_ms: float,
    max_period_ms: float,
) -> dict[str, Any]:
    normalized = robust_standardize(matrix)
    _, _, right = np.linalg.svd(normalized, full_matrices=False)
    component = normalized @ right[0]
    component -= np.mean(component)
    correlation = normalized_autocorrelation(component)
    min_lag = max(2, int(math.ceil(min_period_ms / sample_interval_ms)))
    max_lag = min(len(component) // 3, int(max_period_ms / sample_interval_ms))
    if max_lag <= min_lag:
        raise ValueError("Capture is too short for the requested period range")
    candidates = [
        lag for lag in range(min_lag + 1, max_lag)
        if correlation[lag] >= correlation[lag - 1]
        and correlation[lag] > correlation[lag + 1]
    ]
    if not candidates:
        candidates = [int(np.argmax(correlation[min_lag:max_lag + 1])) + min_lag]

    scored: list[tuple[float, int, float]] = []
    for lag in candidates:
        count = len(matrix) // lag
        if count < 3:
            continue
        cycles = matrix[:count * lag].reshape(count, lag, matrix.shape[1])
        template = np.median(cycles, axis=0)
        denominator = float(np.linalg.norm(template)) + 1e-12
        similarities = [
            1.0 - min(1.0, float(np.linalg.norm(cycle - template)) / denominator)
            for cycle in cycles
        ]
        template_score = float(np.mean(similarities))
        score = 0.65 * max(0.0, float(correlation[lag])) + 0.35 * template_score
        scored.append((score, lag, template_score))
    if not scored:
        raise ValueError("Fewer than three candidate cycles were captured")
    best = max(item[0] for item in scored)
    score, lag, template_score = min(
        (item for item in scored if item[0] >= best - 0.02), key=lambda item: item[1]
    )
    return {
        "period_samples": int(lag),
        "period_ms": float(lag * sample_interval_ms),
        "confidence": float(max(0.0, min(1.0, score))),
        "autocorrelation": float(correlation[lag]),
        "template_similarity": template_score,
        "complete_cycles": int(len(matrix) // lag),
    }

def distribution(values: np.ndarray) -> dict[str, float]:
    result = {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }
    for quantile in (10, 25, 50, 75, 90):
        result[f"p{quantile}"] = float(np.percentile(values, quantile))
    result["high_duty_ratio"] = float(np.mean(values > result["p50"] + result["std"]))
    return result


def extract_marker_free_features(
    timestamps_ns: np.ndarray,
    signals: dict[str, np.ndarray],
    sample_interval_ms: float = 1.0,
    min_period_ms: float = 1.0,
    max_period_ms: float = 10_000.0,
) -> dict[str, Any]:
    if len(timestamps_ns) < 4:
        raise ValueError("At least four timestamped samples are required")
    order = np.argsort(timestamps_ns)
    timestamps = np.asarray(timestamps_ns, dtype=float)[order]
    unique = np.concatenate(([True], np.diff(timestamps) > 0))
    timestamps = timestamps[unique]
    step_ns = sample_interval_ms * 1e6
    grid = np.arange(timestamps[0], timestamps[-1] + step_ns * 0.5, step_ns)

    names: list[str] = []
    columns: list[np.ndarray] = []
    for name, raw in sorted(signals.items()):
        values = np.asarray(raw, dtype=float)[order][unique]
        finite = np.isfinite(values)
        if np.count_nonzero(finite) < 4:
            continue
        interpolated = np.interp(grid, timestamps[finite], values[finite])
        if np.std(interpolated) <= 1e-12:
            continue
        names.append(name)
        columns.append(interpolated)
    if not columns:
        raise ValueError("No varying low-level profiler signals were found")
    matrix = np.column_stack(columns)
    trim = max(1, int(len(grid) * 0.05))
    if len(grid) - 2 * trim >= 30:
        grid = grid[trim:-trim]
        matrix = matrix[trim:-trim]

    period = detect_period(matrix, sample_interval_ms, min_period_ms, max_period_ms)
    lag = period["period_samples"]
    count = len(matrix) // lag
    cycles = matrix[:count * lag].reshape(count, lag, matrix.shape[1])
    per_signal: dict[str, dict[str, float]] = {}
    for index, name in enumerate(names):
        features = distribution(cycles[:, :, index].reshape(-1))
        means = np.mean(cycles[:, :, index], axis=1)
        features["cycle_mean_cv"] = float(np.std(means) / (abs(np.mean(means)) + 1e-12))
        per_signal[name] = features

    correlations: dict[str, float] = {}
    if len(names) >= 2:
        values = np.corrcoef(matrix, rowvar=False)
        for left in range(len(names)):
            for right_index in range(left + 1, len(names)):
                correlations[f"{names[left]}__{names[right_index]}"] = float(values[left, right_index])
    return {
        "schema_version": 1,
        "feature_contract": "marker_free_semantic_low_level_timeseries",
        "capture_duration_ms": float((grid[-1] - grid[0]) / 1e6),
        "sample_interval_ms": sample_interval_ms,
        "sample_count": int(len(grid)),
        "signal_count": len(names),
        "period_detection": period,
        "per_signal": per_signal,
        "cross_signal_correlation": correlations,
    }
