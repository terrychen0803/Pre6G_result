from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from common import DEFAULT_CACHE_DIR, DEFAULT_OUTPUT_DIR
from detect_marker_free_cycles import autocorrelation
from detect_multiscale_marker_free_cycles import detect_profile as detect_long_profile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENT_BIN_MS = 0.05
EVENT_MIN_PERIOD_MS = 0.20
EVENT_MAX_PERIOD_MS = 15.0
EVENT_WINDOWS_SECONDS = (3, 5)
EVENT_HASH_BUCKETS = 8
EVENT_ROUTE_CONFIDENCE = 0.75
EVENT_ROUTE_PROMINENCE = 0.35


def code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def event_candidates(
    starts: np.ndarray,
    names: np.ndarray,
    capture_end_ns: int,
    window_seconds: int,
) -> list[dict]:
    window_ns = int(window_seconds * 1e9)
    lower = capture_end_ns - window_ns
    keep = starts >= lower
    starts = starts[keep]
    names = names[keep]
    bin_ns = int(EVENT_BIN_MS * 1e6)
    bin_count = int(window_ns // bin_ns) + 1
    indices = ((starts - lower) // bin_ns).astype(np.int64)
    signals = []
    for bucket in range(EVENT_HASH_BUCKETS):
        signal = np.bincount(
            indices[names % EVENT_HASH_BUCKETS == bucket], minlength=bin_count
        ).astype(float)
        if np.var(signal) > 0:
            signals.append(np.log1p(signal))
    if not signals:
        return []
    channel_acf = np.asarray([autocorrelation(signal) for signal in signals])
    consensus = np.median(channel_acf, axis=0)
    lag_ms = np.arange(len(consensus)) * EVENT_BIN_MS
    valid = np.where(
        (lag_ms >= EVENT_MIN_PERIOD_MS) & (lag_ms <= EVENT_MAX_PERIOD_MS)
    )[0]
    peaks, properties = find_peaks(
        consensus[valid], prominence=0.005, distance=2
    )
    if not len(peaks):
        return []
    peak_indices = valid[peaks]
    strengths = np.clip(consensus[peak_indices], 0.0, 1.0) + np.clip(
        properties["prominences"], 0.0, 1.5
    )
    order = np.argsort(strengths)[::-1][:8]
    rows = []
    for rank, selected in enumerate(order, start=1):
        index = int(peak_indices[selected])
        period = float(lag_ms[index])
        tolerance = max(2 * EVENT_BIN_MS, 0.05 * period)
        votes = []
        for acf in channel_acf:
            local_peaks, _ = find_peaks(acf[valid], prominence=0.005, distance=2)
            local_periods = lag_ms[valid[local_peaks]]
            votes.append(
                bool(
                    len(local_periods)
                    and np.min(np.abs(local_periods - period)) <= tolerance
                )
            )
        rows.append(
            {
                "rank": rank,
                "period_ms": period,
                "acf_height": float(consensus[index]),
                "acf_prominence": float(properties["prominences"][selected]),
                "bucket_agreement": float(np.mean(votes)),
                "window_seconds": window_seconds,
                "event_count": int(len(starts)),
            }
        )
    return rows


def detect_fine_events(database: Path) -> dict:
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    try:
        capture_end = int(
            connection.execute(
                "SELECT MAX(start) FROM CUPTI_ACTIVITY_KIND_KERNEL"
            ).fetchone()[0]
        )
        lower = capture_end - int(max(EVENT_WINDOWS_SECONDS) * 1e9)
        rows = connection.execute(
            "SELECT start, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL "
            "WHERE start>=? ORDER BY start",
            (lower,),
        ).fetchall()
    finally:
        connection.close()
    starts = np.asarray([row[0] for row in rows], dtype=np.int64)
    names = np.asarray([row[1] for row in rows], dtype=np.int64)
    by_window = {
        window: event_candidates(starts, names, capture_end, window)
        for window in EVENT_WINDOWS_SECONDS
    }
    primary = by_window[max(EVENT_WINDOWS_SECONDS)][0]
    shorter = by_window[min(EVENT_WINDOWS_SECONDS)]
    if shorter:
        relative_errors = [
            abs(primary["period_ms"] - item["period_ms"])
            / max(primary["period_ms"], item["period_ms"])
            for item in shorter
        ]
        best_index = int(np.argmin(relative_errors))
        stability = float(np.clip(1.0 - relative_errors[best_index] / 0.10, 0, 1))
        matched_short_period = float(shorter[best_index]["period_ms"])
    else:
        stability = 0.0
        matched_short_period = np.nan
    confidence = float(
        np.clip(
            0.45 * min(1.0, primary["acf_prominence"] / 0.5)
            + 0.25 * primary["bucket_agreement"]
            + 0.20 * stability
            + 0.10 * min(1.0, max(0.0, primary["acf_height"]) / 0.5),
            0,
            1,
        )
    )
    return {
        "fine_event_period_ms": primary["period_ms"],
        "fine_event_confidence": confidence,
        "fine_event_acf_height": primary["acf_height"],
        "fine_event_acf_prominence": primary["acf_prominence"],
        "fine_event_bucket_agreement": primary["bucket_agreement"],
        "fine_event_cross_window_stability": stability,
        "fine_event_matched_3s_period_ms": matched_short_period,
        "fine_event_count_5s": primary["event_count"],
        "fine_event_candidates_json": json.dumps(by_window[max(EVENT_WINDOWS_SECONDS)][:5]),
    }


def discover_profiles(family: str) -> list[dict]:
    if family == "yolo":
        manifest = pd.read_csv(DEFAULT_OUTPUT_DIR / "marker_free_timeseries_manifest.csv")
        return [
            {
                "family": family,
                "device_id": row.device_id,
                "workload_id": row.workload_id,
                "timeseries": Path(row.cache_path),
                "sqlite": DEFAULT_CACHE_DIR
                / "nsys"
                / row.device_id
                / row.workload_id
                / "nsys_gpu_metrics"
                / "profile.sqlite",
            }
            for row in manifest.itertuples(index=False)
        ]
    mapping = {
        "gemma": "pre6g_gemma4_5090_to_4090",
        "dnn": "pre6g_dnn_5090_to_4090",
        "vit": "pre6g_vit_5090_to_4090",
    }
    stem = mapping[family]
    manifest_name = "gemma_manifest.csv" if family == "gemma" else "manifest.csv"
    manifest = pd.read_csv(DEFAULT_OUTPUT_DIR / stem / manifest_name)
    profiles = []
    for row in manifest.itertuples(index=False):
        sqlite_path = Path(row.profile_sqlite_path) if hasattr(row, "profile_sqlite_path") else DEFAULT_CACHE_DIR / stem / "sqlite" / f"{row.workload_id}.sqlite"
        profiles.append(
            {
                "family": family,
                "device_id": "RTX5090",
                "workload_id": row.workload_id,
                "timeseries": DEFAULT_CACHE_DIR / stem / "timeseries_1ms" / "full" / f"{row.workload_id}.npz",
                "sqlite": sqlite_path,
            }
        )
    return profiles


def detect_one(profile: dict) -> dict:
    _, long_candidates, long_detection = detect_long_profile(profile["timeseries"])
    fine = detect_fine_events(profile["sqlite"])
    use_fine = (
        fine["fine_event_confidence"] >= EVENT_ROUTE_CONFIDENCE
        and fine["fine_event_acf_prominence"] >= EVENT_ROUTE_PROMINENCE
    )
    primary_period = (
        fine["fine_event_period_ms"]
        if use_fine
        else float(long_detection["detected_period_ms"])
    )
    primary_confidence = (
        fine["fine_event_confidence"]
        if use_fine
        else float(long_detection["confidence"])
    )
    long_alternatives = [
        {
            "period_ms": float(item["period_ms"]),
            "score": float(item["final_score"]),
            "source": "gpu_metrics",
        }
        for item in long_candidates[:3]
    ]
    fine_alternatives = [
        {**item, "source": "cuda_events"}
        for item in json.loads(fine["fine_event_candidates_json"])[:3]
    ]
    return {
        "workload_family": profile["family"],
        "device_id": profile["device_id"],
        "workload_id": profile["workload_id"],
        "primary_period_ms": primary_period,
        "primary_confidence": primary_confidence,
        "selected_branch": "cuda_events" if use_fine else "gpu_metrics",
        "long_metric_period_ms": float(long_detection["detected_period_ms"]),
        "long_metric_confidence": float(long_detection["confidence"]),
        "long_metric_emission_horizon_seconds": float(long_detection["emission_horizon_seconds"]),
        "alternative_periods_json": json.dumps(fine_alternatives + long_alternatives),
        "timeseries_path": str(profile["timeseries"]),
        "sqlite_path": str(profile["sqlite"]),
        **fine,
    }


def attach_audit(family: str, detections: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if family == "yolo":
        oracle = pd.read_csv(
            DEFAULT_OUTPUT_DIR / "model_experiments" / "pre6g_multiscale_detector_part_c2" / "yolo_deployment_detections.csv"
        )[["device_id", "workload_id", "oracle_period_ms"]]
        result = detections.merge(oracle, on=["device_id", "workload_id"], validate="one_to_one")
        label = "oracle_period_ms"
    else:
        stem = {
            "gemma": "pre6g_gemma4_5090_to_4090",
            "dnn": "pre6g_dnn_5090_to_4090",
            "vit": "pre6g_vit_5090_to_4090",
        }[family]
        name = "gemma_manifest.csv" if family == "gemma" else "manifest.csv"
        manifest = pd.read_csv(DEFAULT_OUTPUT_DIR / stem / name)
        result = detections.merge(
            manifest[["workload_id", "source_baseline_median_ms", "target_baseline_median_ms", "runtime_regression_eligible"]],
            on="workload_id",
            validate="one_to_one",
        )
        label = "source_baseline_median_ms"
    result["audit_gap_percent"] = (
        (result.primary_period_ms - result[label]).abs() / result[label] * 100
    )
    return result, label


def main() -> None:
    parser = argparse.ArgumentParser(description="Generalized dual-resolution marker-free detector.")
    parser.add_argument("--family", choices=("yolo", "gemma", "dnn", "vit"), required=True)
    args = parser.parse_args()
    output = DEFAULT_OUTPUT_DIR / "model_experiments" / "pre6g_generalized_detector_part_c3"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    profiles = discover_profiles(args.family)
    for index, profile in enumerate(profiles, start=1):
        print(f"[{index:02d}/{len(profiles):02d}] {profile['device_id']}/{profile['workload_id']}")
        rows.append(detect_one(profile))
    blind = pd.DataFrame(rows)
    blind_path = output / f"{args.family}_blind_detections.csv"
    blind.to_csv(blind_path, index=False)
    audited, label = attach_audit(args.family, blind)
    audited.to_csv(output / f"{args.family}_audited_detections.csv", index=False)
    contract = {
        "program": Path(__file__).name,
        "code_sha256": code_sha256(),
        "family": args.family,
        "detector_inputs": ["five GPU metric channels", "CUDA kernel start timestamp", "CUDA kernel short-name id"],
        "forbidden_detector_inputs": ["NVTX", "callbacks", "baseline runtime", "target runtime", "model/workload metadata"],
        "event_bin_ms": EVENT_BIN_MS,
        "event_period_range_ms": [EVENT_MIN_PERIOD_MS, EVENT_MAX_PERIOD_MS],
        "event_route_confidence": EVENT_ROUTE_CONFIDENCE,
        "event_route_prominence": EVENT_ROUTE_PROMINENCE,
        "audit_label": label,
        "audit_joined_after_blind_output": True,
    }
    (output / f"{args.family}_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    evaluated = audited[audited.audit_gap_percent.notna()]
    metrics = {
        "family": args.family,
        "profiles": len(audited),
        "evaluated_profiles": len(evaluated),
        "cuda_event_branch": int((audited.selected_branch == "cuda_events").sum()),
        "gpu_metric_branch": int((audited.selected_branch == "gpu_metrics").sum()),
        "mean_audit_gap_percent": float(evaluated.audit_gap_percent.mean()),
        "median_audit_gap_percent": float(evaluated.audit_gap_percent.median()),
        "p90_audit_gap_percent": float(evaluated.audit_gap_percent.quantile(0.9)),
        "max_audit_gap_percent": float(evaluated.audit_gap_percent.max()),
        "within_10_percent": float((evaluated.audit_gap_percent <= 10).mean()),
    }
    (output / f"{args.family}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
