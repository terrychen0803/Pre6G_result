from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import sqlite3
import statistics
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_ROOT = (
    PROJECT_ROOT / "analysis" / "high_load_01" / "cache" / "nsys_trace_events" / "RTX5090"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis" / "high_load_01" / "trace_event_period"
HORIZONS_SECONDS = (5, 7, 9, 12, 15, 20, 30)
TAIL_FRACTION = 0.60
MIN_OCCURRENCES = 8
MAX_OCCURRENCES = 800
MIN_PERIOD_MS = 20.0
MAX_PERIOD_MS = 2000.0
CLUSTER_TOLERANCE = 0.08
MAX_CLUSTER_MEMBERS = 12
DEPLOYMENT_HORIZONS = (7, 9, 12, 15, 20, 30)
DEPLOYMENT_MIN_CONFIDENCE = 0.60
DEPLOYMENT_STABILITY_TOLERANCE = 0.10


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    fraction = position - low
    return float(ordered[low] * (1 - fraction) + ordered[high] * fraction)


def median_absolute_deviation(values: list[float]) -> float:
    center = median(values)
    return median([abs(value - center) for value in values])


def relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(left, right)


def load_trace(database: Path) -> dict:
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        required = {"CUPTI_ACTIVITY_KIND_KERNEL", "NVTX_EVENTS"}
        if not required <= tables:
            raise RuntimeError(f"missing required tables: {sorted(required - tables)}")
        events = [
            (int(start), int(short_name))
            for start, short_name in connection.execute(
                "SELECT start,shortName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
            )
        ]
        names = {
            int(identifier): str(value)
            for identifier, value in connection.execute("SELECT id,value FROM StringIds")
        }
        nvtx = [
            (str(text), int(start))
            for text, start in connection.execute(
                "SELECT text,start FROM NVTX_EVENTS "
                "WHERE text LIKE 'TRAIN_ITER_%' ORDER BY start"
            )
        ]
        pid_count, context_count = connection.execute(
            "SELECT COUNT(DISTINCT globalPid),COUNT(DISTINCT contextId) "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchone()
        diagnostic = "\n".join(
            str(row[0])
            for row in connection.execute("SELECT text FROM DIAGNOSTIC_EVENT")
            if row[0]
        ) if "DIAGNOSTIC_EVENT" in tables else ""
    finally:
        connection.close()
    if not events:
        raise RuntimeError("trace contains no CUDA kernel events")
    return {
        "events": events,
        "names": names,
        "nvtx": nvtx,
        "pid_count": int(pid_count),
        "context_count": int(context_count),
        "hardware_trace": "Hardware tracing used for CUDA tracing" in diagnostic,
        "nvtx_warning": "Not all NVTX events might have been collected" in diagnostic,
    }


def build_candidates(events: list[tuple[int, int]], horizon_seconds: int) -> tuple[list[dict], int, int]:
    capture_start = events[0][0]
    capture_end = capture_start + int(horizon_seconds * 1e9)
    occurrences: dict[int, list[int]] = collections.defaultdict(list)
    for timestamp, token in events:
        if timestamp > capture_end:
            break
        occurrences[token].append(timestamp)

    candidates = []
    for token, timestamps in occurrences.items():
        count = len(timestamps)
        if not MIN_OCCURRENCES <= count <= MAX_OCCURRENCES:
            continue
        gaps_ms = [
            (right - left) / 1e6
            for left, right in zip(timestamps, timestamps[1:])
        ]
        all_period = median(gaps_ms)
        tail_start = int(len(gaps_ms) * (1.0 - TAIL_FRACTION))
        tail_gaps = gaps_ms[tail_start:]
        period = median(tail_gaps)
        if not (MIN_PERIOD_MS <= all_period <= MAX_PERIOD_MS):
            continue
        if not (MIN_PERIOD_MS <= period <= MAX_PERIOD_MS):
            continue
        robust_cv = median_absolute_deviation(tail_gaps) / period
        ordinary_cv = (
            statistics.stdev(gaps_ms) / statistics.mean(gaps_ms)
            if len(gaps_ms) > 1
            else math.inf
        )
        coverage = (timestamps[-1] - timestamps[0]) / max(1, capture_end - capture_start)
        score = (
            math.log(count)
            * coverage
            / (1.0 + 2.0 * robust_cv + 0.5 * ordinary_cv)
        )
        candidates.append(
            {
                "token": token,
                "count": count,
                "period_ms": period,
                "all_period_ms": all_period,
                "robust_cv": robust_cv,
                "ordinary_cv": ordinary_cv,
                "coverage": coverage,
                "score": score,
            }
        )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates, capture_start, capture_end


def cluster_candidates(candidates: list[dict]) -> list[dict]:
    clusters = []
    used: set[int] = set()
    for seed in candidates:
        if seed["token"] in used:
            continue
        members = [
            item
            for item in candidates
            if item["token"] not in used
            and relative_difference(item["period_ms"], seed["period_ms"])
            <= CLUSTER_TOLERANCE
        ]
        used.update(item["token"] for item in members)
        members = sorted(members, key=lambda item: item["score"], reverse=True)[
            :MAX_CLUSTER_MEMBERS
        ]
        clusters.append(
            {
                "period_ms": median([item["period_ms"] for item in members]),
                "score": sum(item["score"] for item in members),
                "member_count": len(members),
                "leader": members[0],
                "members": members,
            }
        )
    return sorted(clusters, key=lambda item: item["score"], reverse=True)


def select_cluster(clusters: list[dict]) -> tuple[dict | None, bool]:
    if not clusters:
        return None, False
    selected = clusters[0]
    # Repeated data-loader/sort kernels often appear twice per training iteration.
    # Prefer a supported 2x-period cluster when the primary cluster is its harmonic.
    higher = [
        item
        for item in clusters[1:]
        if 1.7 <= item["period_ms"] / selected["period_ms"] <= 2.3
        and item["score"] >= 0.30 * selected["score"]
        and item["leader"]["robust_cv"] <= 0.60
    ]
    if higher:
        return max(higher, key=lambda item: item["score"]), True
    return selected, False


def detect(events: list[tuple[int, int]], names: dict[int, str], horizon_seconds: int) -> dict:
    candidates, capture_start, capture_end = build_candidates(events, horizon_seconds)
    clusters = cluster_candidates(candidates)
    selected, harmonic_corrected = select_cluster(clusters)
    if selected is None:
        return {
            "accepted": False,
            "rejection_reason": "no recurrent anchor candidates",
            "capture_start_ns": capture_start,
            "capture_end_ns": capture_end,
            "candidate_count": 0,
        }
    leader = selected["leader"]
    accepted = (
        selected["member_count"] >= 2
        and leader["coverage"] >= 0.25
        and leader["robust_cv"] <= 0.60
    )
    reasons = []
    if selected["member_count"] < 2:
        reasons.append("no anchor consensus")
    if leader["coverage"] < 0.25:
        reasons.append("insufficient temporal coverage")
    if leader["robust_cv"] > 0.60:
        reasons.append("unstable anchor intervals")
    confidence = min(
        1.0,
        0.35 * min(1.0, selected["member_count"] / 6.0)
        + 0.30 * min(1.0, leader["coverage"] / 0.75)
        + 0.35 * max(0.0, 1.0 - leader["robust_cv"] / 0.60),
    )
    return {
        "accepted": accepted,
        "rejection_reason": " | ".join(reasons),
        "detected_period_ms": selected["period_ms"],
        "confidence": confidence,
        "harmonic_corrected": harmonic_corrected,
        "anchor_name": names.get(leader["token"], str(leader["token"])),
        "anchor_occurrences": leader["count"],
        "anchor_coverage": leader["coverage"],
        "anchor_robust_cv": leader["robust_cv"],
        "cluster_member_count": selected["member_count"],
        "cluster_score": selected["score"],
        "candidate_count": len(candidates),
        "capture_start_ns": capture_start,
        "capture_end_ns": capture_end,
    }


def oracle_metrics(nvtx: list[tuple[str, int]], capture_end: int) -> dict:
    all_starts = [start for _, start in nvtx]
    steady_starts = [start for text, start in nvtx if "TRAIN_ITER_0021" <= text <= "TRAIN_ITER_0128"]
    full_gaps = [
        (right - left) / 1e6
        for left, right in zip(steady_starts, steady_starts[1:])
    ]
    prefix_starts = [start for start in all_starts if start <= capture_end]
    prefix_gaps = [
        (right - left) / 1e6
        for left, right in zip(prefix_starts, prefix_starts[1:])
    ]
    tail_start = int(len(prefix_gaps) * (1.0 - TAIL_FRACTION))
    prefix_tail = prefix_gaps[tail_start:]
    return {
        "observed_iteration_count": len(prefix_starts),
        "prefix_oracle_period_ms": median(prefix_tail) if prefix_tail else math.nan,
        "full_steady_oracle_period_ms": median(full_gaps) if full_gaps else math.nan,
        "full_steady_period_cv": (
            statistics.stdev(full_gaps) / statistics.mean(full_gaps)
            if len(full_gaps) > 1
            else math.nan
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    result = []
    for horizon in HORIZONS_SECONDS:
        horizon_rows = [row for row in rows if row["horizon_seconds"] == horizon]
        accepted = [row for row in horizon_rows if row["accepted"]]
        evaluated = [
            row
            for row in accepted
            if row["observed_iteration_count"] >= MIN_OCCURRENCES
            and math.isfinite(row["prefix_period_ape_percent"])
        ]
        prefix_errors = [row["prefix_period_ape_percent"] for row in evaluated]
        full_errors = [row["full_period_ape_percent"] for row in evaluated]
        result.append(
            {
                "horizon_seconds": horizon,
                "profiles": len(horizon_rows),
                "accepted": len(accepted),
                "coverage": len(accepted) / len(horizon_rows),
                "evaluated": len(evaluated),
                "prefix_mean_ape_percent": statistics.mean(prefix_errors) if prefix_errors else math.nan,
                "prefix_median_ape_percent": median(prefix_errors) if prefix_errors else math.nan,
                "prefix_p90_ape_percent": percentile(prefix_errors, 0.90),
                "prefix_max_ape_percent": max(prefix_errors) if prefix_errors else math.nan,
                "prefix_within_10_percent": (
                    sum(error <= 10.0 for error in prefix_errors) / len(prefix_errors)
                    if prefix_errors
                    else math.nan
                ),
                "full_mean_ape_percent": statistics.mean(full_errors) if full_errors else math.nan,
                "full_median_ape_percent": median(full_errors) if full_errors else math.nan,
                "full_p90_ape_percent": percentile(full_errors, 0.90),
                "full_max_ape_percent": max(full_errors) if full_errors else math.nan,
            }
        )
    return result


def select_adaptive(rows: list[dict]) -> tuple[list[dict], dict]:
    """Select detections without consulting NVTX or any other audit fields."""
    by_workload: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_workload[row["workload_id"]].append(row)
    selected_rows = []
    for workload_id, profile_rows in sorted(by_workload.items()):
        ordered = sorted(profile_rows, key=lambda item: item["horizon_seconds"])
        previous = None
        chosen = None
        for row in ordered:
            if row["horizon_seconds"] not in DEPLOYMENT_HORIZONS:
                continue
            if row["accepted"] and row["confidence"] >= DEPLOYMENT_MIN_CONFIDENCE:
                if previous is not None and relative_difference(
                    row["detected_period_ms"], previous["detected_period_ms"]
                ) <= DEPLOYMENT_STABILITY_TOLERANCE:
                    chosen = dict(row)
                    chosen["selection_reason"] = "two-window stability"
                    break
                previous = row
            else:
                previous = None
        if chosen is None:
            accepted = [row for row in ordered if row["accepted"]]
            chosen = dict(accepted[-1] if accepted else ordered[-1])
            chosen["selection_reason"] = "latest accepted fallback" if accepted else "rejected"
        selected_rows.append(chosen)

    evaluated = [
        row
        for row in selected_rows
        if row["accepted"] and math.isfinite(row["prefix_period_ape_percent"])
    ]
    prefix_errors = [row["prefix_period_ape_percent"] for row in evaluated]
    full_errors = [row["full_period_ape_percent"] for row in evaluated]
    horizons = [row["horizon_seconds"] for row in evaluated]
    metrics = {
        "profiles": len(selected_rows),
        "accepted": len(evaluated),
        "coverage": len(evaluated) / len(selected_rows),
        "mean_emission_horizon_seconds": statistics.mean(horizons),
        "median_emission_horizon_seconds": median(horizons),
        "prefix_mean_ape_percent": statistics.mean(prefix_errors),
        "prefix_median_ape_percent": median(prefix_errors),
        "prefix_p90_ape_percent": percentile(prefix_errors, 0.90),
        "prefix_max_ape_percent": max(prefix_errors),
        "prefix_within_10_percent": sum(error <= 10 for error in prefix_errors) / len(prefix_errors),
        "full_mean_ape_percent": statistics.mean(full_errors),
        "full_median_ape_percent": median(full_errors),
        "full_p90_ape_percent": percentile(full_errors, 0.90),
        "full_max_ape_percent": max(full_errors),
    }
    return selected_rows, metrics


def write_report(output: Path, metrics: list[dict], adaptive: dict, inventory: list[dict]) -> None:
    lines = [
        "# High-load trace-only marker-free period experiment",
        "",
        "The detector reads only CUDA kernel start timestamps and kernel short-name IDs. "
        "NVTX is joined after detection for audit only.",
        "",
        "| Horizon | Accepted | Evaluated | Prefix mean APE | Prefix median APE | Prefix P90 APE | Prefix <=10% | Full-run median APE |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            f"| {row['horizon_seconds']}s | {row['accepted']}/{row['profiles']} | "
            f"{row['evaluated']} | {row['prefix_mean_ape_percent']:.2f}% | "
            f"{row['prefix_median_ape_percent']:.2f}% | {row['prefix_p90_ape_percent']:.2f}% | "
            f"{row['prefix_within_10_percent']:.1%} | {row['full_median_ape_percent']:.2f}% |"
        )
    all_single_process = all(row["pid_count"] == 1 for row in inventory)
    all_single_context = all(row["context_count"] == 1 for row in inventory)
    all_hardware = all(row["hardware_trace"] for row in inventory)
    lines.extend(
        [
            "",
            "## Marker-blind deployment policy",
            "",
            f"- Coverage: {adaptive['accepted']}/{adaptive['profiles']} ({adaptive['coverage']:.1%})",
            f"- Mean emission horizon: {adaptive['mean_emission_horizon_seconds']:.2f}s",
            f"- Prefix mean / median / P90 / max APE: {adaptive['prefix_mean_ape_percent']:.2f}% / "
            f"{adaptive['prefix_median_ape_percent']:.2f}% / {adaptive['prefix_p90_ape_percent']:.2f}% / "
            f"{adaptive['prefix_max_ape_percent']:.2f}%",
            f"- Prefix within 10%: {adaptive['prefix_within_10_percent']:.1%}",
            f"- Full-run mean / median APE: {adaptive['full_mean_ape_percent']:.2f}% / "
            f"{adaptive['full_median_ape_percent']:.2f}%",
            "",
            "## Integrity",
            "",
            f"- Profiles checked: {len(inventory)}",
            f"- All profiles contain one CUDA process: {all_single_process}",
            f"- All profiles contain one CUDA context: {all_single_context}",
            f"- All profiles used CUDA hardware tracing: {all_hardware}",
            "",
            "## Interpretation",
            "",
            "Prefix APE measures whether the detector recovers the period actually visible inside the same profiling window. "
            "Full-run APE compares an early estimate with the later full steady-state median; under non-stationary background contention, "
            "this includes future load drift and is not purely detector error.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate marker-free CUDA-event period detection.")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    rows = []
    inventory = []
    databases = sorted(args.cache_root.glob("C??/nsys_trace/profile.sqlite"))
    if not databases:
        raise SystemExit(f"no trace SQLite databases found under {args.cache_root}")
    for index, database in enumerate(databases, start=1):
        workload_id = database.parts[-3]
        print(f"[{index:02d}/{len(databases):02d}] {workload_id}", flush=True)
        trace = load_trace(database)
        inventory.append(
            {
                "workload_id": workload_id,
                "database": str(database.resolve()),
                "kernel_count": len(trace["events"]),
                "nvtx_iteration_count": len(trace["nvtx"]),
                "pid_count": trace["pid_count"],
                "context_count": trace["context_count"],
                "hardware_trace": trace["hardware_trace"],
                "nvtx_warning": trace["nvtx_warning"],
            }
        )
        for horizon in HORIZONS_SECONDS:
            detection = detect(trace["events"], trace["names"], horizon)
            oracle = oracle_metrics(trace["nvtx"], detection["capture_end_ns"])
            row = {
                "device_id": "RTX5090",
                "workload_id": workload_id,
                "horizon_seconds": horizon,
                **detection,
                **oracle,
            }
            if detection["accepted"] and oracle["observed_iteration_count"] >= MIN_OCCURRENCES:
                row["prefix_period_ape_percent"] = (
                    abs(detection["detected_period_ms"] - oracle["prefix_oracle_period_ms"])
                    / oracle["prefix_oracle_period_ms"]
                    * 100.0
                )
                row["full_period_ape_percent"] = (
                    abs(detection["detected_period_ms"] - oracle["full_steady_oracle_period_ms"])
                    / oracle["full_steady_oracle_period_ms"]
                    * 100.0
                )
            else:
                row["prefix_period_ape_percent"] = math.nan
                row["full_period_ape_percent"] = math.nan
            rows.append(row)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics = summarize(rows)
    adaptive_rows, adaptive_metrics = select_adaptive(rows)
    write_csv(output / "detections.csv", rows)
    write_csv(output / "metrics.csv", metrics)
    write_csv(output / "adaptive_deployment.csv", adaptive_rows)
    write_csv(output / "trace_inventory.csv", inventory)
    contract = {
        "program": Path(__file__).name,
        "program_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "detector_inputs": ["CUDA kernel start timestamp", "CUDA kernel short-name ID"],
        "forbidden_detector_inputs": [
            "NVTX",
            "iterations.csv",
            "workload ID or configuration",
            "baseline runtime",
            "GPU metrics",
        ],
        "audit_inputs_joined_after_detection": ["NVTX TRAIN_ITER start timestamps"],
        "horizons_seconds": HORIZONS_SECONDS,
        "tail_fraction": TAIL_FRACTION,
        "harmonic_rule": "prefer supported 2x cluster when primary is a half-period harmonic",
        "deployment_policy": {
            "horizons_seconds": DEPLOYMENT_HORIZONS,
            "minimum_confidence": DEPLOYMENT_MIN_CONFIDENCE,
            "two_window_relative_tolerance": DEPLOYMENT_STABILITY_TOLERANCE,
            "selection_uses_audit_labels": False,
        },
    }
    (output / "experiment_contract.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )
    (output / "adaptive_metrics.json").write_text(
        json.dumps(adaptive_metrics, indent=2), encoding="utf-8"
    )
    write_report(output, metrics, adaptive_metrics, inventory)
    print(json.dumps({"fixed_horizons": metrics, "adaptive": adaptive_metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
