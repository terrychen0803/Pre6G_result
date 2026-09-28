from __future__ import annotations

import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "analysis_cache"
DEFAULT_EXPERIMENTS = PROJECT_ROOT / "configs" / "yolo26" / "experiments.csv"
DEFAULT_DATA_ROOT = Path(
    os.environ.get("PRE6G_DATA_ROOT", PROJECT_ROOT / "data" / "raw")
).expanduser()


RUN_PATTERN = re.compile(
    r"^(baseline|nsys_trace|nsys_gpu_metrics|nsys_cpu_metrics|"
    r"hpctoolkit_cuda_trace|hpctoolkit_cuda)_(\d+)$"
)


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return None


def parse_number(value: Any, cast: type = float) -> Any:
    if value is None or value == "":
        return None
    try:
        return cast(value)
    except (TypeError, ValueError):
        return None


def load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, None
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            return None, "JSON root is not an object"
        return value, None
    except Exception as exc:  # preserve bad runs instead of aborting the scan
        return None, f"{type(exc).__name__}: {exc}"


def first_nested_value(value: Any, keys: Iterable[str]) -> Any:
    wanted = {key.lower() for key in keys}
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in wanted and item not in (None, ""):
                return item
        for item in value.values():
            found = first_nested_value(item, wanted)
            if found not in (None, ""):
                return found
    elif isinstance(value, list):
        for item in value:
            found = first_nested_value(item, wanted)
            if found not in (None, ""):
                return found
    return None


def discover_runs_roots(project_root: Path = PROJECT_ROOT) -> list[Path]:
    candidates: list[Path] = []
    data_root = Path(
        os.environ.get("PRE6G_DATA_ROOT", project_root / "data" / "raw")
    ).expanduser()
    direct_candidates = [project_root / "runs", data_root / "runs"]
    for direct in direct_candidates:
        if direct.is_dir() and direct.resolve() not in candidates:
            candidates.append(direct.resolve())
    for path in sorted(project_root.glob("*_collection_*/runs")):
        resolved = path.resolve()
        if resolved not in candidates:
            candidates.append(resolved)
    if data_root.is_dir():
        for path in sorted(data_root.glob("*/runs")):
            resolved = path.resolve()
            if resolved not in candidates:
                candidates.append(resolved)
    return candidates


def iter_run_directories(runs_roots: Iterable[Path]):
    for runs_root in sorted(Path(path).resolve() for path in runs_roots):
        for device_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
            for workload_dir in sorted(path for path in device_dir.iterdir() if path.is_dir()):
                for run_dir in sorted(path for path in workload_dir.iterdir() if path.is_dir()):
                    match = RUN_PATTERN.match(run_dir.name)
                    if match:
                        yield {
                            "runs_root": runs_root,
                            "device_id": device_dir.name,
                            "workload_id": workload_dir.name,
                            "run_type": match.group(1),
                            "repeat": int(match.group(2)),
                            "run_directory": run_dir,
                        }


def load_experiments(path: Path = DEFAULT_EXPERIMENTS) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            result[row["workload_id"]] = {
                "workload_id": row["workload_id"],
                "model": row["model"],
                "batch": int(row["batch"]),
                "imgsz": int(row["imgsz"]),
                "amp": parse_bool(row["amp"]),
                "max_iters": int(row["max_iters"]),
                "warmup_iters": int(row["warmup_iters"]),
                "nbs": int(row["nbs"]),
            }
    return result


def relative_path(path: Path, project_root: Path = PROJECT_ROOT) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def format_value(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "missing"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)
