from __future__ import annotations

import argparse
import re
from pathlib import Path

from common import DEFAULT_OUTPUT_DIR, PROJECT_ROOT, relative_path, write_csv


DB_MAGIC = {
    "meta.db": b"HPCTOOLKITmeta\x04\x00",
    "profile.db": b"HPCTOOLKITprof\x04\x00",
    "cct.db": b"HPCTOOLKITctxt\x04\x00",
}
FIELDS = [
    "device_id", "workload_id", "hpct_postprocess_available",
    "hpct_database_path", "hpct_database_format_version",
    "hpct_meta_db_valid", "hpct_meta_db_bytes",
    "hpct_profile_db_valid", "hpct_profile_db_bytes",
    "hpct_cct_db_valid", "hpct_cct_db_bytes",
    "hpct_trace_db_exists", "hpct_trace_db_bytes",
    "hpct_metric_taxonomy_exists", "hpct_metric_count",
    "hpct_hpcstruct_log_exists", "hpct_hpcstruct_error_count",
    "hpct_hpcstruct_gpu_begin_count", "hpct_hpcstruct_gpu_end_count",
    "hpct_hpcprof_log_exists", "hpct_hpcprof_error_count",
    "hpct_source_tree_exists", "hpct_original_hpcrun_log_exists",
    "hpct_hpcrun_missing_correlation_count",
    "hpct_accelerator_trace_records", "hpct_accelerator_trace_processed",
    "hpct_accelerator_trace_dropped", "hpct_accelerator_samples_recorded",
    "hpct_warning",
]
CATALOG_FIELDS = ["device_id", "metric_name", "metric_family", "unit_hint"]


def valid_magic(path: Path, expected: bytes) -> bool:
    if not path.is_file() or path.stat().st_size <= len(expected):
        return False
    try:
        with path.open("rb") as handle:
            return handle.read(len(expected)) == expected
    except OSError:
        return False


def error_count(text: str) -> int:
    return len(re.findall(r"(?im)\b(error|failed|fatal)\b", text))


def metric_names(path: Path) -> list[str]:
    if not path.is_file():
        return []
    values = re.findall(r"(?m)^\s+metric:\s+(.+?)\s*$", path.read_text(encoding="utf-8", errors="replace"))
    return sorted(set(value.strip() for value in values))


def unit_hint(name: str) -> str:
    match = re.search(r"\(([^)]+)\)\s*$", name)
    return match.group(1) if match else "count_or_accumulator"


def first_int_match(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate postprocessed HPCToolkit v4 databases.")
    parser.add_argument("--root", action="append", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    roots = [path.resolve() for path in args.root] if args.root else sorted(PROJECT_ROOT.glob("RTX*_hpctoolkit_postprocess"))
    run_roots = [root / "runs" for root in roots if (root / "runs").is_dir()]
    if not run_roots:
        raise SystemExit("No HPCToolkit postprocess runs roots found.")
    rows, catalog = [], []
    device_dirs = [path for run_root in run_roots for path in run_root.iterdir() if path.is_dir()]
    for device_dir in sorted(device_dirs):
        catalog_seen: set[str] = set()
        for workload_dir in sorted(path for path in device_dir.iterdir() if path.is_dir()):
            run_dir = workload_dir / "hpctoolkit_cuda_01"
            database = run_dir / "hpctoolkit-database"
            struct_log, prof_log = run_dir / "hpcstruct.log", run_dir / "hpcprof.log"
            struct_text = struct_log.read_text(encoding="utf-8", errors="replace") if struct_log.is_file() else ""
            prof_text = prof_log.read_text(encoding="utf-8", errors="replace") if prof_log.is_file() else ""
            metrics_path = database / "metrics" / "default.yaml"
            metrics = metric_names(metrics_path)
            original_candidates = sorted(PROJECT_ROOT.glob(f"*_collection_*/runs/{device_dir.name}/{workload_dir.name}/hpctoolkit_cuda_01"))
            original = original_candidates[0] if original_candidates else None
            hpcrun_log = original / "hpcrun.log" if original else Path()
            hpcrun_text = hpcrun_log.read_text(encoding="utf-8", errors="replace") if original and hpcrun_log.is_file() else ""
            measurement_text = ""
            if original:
                for log_path in (original / "hpctoolkit-measurements").glob("*.log"):
                    measurement_text += log_path.read_text(encoding="utf-8", errors="replace") + "\n"
            db_valid = {name: valid_magic(database / name, magic) for name, magic in DB_MAGIC.items()}
            warnings = []
            if not all(db_valid.values()): warnings.append("one or more required HPCToolkit v4 databases are invalid")
            if error_count(struct_text): warnings.append("hpcstruct log contains error/failure text")
            if error_count(prof_text): warnings.append("hpcprof log contains error/failure text")
            if not metrics: warnings.append("metric taxonomy missing or empty")
            row = {
                "device_id": device_dir.name, "workload_id": workload_dir.name,
                "hpct_postprocess_available": all(db_valid.values()) and bool(metrics),
                "hpct_database_path": relative_path(database), "hpct_database_format_version": "4.0",
                "hpct_trace_db_exists": (database / "trace.db").is_file(),
                "hpct_trace_db_bytes": (database / "trace.db").stat().st_size if (database / "trace.db").is_file() else None,
                "hpct_metric_taxonomy_exists": metrics_path.is_file(), "hpct_metric_count": len(metrics),
                "hpct_hpcstruct_log_exists": struct_log.is_file(), "hpct_hpcstruct_error_count": error_count(struct_text),
                "hpct_hpcstruct_gpu_begin_count": len(re.findall(r"begin parallel .*analysis of GPU binary", struct_text)),
                "hpct_hpcstruct_gpu_end_count": len(re.findall(r"end parallel .*analysis of GPU binary", struct_text)),
                "hpct_hpcprof_log_exists": prof_log.is_file(), "hpct_hpcprof_error_count": error_count(prof_text),
                "hpct_source_tree_exists": (database / "src").is_dir(),
                "hpct_original_hpcrun_log_exists": bool(original and hpcrun_log.is_file()),
                "hpct_hpcrun_missing_correlation_count": hpcrun_text.count("missing correlation id"),
                "hpct_accelerator_trace_records": first_int_match(r"accelerator trace records:\s*(\d+)", measurement_text),
                "hpct_accelerator_trace_processed": first_int_match(r"accelerator trace records:.*?processed:\s*(\d+)", measurement_text),
                "hpct_accelerator_trace_dropped": first_int_match(r"accelerator trace records:.*?dropped:\s*(\d+)", measurement_text),
                "hpct_accelerator_samples_recorded": first_int_match(r"accelerator samples:.*?recorded:\s*(\d+)", measurement_text),
                "hpct_warning": "; ".join(warnings),
            }
            for name in DB_MAGIC:
                key, path = name.removesuffix(".db"), database / name
                row[f"hpct_{key}_db_valid"] = db_valid[name]
                row[f"hpct_{key}_db_bytes"] = path.stat().st_size if path.is_file() else None
            rows.append(row)
            for name in metrics:
                if name not in catalog_seen:
                    catalog.append({"device_id": device_dir.name, "metric_name": name, "metric_family": name.split(":", 1)[0].split(" ", 1)[0], "unit_hint": unit_hint(name)})
                    catalog_seen.add(name)
    out = args.output_dir.resolve()
    write_csv(out / "hpctoolkit_postprocess_inventory.csv", rows, FIELDS)
    write_csv(out / "hpctoolkit_metric_catalog.csv", catalog, CATALOG_FIELDS)
    print(f"Validated {len(rows)} postprocessed HPCToolkit databases; cataloged {len(catalog)} device/metric rows.")


if __name__ == "__main__":
    main()
