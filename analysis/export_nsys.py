from __future__ import annotations

import argparse
import csv
import sqlite3
import subprocess
from pathlib import Path

from common import DEFAULT_CACHE_DIR, discover_runs_roots, iter_run_directories, relative_path, write_csv


DEFAULT_NSYS = DEFAULT_CACHE_DIR / "nsight_systems_2026_4_1" / "target-windows-x64" / "nsys.exe"
FIELDS = ["device_id", "workload_id", "mode", "report", "sqlite", "status", "return_code", "message"]


def valid_sqlite(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 100:
        return False
    try:
        connection = sqlite3.connect(path)
        try:
            return connection.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
            ).fetchone()[0] > 0
        finally:
            connection.close()
    except sqlite3.DatabaseError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministically export Nsight reports to SQLite.")
    parser.add_argument("--runs-root", action="append", type=Path)
    parser.add_argument("--mode", action="append", choices=["nsys_trace", "nsys_gpu_metrics", "nsys_cpu_metrics"])
    parser.add_argument("--nsys", type=Path, default=DEFAULT_NSYS)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR / "nsys")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    modes = set(args.mode or ["nsys_trace", "nsys_gpu_metrics", "nsys_cpu_metrics"])
    roots = [p.resolve() for p in args.runs_root] if args.runs_root else discover_runs_roots()
    nsys = args.nsys.resolve()
    if not nsys.exists(): raise SystemExit(f"nsys CLI not found: {nsys}")
    manifest = []
    for item in iter_run_directories(roots):
        mode = item["run_type"]
        if mode not in modes: continue
        report = item["run_directory"] / "profile.nsys-rep"
        output = args.cache_dir.resolve() / item["device_id"] / item["workload_id"] / mode / "profile.sqlite"
        output.parent.mkdir(parents=True, exist_ok=True)
        if valid_sqlite(output) and not args.force:
            status, code, message = "cached", 0, ""
        elif not report.exists():
            status, code, message = "missing_report", None, "profile.nsys-rep not found"
        else:
            command = [str(nsys), "export", "--type", "sqlite", "--force-overwrite=true", "--output", str(output), str(report)]
            completed = subprocess.run(command, text=True, capture_output=True)
            status, code = ("exported" if completed.returncode == 0 else "failed"), completed.returncode
            message = (completed.stderr or completed.stdout).strip().replace("\n", " | ")
        manifest.append({"device_id": item["device_id"], "workload_id": item["workload_id"], "mode": mode, "report": relative_path(report), "sqlite": relative_path(output), "status": status, "return_code": code, "message": message})
        print(f"{item['device_id']}/{item['workload_id']}/{mode}: {status}", flush=True)
    write_csv(args.cache_dir.resolve() / "export_manifest.csv", manifest, FIELDS)


if __name__ == "__main__":
    main()
