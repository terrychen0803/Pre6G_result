from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from common import DEFAULT_EXPERIMENTS, DEFAULT_RUNS_ROOT, PROJECT_ROOT, load_workloads, parse_workload_ids, print_command, run_with_log, sanitize_name

TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_vit.py"
CHECK_ENV_SCRIPT = PROJECT_ROOT / "scripts" / "check_environment.py"


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect low-instrumentation baseline ground truth")
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workloads", default="all")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--experiments", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")

    experiments = args.experiments.resolve()
    runs_root = args.runs_root.resolve()
    workload_ids = parse_workload_ids(args.workloads, set(load_workloads(experiments)))
    device_label = sanitize_name(args.device_id)
    environment_command = [sys.executable, str(CHECK_ENV_SCRIPT), "--gpu", "--json-output", str(runs_root / device_label / "environment.json")]
    print_command(environment_command)
    if not args.dry_run and subprocess.run(environment_command, check=False).returncode != 0:
        raise SystemExit("GPU environment check failed")

    for workload_id in workload_ids:
        for repeat in range(1, args.repeats + 1):
            run_dir = runs_root / device_label / workload_id / f"baseline_{repeat:02d}"
            if (run_dir / "summary.json").exists() and not args.force:
                print(f"[SKIP] {workload_id} repeat={repeat}")
                continue
            if run_dir.exists() and args.force and not args.dry_run:
                shutil.rmtree(run_dir)
            command = [
                sys.executable, str(TRAIN_SCRIPT), "--workload-id", workload_id,
                "--device-id", device_label, "--device", args.device,
                "--run-type", "baseline", "--repeat", str(repeat),
                "--experiments", str(experiments), "--runs-root", str(runs_root),
            ]
            print(f"[BASELINE] {workload_id} repeat={repeat}")
            print_command(command)
            if args.dry_run:
                continue
            if run_with_log(command, run_dir / "run.log", PROJECT_ROOT) != 0:
                raise SystemExit("Baseline process failed")
            if not (run_dir / "summary.json").exists():
                raise RuntimeError(f"Missing summary: {run_dir}")
    print("Baseline collection complete")


if __name__ == "__main__":
    main()
