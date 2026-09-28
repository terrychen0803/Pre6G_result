from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from common import DEFAULT_EXPERIMENTS, DEFAULT_RUNS_ROOT, PROJECT_ROOT, load_workloads, parse_workload_ids, print_command, run_with_log, sanitize_name

TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_vit.py"


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect optional HPCToolkit GPU profiles")
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workloads", default="all")
    parser.add_argument("--profile-mode", choices=["cuda", "cuda-trace"], default="cuda")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--hpct-root", type=Path, default=None)
    parser.add_argument("--experiments", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    hpcrun = str(args.hpct_root.resolve() / "bin" / "hpcrun") if args.hpct_root else shutil.which("hpcrun")
    if hpcrun is None:
        if args.dry_run:
            hpcrun = "hpcrun"
        else:
            raise SystemExit("hpcrun was not found")
    experiments = args.experiments.resolve()
    runs_root = args.runs_root.resolve()
    workload_ids = parse_workload_ids(args.workloads, set(load_workloads(experiments)))
    device_label = sanitize_name(args.device_id)
    run_type = "hpctoolkit_cuda" if args.profile_mode == "cuda" else "hpctoolkit_cuda_trace"
    for workload_id in workload_ids:
        run_dir = runs_root / device_label / workload_id / f"{run_type}_{args.repeat:02d}"
        measurements = run_dir / "hpctoolkit-measurements"
        command = [hpcrun, "-e", "gpu=cuda"]
        if args.profile_mode == "cuda-trace":
            command.append("-t")
        command.extend([
            "-o", str(measurements), sys.executable, str(TRAIN_SCRIPT),
            "--workload-id", workload_id, "--device-id", device_label,
            "--device", args.device, "--run-type", run_type,
            "--repeat", str(args.repeat), "--experiments", str(experiments),
            "--runs-root", str(runs_root),
        ])
        print_command(command)
        if args.dry_run:
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "profiling_config.json").write_text(
            json.dumps({"profile_mode": args.profile_mode, "command": command}, indent=2), encoding="utf-8"
        )
        if run_with_log(command, run_dir / "hpcrun.log", PROJECT_ROOT) != 0:
            raise SystemExit("HPCToolkit collection failed")
    print("HPCToolkit collection complete")


if __name__ == "__main__":
    main()
