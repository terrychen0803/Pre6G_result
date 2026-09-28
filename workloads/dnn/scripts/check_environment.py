from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STACK_FILE = PROJECT_ROOT / "configs" / "software_stack.json"


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or result.stderr.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the workload collection environment")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    expected = json.loads(STACK_FILE.read_text(encoding="utf-8-sig"))
    actual_python = ".".join(platform.python_version().split(".")[:2])
    packages: dict[str, str | None] = {}
    package_ok = True
    for package, expected_version in expected["expected_versions"].items():
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "unknown")
        except Exception:
            version = None
        packages[package] = version
        if version is None or (expected_version is not None and version != expected_version):
            package_ok = False

    cuda_available = False
    gpu: dict[str, object] = {}
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            properties = torch.cuda.get_device_properties(0)
            gpu = {
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "multiprocessor_count": properties.multi_processor_count,
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "torch_cuda_version": torch.version.cuda,
            }
    except Exception:
        pass

    record = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_target_match": actual_python == expected["python_major_minor"],
        "packages": packages,
        "cuda_available": cuda_available,
        "gpu": gpu,
        "nvidia_smi_query": command_output([
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,power.limit,clocks.max.sm,clocks.max.memory,memory.total",
            "--format=csv,noheader,nounits",
        ]),
        "nsys_version": command_output(["nsys", "--version"]),
    }
    print(json.dumps(record, indent=2))
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    passed = record["python_target_match"] and package_ok and (cuda_available if args.gpu else True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
