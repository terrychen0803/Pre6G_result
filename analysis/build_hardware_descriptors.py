from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HARDWARE_ROOT = PROJECT_ROOT / "hardware"


DESCRIPTOR_FIELDS = [
    "device_id",
    "gpu_name",
    "gpu_architecture",
    "gpu_compute_capability",
    "gpu_sm_count",
    "gpu_cuda_core_count",
    "gpu_tensor_core_count",
    "gpu_vram_visible_bytes",
    "gpu_vram_visible_gib",
    "gpu_fb_memory_total_mib",
    "gpu_memory_bus_width_bits",
    "gpu_memory_bandwidth_gbps",
    "gpu_configured_power_limit_w",
    "gpu_default_power_limit_w",
    "gpu_max_power_limit_w",
    "gpu_max_graphics_clock_mhz",
    "gpu_max_sm_clock_mhz",
    "gpu_max_memory_clock_mhz",
    "pcie_device_max_generation",
    "pcie_host_max_generation",
    "pcie_negotiated_generation",
    "pcie_max_link_width",
    "pcie_negotiated_link_width",
    "cpu_architecture",
    "cpu_vendor",
    "cpu_model",
    "cpu_logical_count",
    "cpu_physical_core_count",
    "cpu_threads_per_core",
    "cpu_socket_count",
    "cpu_max_mhz",
    "cpu_l1d_cache_bytes",
    "cpu_l1i_cache_bytes",
    "cpu_l2_cache_bytes",
    "cpu_l3_cache_bytes",
    "system_memory_total_bytes",
    "env_os_kernel",
    "env_driver_version",
    "env_driver_cuda_version",
    "env_cuda_toolkit_release",
    "env_nvcc_version",
    "env_python_version",
    "env_pytorch_version",
    "env_pytorch_cuda_version",
    "env_ultralytics_version",
]


OBSERVATION_FIELDS = [
    "device_id",
    "observation_timestamp",
    "gpu_performance_state",
    "gpu_utilization_percent",
    "gpu_memory_utilization_percent",
    "gpu_temperature_c",
    "gpu_average_power_draw_w",
    "gpu_instantaneous_power_draw_w",
    "gpu_graphics_clock_mhz",
    "gpu_sm_clock_mhz",
    "gpu_memory_clock_mhz",
    "gpu_fb_memory_used_mib",
    "gpu_fb_memory_free_mib",
]


KNOWN_UNAVAILABLE_PREDICTORS = [
    "gpu_cuda_core_count",
    "gpu_tensor_core_count",
    "gpu_memory_bus_width_bits",
    "gpu_memory_bandwidth_gbps",
]


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8-sig", errors="replace")


def parse_number(value: str | None, cast: type = float) -> Any:
    if value is None:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value.replace(",", ""))
    if not match:
        return None
    return cast(match.group(0))


def parse_size_bytes(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"([\d.]+)\s*([KMGT]i?B)", value, re.IGNORECASE)
    if not match:
        return None
    scale = {
        "kb": 1000,
        "kib": 1024,
        "mb": 1000**2,
        "mib": 1024**2,
        "gb": 1000**3,
        "gib": 1024**3,
        "tb": 1000**4,
        "tib": 1024**4,
    }[match.group(2).lower()]
    return round(float(match.group(1)) * scale)


def parse_key_value_text(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip()
    return result


def parse_indented_nvidia_smi(text: str) -> dict[str, str]:
    """Convert the hierarchical ``nvidia-smi -q`` text to dotted keys."""
    result: dict[str, str] = {}
    sections: list[tuple[int, str]] = []

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("="):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()

        if re.match(
            r"^GPU\s+[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-9A-Fa-f]$",
            stripped,
        ):
            sections.clear()
            continue

        while sections and sections[-1][0] >= indent:
            sections.pop()

        if ":" not in stripped:
            if indent > 0:
                sections.append((indent, stripped))
            continue

        key, value = (part.strip() for part in stripped.split(":", 1))
        dotted_key = ".".join([item[1] for item in sections] + [key])
        result[dotted_key] = value

    return result


def pick(mapping: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def parse_stack(text: str) -> dict[str, str]:
    return parse_key_value_text(text)


def parse_summary_csv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        row = next(csv.DictReader(handle), None)
    return {key.strip(): value.strip() for key, value in (row or {}).items()}


def parse_free_memory(text: str) -> int | None:
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0] == "Mem:" and len(parts) >= 2:
            return int(parts[1])
    return None


def parse_device(device_dir: Path, hardware_root: Path) -> tuple[dict, dict, dict]:
    raw = device_dir / "raw"
    lscpu = parse_key_value_text(read_text(raw / "lscpu.txt"))
    nvsmi = parse_indented_nvidia_smi(read_text(raw / "nvidia_smi_full.txt"))
    summary = parse_summary_csv(raw / "nvidia_smi_summary.csv")
    stack = parse_stack(read_text(raw / "python_stack.txt"))

    torch_properties_path = raw / "torch_device_properties.json"
    torch_properties = (
        json.loads(read_text(torch_properties_path))
        if torch_properties_path.exists()
        else {}
    )

    nvcc_text = read_text(raw / "nvcc.txt")
    nvcc_release = re.search(r"release\s+([\d.]+)", nvcc_text)
    nvcc_version = re.search(r"\bV([\d.]+)", nvcc_text)

    sockets = parse_number(lscpu.get("Socket(s)"), int)
    cores_per_socket = parse_number(lscpu.get("Core(s) per socket"), int)
    physical_cores = (
        sockets * cores_per_socket
        if sockets is not None and cores_per_socket is not None
        else None
    )

    descriptor = {
        "device_id": device_dir.name,
        "gpu_name": torch_properties.get("gpu_name") or summary.get("name"),
        "gpu_architecture": pick(nvsmi, "Product Architecture"),
        "gpu_compute_capability": torch_properties.get("compute_capability"),
        "gpu_sm_count": torch_properties.get("sm_count"),
        "gpu_cuda_core_count": None,
        "gpu_tensor_core_count": None,
        "gpu_vram_visible_bytes": torch_properties.get("cuda_visible_memory_bytes"),
        "gpu_vram_visible_gib": torch_properties.get("cuda_visible_memory_gib"),
        "gpu_fb_memory_total_mib": parse_number(summary.get("memory.total [MiB]"), int),
        "gpu_memory_bus_width_bits": None,
        "gpu_memory_bandwidth_gbps": None,
        "gpu_configured_power_limit_w": parse_number(
            pick(nvsmi, "GPU Power Readings.Current Power Limit")
            or summary.get("power.limit [W]")
        ),
        "gpu_default_power_limit_w": parse_number(
            pick(nvsmi, "GPU Power Readings.Default Power Limit")
        ),
        "gpu_max_power_limit_w": parse_number(
            pick(nvsmi, "GPU Power Readings.Max Power Limit")
        ),
        "gpu_max_graphics_clock_mhz": parse_number(
            pick(nvsmi, "Max Clocks.Graphics"), int
        ),
        "gpu_max_sm_clock_mhz": parse_number(pick(nvsmi, "Max Clocks.SM"), int),
        "gpu_max_memory_clock_mhz": parse_number(
            pick(nvsmi, "Max Clocks.Memory"), int
        ),
        "pcie_device_max_generation": parse_number(
            pick(nvsmi, "PCI.GPU Link Info.PCIe Generation.Device Max"), int
        ),
        "pcie_host_max_generation": parse_number(
            pick(nvsmi, "PCI.GPU Link Info.PCIe Generation.Host Max"), int
        ),
        "pcie_negotiated_generation": parse_number(
            pick(
                nvsmi,
                "PCI.GPU Link Info.PCIe Generation.Device Current",
                "PCI.GPU Link Info.PCIe Generation.Current",
            ),
            int,
        ),
        "pcie_max_link_width": parse_number(
            pick(nvsmi, "PCI.GPU Link Info.Link Width.Max"), int
        ),
        "pcie_negotiated_link_width": parse_number(
            pick(nvsmi, "PCI.GPU Link Info.Link Width.Current"), int
        ),
        "cpu_architecture": lscpu.get("Architecture"),
        "cpu_vendor": lscpu.get("Vendor ID"),
        "cpu_model": lscpu.get("Model name"),
        "cpu_logical_count": parse_number(lscpu.get("CPU(s)"), int),
        "cpu_physical_core_count": physical_cores,
        "cpu_threads_per_core": parse_number(lscpu.get("Thread(s) per core"), int),
        "cpu_socket_count": sockets,
        "cpu_max_mhz": parse_number(lscpu.get("CPU max MHz")),
        "cpu_l1d_cache_bytes": parse_size_bytes(lscpu.get("L1d cache")),
        "cpu_l1i_cache_bytes": parse_size_bytes(lscpu.get("L1i cache")),
        "cpu_l2_cache_bytes": parse_size_bytes(lscpu.get("L2 cache")),
        "cpu_l3_cache_bytes": parse_size_bytes(lscpu.get("L3 cache")),
        "system_memory_total_bytes": parse_free_memory(read_text(raw / "memory.txt")),
        "env_os_kernel": read_text(raw / "uname.txt").strip() or None,
        "env_driver_version": summary.get("driver_version"),
        "env_driver_cuda_version": pick(nvsmi, "CUDA Version"),
        "env_cuda_toolkit_release": nvcc_release.group(1) if nvcc_release else None,
        "env_nvcc_version": nvcc_version.group(1) if nvcc_version else None,
        "env_python_version": stack.get("Python"),
        "env_pytorch_version": stack.get("PyTorch"),
        "env_pytorch_cuda_version": stack.get("PyTorch CUDA"),
        "env_ultralytics_version": stack.get("Ultralytics"),
    }

    observation = {
        "device_id": device_dir.name,
        "observation_timestamp": pick(nvsmi, "Timestamp"),
        "gpu_performance_state": pick(nvsmi, "Performance State"),
        "gpu_utilization_percent": parse_number(pick(nvsmi, "Utilization.GPU")),
        "gpu_memory_utilization_percent": parse_number(
            pick(nvsmi, "Utilization.Memory")
        ),
        "gpu_temperature_c": parse_number(pick(nvsmi, "Temperature.GPU Current Temp")),
        "gpu_average_power_draw_w": parse_number(
            pick(nvsmi, "GPU Power Readings.Average Power Draw")
        ),
        "gpu_instantaneous_power_draw_w": parse_number(
            pick(nvsmi, "GPU Power Readings.Instantaneous Power Draw")
        ),
        "gpu_graphics_clock_mhz": parse_number(pick(nvsmi, "Clocks.Graphics"), int),
        "gpu_sm_clock_mhz": parse_number(pick(nvsmi, "Clocks.SM"), int),
        "gpu_memory_clock_mhz": parse_number(pick(nvsmi, "Clocks.Memory"), int),
        "gpu_fb_memory_used_mib": parse_number(
            pick(nvsmi, "FB Memory Usage.Used"), int
        ),
        "gpu_fb_memory_free_mib": parse_number(
            pick(nvsmi, "FB Memory Usage.Free"), int
        ),
    }

    metadata = {
        "source_directory": raw.relative_to(hardware_root).as_posix(),
        "missing_predictor_fields": [
            field for field in KNOWN_UNAVAILABLE_PREDICTORS if descriptor[field] is None
        ],
    }
    return descriptor, observation, metadata


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize raw GPU-node hardware descriptions."
    )
    parser.add_argument("--hardware-root", type=Path, default=DEFAULT_HARDWARE_ROOT)
    args = parser.parse_args()

    hardware_root = args.hardware_root.resolve()
    device_dirs = sorted(
        path for path in hardware_root.iterdir() if path.is_dir() and (path / "raw").is_dir()
    )
    if not device_dirs:
        raise SystemExit(f"No <device>/raw directories found under {hardware_root}")

    descriptors: list[dict] = []
    observations: list[dict] = []
    metadata: list[dict] = []
    for device_dir in device_dirs:
        descriptor, observation, device_metadata = parse_device(device_dir, hardware_root)
        descriptors.append(descriptor)
        observations.append(observation)
        metadata.append(device_metadata)

    write_csv(hardware_root / "hardware_descriptors.csv", descriptors, DESCRIPTOR_FIELDS)
    write_csv(hardware_root / "collection_observations.csv", observations, OBSERVATION_FIELDS)

    structured = {
        "schema_version": 1,
        "devices": [
            {
                "device_id": descriptor["device_id"],
                "descriptor": descriptor,
                "collection_observation": observation,
                "provenance": device_metadata,
            }
            for descriptor, observation, device_metadata in zip(
                descriptors, observations, metadata, strict=True
            )
        ],
    }
    (hardware_root / "hardware_descriptors.json").write_text(
        json.dumps(structured, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Normalized {len(descriptors)} devices")
    print(hardware_root / "hardware_descriptors.csv")
    print(hardware_root / "collection_observations.csv")
    print(hardware_root / "hardware_descriptors.json")


if __name__ == "__main__":
    main()
