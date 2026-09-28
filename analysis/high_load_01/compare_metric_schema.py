from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLEAN = PROJECT_ROOT / "analysis_cache" / "nsys"
DEFAULT_HIGH = PROJECT_ROOT / "analysis" / "high_load_01" / "cache" / "nsys"
DEFAULT_OUTPUT = PROJECT_ROOT / "analysis" / "high_load_01" / "qc"
REQUIRED_CHANNELS = {
    "SM Issue [Throughput %]",
    "SMs Active [Throughput %]",
    "Tensor Active [Throughput %]",
    "DRAM Read Bandwidth [Throughput %]",
    "DRAM Write Bandwidth [Throughput %]",
}


def descriptors(path: Path) -> list[str]:
    connection = sqlite3.connect(path)
    try:
        return [
            str(row[0])
            for row in connection.execute(
                "select metricName from TARGET_INFO_GPU_METRICS order by metricId"
            )
        ]
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare clean/high Nsight GPU metric descriptors.")
    parser.add_argument("--clean-cache", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--high-cache", type=Path, default=DEFAULT_HIGH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for workload in [f"C{index:02d}" for index in range(1, 25)]:
        clean_path = args.clean_cache / "RTX5090" / workload / "nsys_gpu_metrics" / "profile.sqlite"
        high_path = args.high_cache / "RTX5090" / workload / "nsys_gpu_metrics" / "profile.sqlite"
        clean = descriptors(clean_path)
        high = descriptors(high_path)
        rows.append(
            {
                "workload_id": workload,
                "clean_metric_count": len(clean),
                "high_metric_count": len(high),
                "ordered_descriptors_match": clean == high,
                "descriptor_sets_match": set(clean) == set(high),
                "clean_missing_detector_channels": " | ".join(sorted(REQUIRED_CHANNELS - set(clean))),
                "high_missing_detector_channels": " | ".join(sorted(REQUIRED_CHANNELS - set(high))),
                "only_in_clean": " | ".join(sorted(set(clean) - set(high))),
                "only_in_high": " | ".join(sorted(set(high) - set(clean))),
            }
        )
    data = pd.DataFrame(rows)
    data.to_csv(output / "gpu_metric_schema_alignment.csv", index=False)
    lines = [
        "# GPU metric schema alignment",
        "",
        f"- Workloads checked: {len(data)}",
        f"- Exact ordered descriptor matches: {int(data.ordered_descriptors_match.sum())}/{len(data)}",
        f"- Descriptor-set matches: {int(data.descriptor_sets_match.sum())}/{len(data)}",
        f"- Clean profiles with all five detector channels: {int((data.clean_missing_detector_channels == '').sum())}/{len(data)}",
        f"- High-load profiles with all five detector channels: {int((data.high_missing_detector_channels == '').sum())}/{len(data)}",
        "",
        "The configuration-level `gb20x` versus automatic metric-set difference is harmless only if the exported descriptor schemas match, as checked above.",
        "",
    ]
    (output / "gpu_metric_schema_alignment.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote metric schema comparison to {output}")


if __name__ == "__main__":
    main()
