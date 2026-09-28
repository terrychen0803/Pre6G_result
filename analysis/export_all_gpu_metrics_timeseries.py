from __future__ import annotations

import argparse
import bisect
import sqlite3
import struct
import sys
import zipfile
from array import array
from pathlib import Path


def scale_for(name: str) -> float:
    """Match the unit conversion used by extract_nsys_gpu_metrics.py."""
    return 1.0 / 1_000_000.0 if "[MHz]" in name or name.endswith("[Avg]") else 1.0


def npy_header(descr: str, shape: tuple[int, ...]) -> bytes:
    shape_text = repr(shape)
    header = (
        "{'descr': "
        + repr(descr)
        + ", 'fortran_order': False, 'shape': "
        + shape_text
        + ", }"
    ).encode("latin1")
    padding = (-((6 + 2 + 2) + len(header) + 1)) % 64
    header += b" " * padding + b"\n"
    if len(header) > 65535:
        raise ValueError("NPY header is too large")
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header


def write_float32(zf: zipfile.ZipFile, name: str, values: array, shape: tuple[int, ...]) -> None:
    if sys.byteorder != "little":
        values.byteswap()
    with zf.open(name + ".npy", "w") as output:
        output.write(npy_header("<f4", shape))
        output.write(values.tobytes())


def write_unicode(zf: zipfile.ZipFile, name: str, values: list[str]) -> None:
    width = max(map(len, values), default=1)
    with zf.open(name + ".npy", "w") as output:
        output.write(npy_header(f"<U{width}", (len(values),)))
        for value in values:
            output.write(value.encode("utf-32le"))
            output.write(b"\x00" * ((width - len(value)) * 4))


def interpolate(timestamps: list[int], values: list[float], grid: range, factor: float) -> array:
    result = array("f")
    for timestamp in grid:
        right = bisect.bisect_left(timestamps, timestamp)
        if right == 0:
            value = values[0]
        elif right == len(timestamps):
            value = values[-1]
        elif timestamps[right] == timestamp:
            value = values[right]
        else:
            left = right - 1
            fraction = (timestamp - timestamps[left]) / (timestamps[right] - timestamps[left])
            value = values[left] + fraction * (values[right] - values[left])
        result.append(value * factor)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Export every Nsight GPU metric as a uniform time-series NPZ.")
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bin-ms", type=float, default=1.0)
    args = parser.parse_args()

    database = args.sqlite.resolve()
    output = args.output.resolve()
    step_ns = round(args.bin_ms * 1_000_000)
    if step_ns <= 0:
        raise ValueError("--bin-ms must be positive")

    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        descriptors = connection.execute(
            "SELECT metricId, metricName FROM TARGET_INFO_GPU_METRICS ORDER BY metricId"
        ).fetchall()
        if not descriptors:
            raise ValueError(f"No GPU metric descriptors in {database}")

        series: list[tuple[list[int], list[float]]] = []
        starts: list[int] = []
        ends: list[int] = []
        for metric_id, metric_name in descriptors:
            samples = connection.execute(
                "SELECT timestamp, value FROM GPU_METRICS WHERE metricId=? ORDER BY timestamp",
                (metric_id,),
            ).fetchall()
            if not samples:
                raise ValueError(f"Metric {metric_id} ({metric_name}) has no samples")
            timestamps = [int(row[0]) for row in samples]
            values = [float(row[1]) for row in samples]
            series.append((timestamps, values))
            starts.append(timestamps[0])
            ends.append(timestamps[-1])

        global_start = max(starts)
        global_end = min(ends)
        grid = range(global_start, global_end + 1, step_ns)
        sample_count = len(grid)
        if sample_count < 2:
            raise ValueError("Metrics have no usable common time range")

        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            times = array("f", ((timestamp - global_start) / 1_000_000.0 for timestamp in grid))
            write_float32(zf, "time_ms", times, (sample_count,))

            matrix = array("f")
            for (_, metric_name), (timestamps, values) in zip(descriptors, series):
                matrix.extend(interpolate(timestamps, values, grid, scale_for(str(metric_name))))
            write_float32(zf, "values", matrix, (len(descriptors), sample_count))
            write_unicode(zf, "channels", [str(row[1]) for row in descriptors])
            write_float32(zf, "metric_ids", array("f", (float(row[0]) for row in descriptors)), (len(descriptors),))
            write_float32(zf, "value_scales", array("f", (scale_for(str(row[1])) for row in descriptors)), (len(descriptors),))
            write_float32(zf, "base_bin_ms", array("f", [args.bin_ms]), (1,))

        print(f"Wrote {output}")
        print(f"channels={len(descriptors)} samples={sample_count} shape=({len(descriptors)}, {sample_count})")
        print(f"duration_ms={(sample_count - 1) * args.bin_ms:.3f}")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
