from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def has_table(connection: sqlite3.Connection, name: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def rows(connection: sqlite3.Connection, sql: str, params=()) -> list[dict]:
    cursor = connection.execute(sql, params)
    names = [item[0] for item in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def probe(path: Path) -> dict:
    connection = sqlite3.connect(path)
    try:
        result: dict = {"path": str(path)}
        if has_table(connection, "TARGET_INFO_GPU_METRICS"):
            result["gpu_metrics"] = rows(
                connection,
                "SELECT * FROM TARGET_INFO_GPU_METRICS ORDER BY typeId, metricId",
            )
        if has_table(connection, "DIAGNOSTIC_EVENT"):
            result["diagnostics"] = rows(
                connection,
                "SELECT timestamp, severity, text FROM DIAGNOSTIC_EVENT ORDER BY timestamp",
            )
        if has_table(connection, "NVTX_EVENTS"):
            result["nvtx_training_summary"] = rows(
                connection,
                "SELECT COUNT(*) AS event_count, MIN(text) AS first_label, MAX(text) AS last_label, "
                "MIN(start) AS first_start_ns, MAX(end) AS last_end_ns FROM NVTX_EVENTS "
                "WHERE text LIKE 'TRAIN_%'",
            )
        if has_table(connection, "META_DATA_CAPTURE"):
            result["cpu_metadata"] = rows(
                connection,
                "SELECT name, value FROM META_DATA_CAPTURE "
                "WHERE lower(name) LIKE '%cpu%' OR lower(value) LIKE '%cpu-metric%' "
                "ORDER BY name",
            )
        return result
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps([probe(path) for path in args.sqlite], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
