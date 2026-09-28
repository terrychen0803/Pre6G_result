from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


KEYWORDS = ("CUDA", "NVTX", "GPU", "METRIC", "SAMPL", "ENUM", "STRING")


def inspect(path: Path) -> dict:
    connection = sqlite3.connect(path)
    try:
        names = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        selected = []
        for name in names:
            if not any(keyword in name.upper() for keyword in KEYWORDS):
                continue
            escaped = name.replace('"', '""')
            columns = [
                f"{row[1]}:{row[2]}"
                for row in connection.execute(f'PRAGMA table_info("{escaped}")')
            ]
            count = connection.execute(f'SELECT COUNT(*) FROM "{escaped}"').fetchone()[0]
            selected.append({"table": name, "rows": count, "columns": columns})
        return {"path": str(path), "table_count": len(names), "selected": selected}
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", nargs="+", type=Path)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.all:
        global KEYWORDS
        KEYWORDS = ("",)
    print(json.dumps([inspect(path) for path in args.sqlite], indent=2))


if __name__ == "__main__":
    main()
