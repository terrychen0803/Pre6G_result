from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_GIT_FILE_BYTES = 50 * 1024 * 1024
REQUIRED_PATHS = (
    "README.md",
    "ENVIRONMENT.md",
    "REPRODUCIBILITY.md",
    "SOURCE_INVENTORY.md",
    "analysis/README.md",
    "configs/yolo26/experiments.csv",
    "data/README.md",
    "docs/research_handoff.md",
    "notebooks/00_research_overview.ipynb",
    "notebooks/99_final_results_summary.ipynb",
)
FORBIDDEN_DIRECTORY_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    "analysis_cache",
    "hf_cache",
    "site-packages",
}
FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".db",
    ".nsys-rep",
    ".npz",
    ".pth",
    ".pt",
    ".safetensors",
    ".sqlite",
}
LOCAL_WORKSPACE_PATTERN = re.compile(
    r"(?:[A-Za-z]:[\\/].*python" + r"_projects[\\/]|/home/[^/]+/|/mnt/[A-Za-z]/)",
    re.IGNORECASE,
)
TEXT_SUFFIXES = {
    ".cmd",
    ".csv",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def add_error(errors: list[str], message: str) -> None:
    errors.append(message)


def validate() -> list[str]:
    errors: list[str] = []

    for relative in REQUIRED_PATHS:
        if not (ROOT / relative).is_file():
            add_error(errors, f"missing required file: {relative}")

    for path in ROOT.rglob("*"):
        relative_path = path.relative_to(ROOT)
        if relative_path.parts and relative_path.parts[0] == ".git":
            continue
        relative = relative_path.as_posix()
        if path.is_dir():
            if path.name in FORBIDDEN_DIRECTORY_NAMES:
                add_error(errors, f"forbidden directory: {relative}")
            continue

        if path.stat().st_size > MAX_GIT_FILE_BYTES:
            add_error(errors, f"file exceeds 50 MiB: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            add_error(errors, f"forbidden runtime artifact: {relative}")

        if path.suffix.lower() == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=relative)
            except (SyntaxError, UnicodeError) as exc:
                add_error(errors, f"invalid Python file {relative}: {exc}")

        if path.suffix.lower() == ".ipynb":
            try:
                value = json.loads(path.read_text(encoding="utf-8-sig"))
                if not isinstance(value, dict) or "cells" not in value:
                    raise ValueError("Notebook root must contain cells")
            except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
                add_error(errors, f"invalid Notebook {relative}: {exc}")

        if path.suffix.lower() == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8-sig"))
            except (json.JSONDecodeError, UnicodeError) as exc:
                add_error(errors, f"invalid JSON file {relative}: {exc}")

        if path.suffix.lower() in TEXT_SUFFIXES:
            try:
                text = path.read_text(encoding="utf-8-sig")
            except UnicodeError as exc:
                add_error(errors, f"non-UTF-8 text file {relative}: {exc}")
                continue
            if "\x00" in text:
                add_error(errors, f"NUL byte in text file: {relative}")
            if relative != "scripts/validate_publication.py" and LOCAL_WORKSPACE_PATTERN.search(text):
                add_error(errors, f"local workspace path in text file: {relative}")

    return errors


def main() -> int:
    errors = validate()
    if errors:
        print(f"Publication validation failed with {len(errors)} issue(s):")
        for error in errors:
            print(f"- {error}")
        return 1
    file_count = sum(
        1
        for path in ROOT.rglob("*")
        if path.is_file() and path.relative_to(ROOT).parts[0] != ".git"
    )
    print(f"Publication validation passed: {file_count} files checked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
