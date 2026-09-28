from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from common import DEFAULT_EXPERIMENTS, PROJECT_ROOT, load_json, load_workloads

DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "synthetic_vit"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic ViT images")
    parser.add_argument("--experiments", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    workloads = load_workloads(args.experiments.resolve())
    config = load_json(PROJECT_ROOT / "configs" / "model_config.json")
    image_sizes = sorted({int(item["image_size"]) for item in workloads.values()})
    classes = sorted({int(item["num_classes"]) for item in workloads.values()})
    if len(classes) != 1:
        raise ValueError("All workloads must share num_classes for one dataset manifest")
    samples = int(config["dataset_samples_per_image_size"])
    channels = int(config["channels"])
    print(f"Generate image_sizes={image_sizes} samples={samples} classes={classes[0]}")
    if args.dry_run:
        return

    import torch

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(int(config["seed"]))
    manifest = {
        "schema_version": 1,
        "dataset": "deterministic_synthetic_image_classification",
        "seed": int(config["seed"]),
        "samples": samples,
        "channels": channels,
        "num_classes": classes[0],
        "files": {},
    }
    for image_size in image_sizes:
        path = output / f"images_{image_size}.pt"
        if path.exists() and not args.force:
            raise FileExistsError(f"{path} exists; pass --force to regenerate")
        # uint8 keeps the portable dataset compact. Conversion to floating point
        # occurs before the measured H2D transfer and is not part of step timing.
        payload = {
            "images": torch.randint(
                0, 256, (samples, channels, image_size, image_size),
                dtype=torch.uint8, generator=generator
            ),
            "labels": torch.randint(classes[0], (samples,), generator=generator),
        }
        torch.save(payload, path)
        manifest["files"][str(image_size)] = {
            "file": path.name,
            "sha256": sha256(path),
            "shape": [samples, channels, image_size, image_size],
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {output / 'manifest.json'}")


if __name__ == "__main__":
    main()
