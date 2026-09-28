from __future__ import annotations

from pathlib import Path
from typing import Any

from common import PROJECT_ROOT, load_json
from training_core import create_parser, execute_training

DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "synthetic_dnn"


def load_dataset(torch, root: Path, workload: dict[str, Any]):
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Run scripts/generate_dataset.py first")
    manifest = load_json(manifest_path)
    entry = manifest["files"].get(str(workload["input_dim"]))
    if entry is None:
        raise ValueError(f"Missing input_dim={workload['input_dim']} dataset")
    payload = torch.load(root / entry["file"], map_location="cpu", weights_only=True)
    return payload["inputs"], payload["labels"], manifest


def build_model(torch, nn, workload: dict[str, Any], config: dict[str, Any]):
    layers: list[Any] = []
    input_dim = int(workload["input_dim"])
    hidden_dim = int(workload["hidden_dim"])
    for index in range(int(workload["depth"])):
        layers.append(nn.Linear(input_dim if index == 0 else hidden_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
    layers.append(nn.Linear(hidden_dim, int(workload["num_classes"])))
    return nn.Sequential(*layers)


def main() -> None:
    parser = create_parser("Run one dense MLP classification training workload", DEFAULT_DATASET_ROOT)
    args = parser.parse_args()
    execute_training(args, build_model, load_dataset)


if __name__ == "__main__":
    main()
