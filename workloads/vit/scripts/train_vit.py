from __future__ import annotations

from pathlib import Path
from typing import Any

from common import PROJECT_ROOT, load_json
from training_core import create_parser, execute_training

DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "synthetic_vit"


def load_dataset(torch, root: Path, workload: dict[str, Any]):
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Run scripts/generate_dataset.py first")
    manifest = load_json(manifest_path)
    entry = manifest["files"].get(str(workload["image_size"]))
    if entry is None:
        raise ValueError(f"Missing image_size={workload['image_size']} dataset")
    payload = torch.load(root / entry["file"], map_location="cpu", weights_only=True)
    images = payload["images"].float().div_(255.0)
    return images, payload["labels"], manifest


class VisionTransformerFactory:
    @staticmethod
    def build(torch, nn, workload: dict[str, Any], config: dict[str, Any]):
        image_size = int(workload["image_size"])
        patch_size = int(workload["patch_size"])
        embed_dim = int(workload["embed_dim"])
        num_heads = int(workload["num_heads"])
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")

        class VisionTransformer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.patch_embed = nn.Conv2d(
                    int(config["channels"]), embed_dim,
                    kernel_size=patch_size, stride=patch_size,
                )
                token_count = (image_size // patch_size) ** 2
                self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
                self.position = nn.Parameter(torch.zeros(1, token_count + 1, embed_dim))
                self.encoder = nn.ModuleList(
                    [
                        nn.TransformerEncoderLayer(
                            d_model=embed_dim,
                            nhead=num_heads,
                            dim_feedforward=int(
                                embed_dim * float(workload["mlp_ratio"])
                            ),
                            dropout=float(config["dropout"]),
                            activation="gelu",
                            batch_first=True,
                            norm_first=True,
                        )
                        for _ in range(int(workload["depth"]))
                    ]
                )
                self.norm = nn.LayerNorm(embed_dim)
                self.head = nn.Linear(embed_dim, int(workload["num_classes"]))
                nn.init.normal_(self.class_token, std=0.02)
                nn.init.normal_(self.position, std=0.02)

            def forward(self, images):
                tokens = self.patch_embed(images).flatten(2).transpose(1, 2)
                class_tokens = self.class_token.expand(images.shape[0], -1, -1)
                tokens = torch.cat((class_tokens, tokens), dim=1) + self.position
                encoded = tokens
                for layer in self.encoder:
                    encoded = layer(encoded)
                return self.head(self.norm(encoded[:, 0]))

        return VisionTransformer()


def main() -> None:
    parser = create_parser("Run one Vision Transformer classification workload", DEFAULT_DATASET_ROOT)
    args = parser.parse_args()
    execute_training(args, VisionTransformerFactory.build, load_dataset)


if __name__ == "__main__":
    main()
