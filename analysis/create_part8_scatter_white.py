from __future__ import annotations

import os
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd

from common import DEFAULT_OUTPUT_DIR


OUTPUT = (
    DEFAULT_OUTPUT_DIR
    / "presentations"
    / "part7_part8_assets"
    / "part8_scatter_white.png"
)


def main() -> None:
    configured_font = os.environ.get("PRE6G_CJK_FONT")
    font_path = Path(configured_font).expanduser() if configured_font else None
    if font_path is None:
        windows_dir = os.environ.get("WINDIR")
        candidate = Path(windows_dir) / "Fonts" / "msjh.ttc" if windows_dir else None
        font_path = candidate if candidate and candidate.is_file() else None
    font = (
        fm.FontProperties(fname=str(font_path))
        if font_path and font_path.is_file()
        else fm.FontProperties(family="sans-serif")
    )

    predictions = pd.read_csv(
        DEFAULT_OUTPUT_DIR
        / "model_experiments"
        / "part8_marker_free_runtime"
        / "predictions.csv"
    )
    primary = predictions[
        (predictions.training_scope == "direction_specific")
        & (predictions.model == "nested_selected_deployment_ridge")
    ]

    fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for direction, group in primary.groupby("direction"):
        is_forward = direction.startswith("RTX4090")
        ax.scatter(
            group.actual_ms,
            group.predicted_ms,
            s=42,
            color="#18a999" if is_forward else "#2e86de",
            edgecolors="#107f74" if is_forward else "#1f65a8",
            linewidths=0.5,
            alpha=0.9,
            label="4090 → 5090" if is_forward else "5090 → 4090",
            zorder=3,
        )

    low = min(primary.actual_ms.min(), primary.predicted_ms.min()) * 0.94
    high = max(primary.actual_ms.max(), primary.predicted_ms.max()) * 1.04
    ax.plot(
        [low, high],
        [low, high],
        "--",
        color="#e59400",
        linewidth=1.7,
        label="理想預測",
        zorder=2,
    )

    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("實際 target iteration time (ms)", fontproperties=font, color="#20252b")
    ax.set_ylabel("預測 iteration time (ms)", fontproperties=font, color="#20252b")
    ax.set_title(
        "Part 8 OOF 預測：實際值 vs. 預測值",
        fontproperties=font,
        fontsize=13,
        weight="bold",
        color="#20252b",
    )

    ax.grid(color="#9aa4ad", alpha=0.28, linewidth=0.7)
    ax.legend(prop=font, frameon=False, loc="upper left", labelcolor="#20252b")
    ax.tick_params(axis="both", colors="#30363d")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font)
    for spine in ax.spines.values():
        spine.set_color("#727b84")
        spine.set_linewidth(0.8)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT, facecolor="white", transparent=False, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
