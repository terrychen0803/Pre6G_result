from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "predictions.csv"
OUTPUT_PATH = HERE / "predictions_scatter_by_protocol.png"


MODEL_LABELS = {
    "period_quality_ridge": "Period quality ridge",
    "behavior_only_ridge": "Behavior-only ridge",
    "period_compact_ridge": "Period compact ridge",
    "period_full_ridge": "Period full ridge",
    "nested_selected_ridge": "Nested selected ridge",
    "period_median_ratio": "Period median ratio",
}

PROTOCOL_LABELS = {
    "yolo_only_loo": "YOLO-only LOO",
    "combined_yolo_gemma_loo": "Combined YOLO + Gemma LOO",
    "train_gemma_test_yolo": "Train Gemma / Test YOLO",
    "train_yolo_test_gemma": "Train YOLO / Test Gemma",
}

PROTOCOL_ORDER = list(PROTOCOL_LABELS)
MODEL_ORDER = list(MODEL_LABELS)
COLORS = dict(zip(MODEL_ORDER, plt.get_cmap("tab10").colors[: len(MODEL_ORDER)]))
MARKERS = {"yolo26": "o", "gemma4": "^"}


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    required = {"protocol", "model", "workload_family", "actual_ms", "predicted_ms"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 12.5), sharex=True, sharey=True)

    axis_min = min(df["actual_ms"].min(), df["predicted_ms"].min())
    axis_max = max(df["actual_ms"].max(), df["predicted_ms"].max())
    padding = (axis_max - axis_min) * 0.045
    lower = max(0, axis_min - padding)
    upper = axis_max + padding

    for ax, protocol in zip(axes.flat, PROTOCOL_ORDER):
        panel = df.loc[df["protocol"] == protocol]
        for model in MODEL_ORDER:
            for family, marker in MARKERS.items():
                points = panel.loc[
                    (panel["model"] == model) & (panel["workload_family"] == family)
                ]
                if points.empty:
                    continue
                ax.scatter(
                    points["actual_ms"],
                    points["predicted_ms"],
                    s=55,
                    marker=marker,
                    color=COLORS[model],
                    edgecolor="white",
                    linewidth=0.55,
                    alpha=0.82,
                    zorder=3,
                )

        ax.plot(
            [lower, upper],
            [lower, upper],
            color="#DE8F05",
            linestyle="--",
            linewidth=2.2,
            zorder=2,
        )
        mape = np.mean(np.abs((panel["predicted_ms"] - panel["actual_ms"]) / panel["actual_ms"])) * 100
        ax.text(
            0.025,
            0.965,
            f"n = {len(panel)}   |   MAPE = {mape:.1f}%",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10.5,
            color="#374151",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.9},
        )
        ax.set_title(PROTOCOL_LABELS[protocol], fontsize=15, weight="semibold", pad=11)
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#9CA3AF", alpha=0.3, linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_color("#6B7280")
            spine.set_linewidth(0.9)

    fig.supxlabel("Actual target iteration time (ms)", fontsize=15, y=0.075)
    fig.supylabel("Predicted iteration time (ms)", fontsize=15, x=0.045)
    fig.suptitle(
        "Runtime Prediction: Actual vs. Predicted",
        fontsize=23,
        weight="semibold",
        y=0.98,
    )

    model_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=8,
            markerfacecolor=COLORS[m], markeredgecolor="white", label=MODEL_LABELS[m]
        )
        for m in MODEL_ORDER
    ]
    family_handles = [
        Line2D(
            [0], [0], marker=marker, linestyle="none", markersize=8,
            markerfacecolor="#4B5563", markeredgecolor="white", label=family.upper()
        )
        for family, marker in MARKERS.items()
    ]
    ideal_handle = Line2D([0], [0], color="#DE8F05", linestyle="--", linewidth=2.2, label="Ideal prediction")
    fig.legend(
        handles=model_handles + family_handles + [ideal_handle],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        frameon=False,
        fontsize=10.5,
        columnspacing=1.35,
        handletextpad=0.55,
    )

    fig.tight_layout(rect=(0.055, 0.105, 1, 0.945), h_pad=2.0, w_pad=1.4)
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
