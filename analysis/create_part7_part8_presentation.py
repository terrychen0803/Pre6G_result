from __future__ import annotations

import os
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

from common import DEFAULT_OUTPUT_DIR, PROJECT_ROOT


OUT = DEFAULT_OUTPUT_DIR / "presentations"
ASSETS = OUT / "part7_part8_assets"
PPTX = OUT / "part7_part8_marker_free_runtime_prediction.pptx"

NAVY = RGBColor(10, 18, 32)
NAVY_2 = RGBColor(18, 30, 49)
PANEL = RGBColor(25, 40, 61)
PANEL_2 = RGBColor(34, 52, 75)
WHITE = RGBColor(245, 248, 252)
MUTED = RGBColor(174, 190, 207)
TEAL = RGBColor(43, 203, 187)
CYAN = RGBColor(73, 167, 255)
AMBER = RGBColor(255, 184, 77)
RED = RGBColor(255, 104, 104)
GREEN = RGBColor(100, 220, 145)
FONT = "Microsoft JhengHei"


def set_run(run, size=18, color=WHITE, bold=False, font=FONT):
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color


def add_text(slide, text, x, y, w, h, size=18, color=WHITE, bold=False,
             align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP, margin=0.04):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.margin_left = Inches(margin)
    tf.margin_right = Inches(margin)
    tf.margin_top = Inches(margin)
    tf.margin_bottom = Inches(margin)
    tf.vertical_anchor = valign
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    set_run(run, size, color, bold)
    return box


def add_rich_text(slide, parts, x, y, w, h, size=18, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.margin_left = tf.margin_right = Inches(0.04)
    p = tf.paragraphs[0]
    p.alignment = align
    for text, color, bold in parts:
        run = p.add_run()
        run.text = text
        set_run(run, size, color, bold)
    return box


def add_box(slide, x, y, w, h, fill=PANEL, line=None, radius=True):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(shape_type, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line or fill
    return shape


def add_bullets(slide, items, x, y, w, h, size=18, color=WHITE, spacing=9,
                bullet_color=None):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(0.06)
    tf.margin_right = Inches(0.02)
    for idx, item in enumerate(items):
        if isinstance(item, tuple):
            text, level = item
        else:
            text, level = item, 0
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = text
        p.level = level
        p.font.name = FONT
        p.font.size = Pt(size - level * 2)
        p.font.color.rgb = color
        p.space_after = Pt(spacing)
        p.line_spacing = 1.06
        p.text = ("• " if level == 0 else "  – ") + text
    return box


def add_title(slide, title, kicker=None, number=None):
    if kicker:
        add_text(slide, kicker.upper(), 0.58, 0.27, 7.6, 0.3, 10, TEAL, True)
    add_text(slide, title, 0.55, 0.58, 12.1, 0.65, 27, WHITE, True)
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.58), Inches(1.26), Inches(1.0), Inches(0.04))
    line.fill.solid(); line.fill.fore_color.rgb = TEAL; line.line.color.rgb = TEAL
    if number is not None:
        add_text(slide, f"{number:02d}", 12.35, 0.33, 0.45, 0.3, 10, MUTED, True, PP_ALIGN.RIGHT)


def add_footer(slide, section="PART 7–8", page=None):
    add_text(slide, "YOLO26 跨異質 GPU Training Runtime Prediction", 0.55, 7.17, 6.7, 0.18, 8, MUTED)
    add_text(slide, section, 10.65, 7.17, 1.7, 0.18, 8, MUTED, True, PP_ALIGN.RIGHT)
    if page is not None:
        add_text(slide, str(page), 12.45, 7.17, 0.3, 0.18, 8, MUTED, True, PP_ALIGN.RIGHT)


def new_slide(prs, title, kicker, page):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid(); slide.background.fill.fore_color.rgb = NAVY
    add_title(slide, title, kicker, page)
    add_footer(slide, kicker, page)
    return slide


def add_metric_card(slide, x, y, w, h, value, label, accent=TEAL, note=None):
    add_box(slide, x, y, w, h, PANEL)
    slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(0.06), Inches(h)).fill.solid()
    bar = slide.shapes[-1]; bar.fill.fore_color.rgb = accent; bar.line.color.rgb = accent
    add_text(slide, value, x + 0.22, y + 0.16, w - 0.35, 0.45, 25, accent, True)
    add_text(slide, label, x + 0.22, y + 0.69, w - 0.35, 0.28, 12, WHITE, True)
    if note:
        add_text(slide, note, x + 0.22, y + 1.02, w - 0.35, h - 1.1, 9, MUTED)


def add_flow(slide, labels, x, y, total_w, h, colors=None, size=14):
    gap = 0.25
    n = len(labels)
    box_w = (total_w - gap * (n - 1)) / n
    for i, label in enumerate(labels):
        bx = x + i * (box_w + gap)
        color = colors[i] if colors else PANEL_2
        add_box(slide, bx, y, box_w, h, color)
        add_text(slide, label, bx + 0.08, y + 0.08, box_w - 0.16, h - 0.16,
                 size, WHITE, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
        if i < n - 1:
            add_text(slide, "→", bx + box_w, y + h / 2 - 0.17, gap, 0.34, 16, TEAL, True, PP_ALIGN.CENTER)


def add_table(slide, data, x, y, w, h, col_widths=None, font_size=11, header_fill=PANEL_2):
    rows, cols = len(data), len(data[0])
    table = slide.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w), Inches(h)).table
    if col_widths:
        for i, width in enumerate(col_widths):
            table.columns[i].width = Inches(width)
    for r in range(rows):
        for c in range(cols):
            cell = table.cell(r, c)
            cell.text = str(data[r][c])
            cell.fill.solid(); cell.fill.fore_color.rgb = header_fill if r == 0 else PANEL
            cell.margin_left = cell.margin_right = Inches(0.07)
            cell.margin_top = cell.margin_bottom = Inches(0.04)
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER if c > 0 else PP_ALIGN.LEFT
                p.vertical_anchor = MSO_ANCHOR.MIDDLE
                for run in p.runs:
                    set_run(run, font_size if r else font_size + 1, WHITE if r else TEAL, r == 0)
    return table


def save_charts():
    ASSETS.mkdir(parents=True, exist_ok=True)
    configured_font = os.environ.get("PRE6G_CJK_FONT")
    font_path = Path(configured_font).expanduser() if configured_font else None
    if font_path is None:
        windows_dir = os.environ.get("WINDIR")
        candidate = Path(windows_dir) / "Fonts" / "msjh.ttc" if windows_dir else None
        font_path = candidate if candidate and candidate.is_file() else None
    fp = (
        fm.FontProperties(fname=str(font_path))
        if font_path and font_path.is_file()
        else fm.FontProperties(family="sans-serif")
    )
    plt.rcParams.update({"axes.facecolor": "#121e31", "figure.facecolor": "#0a1220",
                         "axes.edgecolor": "#62758a", "xtick.color": "#dbe6ef",
                         "ytick.color": "#dbe6ef", "text.color": "#f5f8fc",
                         "axes.labelcolor": "#dbe6ef", "font.size": 11})

    cycle = pd.read_csv(DEFAULT_OUTPUT_DIR / "model_experiments/part7_marker_free_cycles/cycle_metrics.csv")
    both = cycle[cycle.device_id == "both_devices"].set_index("horizon_seconds")
    labels = ["5 秒\n強制輸出", "7 秒\n強制輸出", "9 秒\n強制輸出", "Adaptive\n9→12→15 秒"]
    within = [both.loc[5, "within_10_percent"] * 100, both.loc[7, "within_10_percent"] * 100,
              both.loc[9, "within_10_percent"] * 100, 100]
    colors = ["#ff6868", "#ffb84d", "#49a7ff", "#2bcbbb"]
    fig, ax = plt.subplots(figsize=(8.8, 3.7), dpi=180)
    bars = ax.bar(labels, within, color=colors, width=.62)
    ax.set_ylim(0, 110); ax.set_ylabel("週期誤差 <= 10% 的比例", fontproperties=fp)
    ax.set_title("延長 profiling 並以 confidence gate 可消除強制輸出的離群值", fontproperties=fp, fontsize=13, weight="bold")
    ax.grid(axis="y", alpha=.16)
    for bar, val in zip(bars, within):
        ax.text(bar.get_x()+bar.get_width()/2, val+2, f"{val:.1f}%", ha="center", color="white", weight="bold")
    for label in ax.get_xticklabels()+ax.get_yticklabels(): label.set_fontproperties(fp)
    fig.tight_layout(); fig.savefig(ASSETS / "part7_acceptance.png", transparent=False); plt.close(fig)

    pred = pd.read_csv(DEFAULT_OUTPUT_DIR / "model_experiments/part8_marker_free_runtime/predictions.csv")
    primary = pred[(pred.training_scope == "direction_specific") &
                   (pred.model == "nested_selected_deployment_ridge")]
    fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=180)
    for direction, group in primary.groupby("direction"):
        color = "#2bcbbb" if direction.startswith("RTX4090") else "#49a7ff"
        label = "4090 → 5090" if direction.startswith("RTX4090") else "5090 → 4090"
        ax.scatter(group.actual_ms, group.predicted_ms, s=42, color=color, alpha=.9, label=label)
    low = min(primary.actual_ms.min(), primary.predicted_ms.min()) * .94
    high = max(primary.actual_ms.max(), primary.predicted_ms.max()) * 1.04
    ax.plot([low, high], [low, high], "--", color="#ffb84d", lw=1.5, label="理想預測")
    ax.set_xlim(low, high); ax.set_ylim(low, high)
    ax.set_xlabel("實際 target iteration time (ms)", fontproperties=fp)
    ax.set_ylabel("預測 iteration time (ms)", fontproperties=fp)
    ax.set_title("Part 8 OOF 預測：實際值 vs. 預測值", fontproperties=fp, fontsize=13, weight="bold")
    ax.grid(alpha=.16); ax.legend(prop=fp, frameon=False, loc="upper left")
    for label in ax.get_xticklabels()+ax.get_yticklabels(): label.set_fontproperties(fp)
    fig.tight_layout(); fig.savefig(ASSETS / "part8_scatter.png"); plt.close(fig)

    metrics = pd.read_csv(DEFAULT_OUTPUT_DIR / "model_experiments/part8_marker_free_runtime/metrics.csv")
    subset = metrics[(metrics.training_scope == "direction_specific") &
                     (metrics.direction == "both_directions_combined")]
    models = ["period_direction_median_ratio", "period_only_ridge", "behavior_only_ridge",
              "period_compact_behavior_ridge", "period_behavior_quality_ridge",
              "nested_selected_deployment_ridge"]
    names = ["固定 period ratio", "Period only", "Behavior only", "Period + compact behavior",
             "Period + behavior + quality", "Nested 正式結果"]
    vals = [float(subset[subset.model == model].mape_percent.iloc[0]) for model in models]
    fig, ax = plt.subplots(figsize=(8.5, 4.1), dpi=180)
    bars = ax.barh(names[::-1], vals[::-1], color=["#2bcbbb", "#49a7ff", "#49a7ff", "#ffb84d", "#ffb84d", "#ff6868"])
    ax.set_xlim(0, 12); ax.set_xlabel("MAPE (%)", fontproperties=fp)
    ax.set_title("Period 提供時間尺度；behavior 負責 workload-dependent scaling", fontproperties=fp, fontsize=13, weight="bold")
    ax.grid(axis="x", alpha=.16)
    for bar, val in zip(bars, vals[::-1]):
        ax.text(val+.18, bar.get_y()+bar.get_height()/2, f"{val:.2f}%", va="center", color="white", weight="bold")
    for label in ax.get_xticklabels()+ax.get_yticklabels(): label.set_fontproperties(fp)
    fig.tight_layout(); fig.savefig(ASSETS / "part8_ablation.png"); plt.close(fig)

    return primary


def build_deck():
    OUT.mkdir(parents=True, exist_ok=True)
    primary = save_charts()
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # 1
    slide = prs.slides.add_slide(prs.slide_layouts[6]); slide.background.fill.solid(); slide.background.fill.fore_color.rgb = NAVY
    add_text(slide, "MARKER-FREE PROFILING", .62, .55, 5.5, .35, 11, TEAL, True)
    add_text(slide, "從週期偵測到跨 GPU\nPer-Iteration Runtime Prediction", .62, 1.18, 8.1, 1.55, 31, WHITE, True)
    add_text(slide, "Part 7：Marker-free cycle detection  ×  Part 8：Cross-device runtime modeling", .66, 3.02, 8.9, .45, 15, MUTED)
    add_box(slide, 9.52, .78, 3.12, 5.55, PANEL)
    add_text(slide, "48", 9.92, 1.25, 2.3, .65, 36, TEAL, True, PP_ALIGN.CENTER)
    add_text(slide, "GPU × workload profiles", 9.8, 1.97, 2.55, .35, 12, WHITE, True, PP_ALIGN.CENTER)
    add_text(slide, "1.01%", 9.92, 2.72, 2.3, .58, 31, CYAN, True, PP_ALIGN.CENTER)
    add_text(slide, "Part 7 period MAPE", 9.8, 3.34, 2.55, .32, 12, WHITE, True, PP_ALIGN.CENTER)
    add_text(slide, "4.50%", 9.92, 4.12, 2.3, .58, 31, AMBER, True, PP_ALIGN.CENTER)
    add_text(slide, "Part 8 nested MAPE", 9.8, 4.74, 2.55, .32, 12, WHITE, True, PP_ALIGN.CENTER)
    add_text(slide, "RTX4090 ↔ RTX5090", 9.8, 5.52, 2.55, .3, 12, MUTED, True, PP_ALIGN.CENTER)
    add_text(slide, "資料與結果截至 2026-08-19", .66, 6.76, 4.3, .26, 10, MUTED)

    # 2
    slide = new_slide(prs, "研究問題與部署場景", "CONTEXT", 2)
    add_box(slide, .62, 1.58, 5.65, 4.78, PANEL)
    add_text(slide, "目標", .92, 1.88, 1.2, .35, 15, TEAL, True)
    add_text(slide, "新訓練任務只在來源 GPU\n執行約 9–15 秒 profiling", .92, 2.35, 4.9, .82, 23, WHITE, True)
    add_text(slide, "不修改使用者程式、不知道 iteration 數量，預測另一裝置正常訓練時的 per-iteration runtime。", .92, 3.45, 4.85, 1.15, 16, MUTED)
    add_text(slide, "輸出：target iteration time（ms）", .92, 5.36, 4.85, .42, 16, AMBER, True)
    add_flow(slide, ["未知任務", "Source profiling", "Marker-free\nperiod + behavior", "Target runtime"], 6.72, 2.05, 5.95, 1.2,
             [PANEL_2, PANEL_2, RGBColor(25, 86, 89), RGBColor(78, 61, 31)], 13)
    add_bullets(slide, ["不能依賴 NVTX / callback", "不能使用 source baseline ground truth", "target 端只有硬體描述，尚未執行 workload"], 6.9, 3.8, 5.4, 1.85, 16, WHITE, 12)

    # 3
    slide = new_slide(prs, "End-to-End 研究架構", "PIPELINE", 3)
    add_flow(slide, ["資料收集", "時序對齊", "Part 7\n週期偵測", "Part 8\n週期特徵", "跨裝置模型", "誤差分析"], .62, 1.72, 12.05, 1.1,
             [PANEL_2, PANEL_2, RGBColor(24, 79, 84), RGBColor(24, 79, 84), RGBColor(83, 62, 31), PANEL_2], 13)
    add_box(slide, .72, 3.35, 3.55, 2.45, PANEL)
    add_text(slide, "Deployment inputs", 1.02, 3.67, 2.8, .3, 14, TEAL, True)
    add_bullets(slide, ["Source GPU metrics", "Detected period / confidence", "Source & target hardware"], 1.0, 4.15, 2.9, 1.2, 14, WHITE, 7)
    add_box(slide, 4.85, 3.35, 3.55, 2.45, PANEL)
    add_text(slide, "Label only", 5.15, 3.67, 2.8, .3, 14, AMBER, True)
    add_bullets(slide, ["Target unprofiled baseline", "ground_truth_iter_ms", "只在 training fold 使用"], 5.13, 4.15, 2.9, 1.2, 14, WHITE, 7)
    add_box(slide, 8.98, 3.35, 3.55, 2.45, PANEL)
    add_text(slide, "Evaluation only", 9.28, 3.67, 2.8, .3, 14, RED, True)
    add_bullets(slide, ["NVTX period oracle", "Test target baseline", "不進 deployment features"], 9.26, 4.15, 2.9, 1.2, 14, WHITE, 7)

    # 4
    slide = new_slide(prs, "資料收集與實驗矩陣", "DATA COLLECTION", 4)
    data = [["項目", "RTX4090", "RTX5090", "用途"],
            ["Workloads", "24", "24", "Matched C01–C24"],
            ["Baseline repeats", "3 / workload", "3 / workload", "Target label"],
            ["Nsight GPU metrics", "完整", "完整", "Part 7/8 input"],
            ["Hardware descriptors", "45 欄", "45 欄", "Device mapping"],
            ["HPCToolkit postprocess", "已整理", "已整理", "Future ablation"]]
    add_table(slide, data, .7, 1.65, 7.8, 4.5, [2.2, 1.45, 1.45, 2.7], 12)
    add_metric_card(slide, 8.92, 1.72, 3.55, 1.48, "48", "matched source profiles", TEAL, "2 GPUs × 24 workloads")
    add_metric_card(slide, 8.92, 3.47, 3.55, 1.48, "48", "directed source→target samples", CYAN, "每個 source profile 對應另一裝置")
    add_box(slide, 8.92, 5.23, 3.55, .92, RGBColor(62, 45, 31))
    add_text(slide, "目前限制：只有兩個硬體節點", 9.14, 5.49, 3.1, .32, 13, AMBER, True, PP_ALIGN.CENTER)

    # 5
    slide = new_slide(prs, "資料角色與 Leakage 邊界", "EXPERIMENT DESIGN", 5)
    add_box(slide, .7, 1.58, 3.65, 4.72, RGBColor(22, 65, 69))
    add_text(slide, "✓ 可部署輸入", 1.0, 1.9, 3.0, .35, 17, TEAL, True)
    add_bullets(slide, ["五條 source GPU metrics", "Marker-free detected period", "Confidence / horizon", "Source / target hardware"], 1.0, 2.48, 2.95, 2.5, 15, WHITE, 11)
    add_box(slide, 4.85, 1.58, 3.65, 4.72, RGBColor(68, 52, 31))
    add_text(slide, "Label only", 5.15, 1.9, 3.0, .35, 17, AMBER, True)
    add_bullets(slide, ["Target baseline median runtime", "每個 direction 分別建立 y", "Held-out workload 的 y 只用於最後評分"], 5.15, 2.48, 2.95, 2.5, 15, WHITE, 11)
    add_box(slide, 9.0, 1.58, 3.65, 4.72, RGBColor(65, 35, 42))
    add_text(slide, "✕ 禁止模型輸入", 9.3, 1.9, 3.0, .35, 17, RED, True)
    add_bullets(slide, ["NVTX / callback", "batch / imgsz / AMP", "Source baseline runtime", "Target profiling metrics", "Oracle period"], 9.3, 2.48, 2.95, 2.8, 15, WHITE, 10)

    # 6
    slide = new_slide(prs, "Profiling Data：從 SQLite 到 1 ms 時序", "DATA PREPARATION", 6)
    add_flow(slide, ["GPU_METRICS\nraw timestamp", "共同時間範圍", "1 ms grid\nlinear interpolation", "5 × T matrix", "NPZ cache"], .75, 1.7, 11.85, 1.15,
             [PANEL_2, PANEL_2, PANEL_2, RGBColor(24, 79, 84), PANEL_2], 13)
    metrics = [["Metric", "代表行為", "選擇理由"],
               ["SM issue", "Instruction issuing", "一般 compute pattern"],
               ["SMS active", "SM busy / idle", "辨認 activity gap"],
               ["Tensor active", "Tensor Core", "混合精度 / matrix workload"],
               ["DRAM read", "Memory read", "Memory-bound signature"],
               ["DRAM write", "Memory write", "Backward / optimizer behavior"]]
    add_table(slide, metrics, .8, 3.32, 7.55, 2.8, [2.0, 2.25, 3.3], 10)
    add_box(slide, 8.72, 3.32, 3.83, 2.8, PANEL)
    add_text(slide, "為何不先全加？", 9.0, 3.65, 3.2, .3, 15, AMBER, True)
    add_bullets(slide, ["保留跨裝置共同定義", "涵蓋 compute / tensor / memory", "避免高度相關 channels 支配 consensus", "Clock、warp、trace 留待後續 ablation"], 8.98, 4.1, 3.2, 1.65, 13, WHITE, 7)

    # 7
    slide = new_slide(prs, "Part 7：Marker-Free Period Detection", "PART 7 METHOD", 7)
    add_flow(slide, ["Prefix\n5/7/9/12/15s", "Tail window\n50/65/80%", "Robust scale\nmedian / IQR", "5-channel ACF", "Median consensus", "Peak 15–250ms"], .62, 1.58, 12.05, 1.05,
             [PANEL_2, PANEL_2, PANEL_2, RGBColor(24, 79, 84), RGBColor(24, 79, 84), RGBColor(83, 62, 31)], 12)
    add_box(slide, .75, 3.14, 5.75, 2.7, PANEL)
    add_text(slide, "候選週期選擇", 1.05, 3.47, 2.5, .3, 16, TEAL, True)
    add_text(slide, "score(lag) = consensus ACF height + peak prominence", 1.05, 4.04, 4.95, .45, 17, WHITE, True)
    add_bullets(slide, ["Peak prominence ≥ 0.02", "候選至少相隔約 8 ms", "選 score 最大者；可能存在 half/double harmonic"], 1.03, 4.68, 4.9, 1.05, 13, MUTED, 6)
    add_box(slide, 6.86, 3.14, 5.72, 2.7, PANEL)
    add_text(slide, "重要界線", 7.16, 3.47, 2.5, .3, 16, AMBER, True)
    add_text(slide, "找到的是 period 長度，不是真正 iteration start。", 7.16, 4.04, 4.85, .62, 19, WHITE, True)
    add_text(slide, "NVTX 只在偵測完成後建立隱藏 oracle；detector 不查詢 marker table。", 7.16, 4.89, 4.75, .64, 13, MUTED)

    # 8
    slide = new_slide(prs, "Confidence 與 Adaptive Profiling", "PART 7 METHOD", 8)
    add_box(slide, .7, 1.55, 6.0, 2.05, PANEL)
    add_text(slide, "confidence =", 1.03, 1.95, 1.7, .4, 20, WHITE, True)
    add_text(slide, "clip(prominence, 0, 1)", 2.7, 1.95, 2.55, .4, 19, TEAL, True)
    add_text(slide, "× agreement × min(1, cycles / 5)", 1.03, 2.6, 5.1, .4, 19, CYAN, True)
    add_box(slide, 7.08, 1.55, 5.55, 2.05, PANEL)
    add_text(slide, "Confidence 不是正確機率", 7.42, 1.93, 4.7, .35, 18, AMBER, True)
    add_text(slide, "0.63 不代表 63% 正確；它是 peak 清晰度、多 channel 一致性與觀察週期數的 heuristic quality score。", 7.42, 2.45, 4.65, .74, 13, MUTED)
    add_flow(slide, ["9 秒\nconf ≥ 0.4?", "否 → 12 秒", "否 → 15 秒", "仍不足\n拒絕 / 延長"], 1.05, 4.42, 11.25, 1.15,
             [RGBColor(24, 79, 84), PANEL_2, PANEL_2, RGBColor(65, 35, 42)], 14)
    add_text(slide, "每個 outer fold 的 threshold 只由其他 23 workloads 選擇；24 folds 皆選到 0.4。", 1.1, 5.91, 11.1, .4, 13, MUTED, False, PP_ALIGN.CENTER)

    # 9
    slide = new_slide(prs, "Part 7 評估：Workload-Held-Out", "PART 7 EVALUATION", 9)
    add_box(slide, .72, 1.62, 4.0, 4.55, PANEL)
    add_text(slide, "Outer fold：以 C01 為例", 1.03, 1.98, 3.3, .35, 16, TEAL, True)
    add_text(slide, "Train", 1.03, 2.58, .8, .3, 13, MUTED, True)
    add_text(slide, "C02–C24 × 兩裝置", 1.03, 2.95, 3.1, .38, 18, WHITE, True)
    add_text(slide, "Test", 1.03, 3.62, .8, .3, 13, MUTED, True)
    add_text(slide, "C01/4090 + C01/5090", 1.03, 3.99, 3.2, .38, 18, AMBER, True)
    add_text(slide, "測試 workload 的 NVTX period 最後才揭露評分。", 1.03, 4.78, 3.15, .72, 13, MUTED)
    add_box(slide, 5.15, 1.62, 7.48, 4.55, PANEL)
    add_text(slide, "Nested detector selection", 5.48, 1.98, 3.7, .35, 16, CYAN, True)
    add_flow(slide, ["其他 23 workloads", "選 bin / tail", "選 confidence threshold", "套用 held-out C01"], 5.48, 2.65, 6.78, .95,
             [PANEL_2, PANEL_2, PANEL_2, RGBColor(24, 79, 84)], 11)
    add_bullets(slide, ["24 workloads × 2 devices = 48 OOF detections", "每一筆偵測都由未看過該 workload oracle 的 detector 產生", "評估的是同 workload family 中的 unseen configuration"], 5.52, 4.15, 6.15, 1.4, 14, WHITE, 9)

    # 10
    slide = new_slide(prs, "Part 7 結果：Adaptive Policy 消除離群值", "PART 7 RESULTS", 10)
    slide.shapes.add_picture(str(ASSETS / "part7_acceptance.png"), Inches(.7), Inches(1.55), width=Inches(7.75))
    add_metric_card(slide, 8.78, 1.66, 3.7, 1.3, "1.01%", "兩裝置 period MAPE", TEAL)
    add_metric_card(slide, 8.78, 3.14, 3.7, 1.3, "1.86%", "P90 period APE", CYAN)
    add_metric_card(slide, 8.78, 4.62, 3.7, 1.3, "5.61%", "最大 period APE", AMBER)
    add_text(slide, "33 個在 9 秒輸出｜13 個到 12 秒｜2 個到 15 秒｜平均 10.06 秒", .85, 6.25, 11.55, .38, 14, WHITE, True, PP_ALIGN.CENTER)

    # 11
    slide = new_slide(prs, "Part 8：依推定 Period 建立週期區塊", "PART 8 FEATURES", 11)
    add_text(slide, "範例：RTX4090 / C10", .78, 1.55, 3.0, .3, 15, TEAL, True)
    add_rich_text(slide, [("segment 3.15–9.00s  ｜  ", MUTED, False), ("period 57ms", AMBER, True), ("  ｜  102 blocks", MUTED, False)], .78, 1.96, 7.0, .35, 16)
    # timeline blocks
    add_text(slide, "真實 iteration（未知起點）", .78, 2.65, 2.3, .3, 12, MUTED, True)
    for i in range(8):
        x = 3.18 + i * 1.02
        s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(2.56), Inches(.97), Inches(.5))
        s.fill.solid(); s.fill.fore_color.rgb = RGBColor(42, 78, 97); s.line.color.rgb = CYAN
    add_text(slide, "推定 blocks（固定相位偏移）", .78, 3.47, 2.5, .3, 12, MUTED, True)
    for i in range(8):
        x = 3.52 + i * 1.02
        s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(3.38), Inches(.97), Inches(.5))
        s.fill.solid(); s.fill.fore_color.rgb = RGBColor(38, 86, 78); s.line.color.rgb = TEAL
    add_box(slide, .78, 4.42, 5.72, 1.55, PANEL)
    add_text(slide, "實際 reshape", 1.08, 4.72, 1.6, .3, 14, TEAL, True)
    add_text(slide, "[5 channels, 5814 samples]  →  [5, 102 cycles, 57 samples]", 1.08, 5.2, 4.95, .38, 17, WHITE, True)
    add_box(slide, 6.85, 4.42, 5.63, 1.55, PANEL)
    add_text(slide, "設計原則", 7.15, 4.72, 1.6, .3, 14, AMBER, True)
    add_text(slide, "不宣稱知道 forward/backward 邊界；使用 mean、distribution、FFT power、repeatability 等 phase-insensitive 特徵。", 7.15, 5.15, 4.75, .58, 13, WHITE)

    # 12
    slide = new_slide(prs, "83 個 Marker-Free Behavior Features", "PART 8 FEATURES", 12)
    add_metric_card(slide, .72, 1.55, 3.6, 1.42, "70", "5 metrics × 14 features", TEAL, "分布 7｜跨週期 3｜模板 4")
    add_metric_card(slide, 4.87, 1.55, 3.6, 1.42, "10", "Cross-metric correlations", CYAN, "五條 metrics 兩兩相關")
    add_metric_card(slide, 9.02, 1.55, 3.6, 1.42, "3", "Global summaries", AMBER, "Repeatability｜CV｜Correlation")
    features = [["類別", "每條 metric 的特徵", "目的"],
                ["Distribution (7)", "mean, std, CV, P10, median, P90, active fraction", "資源強度與占空型態"],
                ["Cross-cycle (3)", "cycle mean CV, span, trend", "穩態與 iteration 變異"],
                ["Template (4)", "amplitude, harmonic 1/2, repeatability", "週期形狀與一致性"]]
    add_table(slide, features, .78, 3.48, 11.75, 2.45, [2.1, 5.8, 3.85], 11)
    add_text(slide, "正式模型不輸入 feature_cycle_count、segment length 等可能受 capture termination 影響的稽核欄位。", .85, 6.23, 11.55, .35, 12, MUTED, False, PP_ALIGN.CENTER)

    # 13
    slide = new_slide(prs, "Part 8 模型：輸入、映射與輸出", "PART 8 MODEL", 13)
    add_flow(slide, ["Detected period\nlog(ms)", "43/83 behavior\nfeatures", "Confidence\n+ horizon", "Hardware\nlog ratios", "Ridge on\nlog runtime", "Target ms/iter"], .62, 1.7, 12.05, 1.18,
             [RGBColor(24, 79, 84), RGBColor(24, 79, 84), PANEL_2, PANEL_2, RGBColor(83, 62, 31), RGBColor(83, 62, 31)], 12)
    add_box(slide, .75, 3.42, 5.65, 2.38, PANEL)
    add_text(slide, "模型形式", 1.05, 3.72, 1.4, .3, 15, TEAL, True)
    add_text(slide, "log(target runtime) = β₀ + β₁ log(period) + Σ βᵢ behaviorᵢ", 1.05, 4.27, 4.95, .65, 17, WHITE, True)
    add_text(slide, "SimpleImputer → VarianceThreshold → StandardScaler → Ridge", 1.05, 5.19, 4.9, .3, 12, MUTED)
    add_box(slide, 6.85, 3.42, 5.73, 2.38, PANEL)
    add_text(slide, "Period 不是全部", 7.15, 3.72, 2.2, .3, 15, AMBER, True)
    add_text(slide, "相同 4090→5090：C01 ratio 0.646｜C10 0.689｜C24 0.824", 7.15, 4.27, 4.85, .58, 16, WHITE, True)
    add_text(slide, "Behavior features 用來估計 workload-dependent device scaling。", 7.15, 5.08, 4.82, .42, 13, MUTED)

    # 14
    slide = new_slide(prs, "Nested Leave-One-Workload-Out", "PART 8 EVALUATION", 14)
    add_flow(slide, ["Outer holdout\nC01 × 2 directions", "Inner 5-fold\non C02–C24", "選 feature family\n與 Ridge α", "Fit 23 workloads", "Predict C01\n2 rows"], .65, 1.6, 12.0, 1.16,
             [RGBColor(65, 35, 42), PANEL_2, PANEL_2, RGBColor(24, 79, 84), RGBColor(83, 62, 31)], 12)
    add_metric_card(slide, .82, 3.35, 3.4, 1.6, "24", "outer folds", TEAL, "每個 workload 輪流當未見資料")
    add_metric_card(slide, 4.97, 3.35, 3.4, 1.6, "48", "OOF predictions", CYAN, "24 workloads × 2 directions")
    add_metric_card(slide, 9.12, 3.35, 3.4, 1.6, "0", "same-workload leakage", GREEN, "兩裝置同一 workload 一起移除")
    add_text(slide, "OOF 是 24 個 evaluation models 各自對未見 workload 的預測；不是同一個 production model 對訓練資料重測。", .9, 5.55, 11.45, .58, 15, WHITE, True, PP_ALIGN.CENTER)

    # 15
    slide = new_slide(prs, "Part 8 結果與 Feature Ablation", "PART 8 RESULTS", 15)
    slide.shapes.add_picture(str(ASSETS / "part8_ablation.png"), Inches(.65), Inches(1.48), width=Inches(7.9))
    add_metric_card(slide, 8.86, 1.62, 3.58, 1.28, "4.50%", "Nested 正式 MAPE", TEAL)
    add_metric_card(slide, 8.86, 3.08, 3.58, 1.28, "2.68%", "Median APE", CYAN)
    add_metric_card(slide, 8.86, 4.54, 3.58, 1.28, "0.961", "R²", AMBER)
    add_text(slide, "Period-only 7.53% → 加入 behavior 後 4.35–4.50%；relative error reduction 約 40%。", .86, 6.18, 11.45, .38, 14, WHITE, True, PP_ALIGN.CENTER)

    # 16
    slide = new_slide(prs, "OOF Prediction 與方向差異", "PART 8 RESULTS", 16)
    slide.shapes.add_picture(str(ASSETS / "part8_scatter.png"), Inches(.62), Inches(1.43), width=Inches(6.35))
    add_box(slide, 7.28, 1.62, 5.25, 1.35, PANEL)
    add_text(slide, "RTX4090 → RTX5090", 7.58, 1.9, 2.75, .3, 15, TEAL, True)
    add_text(slide, "MAPE 3.25%｜Median 2.00%｜Max 14.01%", 7.58, 2.34, 4.4, .3, 14, WHITE, True)
    add_box(slide, 7.28, 3.18, 5.25, 1.35, PANEL)
    add_text(slide, "RTX5090 → RTX4090", 7.58, 3.46, 2.75, .3, 15, CYAN, True)
    add_text(slide, "MAPE 5.75%｜Median 3.36%｜Max 23.19%", 7.58, 3.9, 4.4, .3, 14, WHITE, True)
    add_box(slide, 7.28, 4.74, 5.25, 1.35, RGBColor(62, 38, 43))
    add_text(slide, "最大離群：C24", 7.58, 5.02, 2.75, .3, 15, RED, True)
    add_text(slide, "Actual 101.09 ms｜Predicted 77.65 ms｜APE 23.19%", 7.58, 5.46, 4.48, .32, 13, WHITE, True)

    # 17
    slide = new_slide(prs, "結果如何解讀？", "ANALYSIS", 17)
    cards = [
        ("Part 7 已非主要瓶頸", "Marker-free period MAPE 1.01%；oracle period-only 也沒有優於 detected period-only。", TEAL),
        ("Behavior 是必要訊息", "固定硬體倍率無法解釋不同 compute / tensor / memory bottleneck 的 scaling。", CYAN),
        ("平均值不等於保證", "MAPE 4.50%、median 2.68%，但最大誤差仍達 23.19%。", AMBER),
        ("硬體泛化尚未證明", "Direction-specific hardware features 為常數；兩裝置不足以建立 unseen-device 結論。", RED),
    ]
    positions = [(.72, 1.55), (6.78, 1.55), (.72, 4.0), (6.78, 4.0)]
    for (title, body, accent), (x, y) in zip(cards, positions):
        add_box(slide, x, y, 5.82, 1.92, PANEL)
        add_text(slide, title, x+.28, y+.25, 5.2, .32, 16, accent, True)
        add_text(slide, body, x+.28, y+.78, 5.2, .78, 14, WHITE)

    # 18
    slide = new_slide(prs, "限制與目前可宣稱的範圍", "LIMITATIONS", 18)
    add_box(slide, .7, 1.55, 5.9, 4.75, PANEL)
    add_text(slide, "可以宣稱", 1.02, 1.9, 2.0, .35, 17, TEAL, True)
    add_text(slide, "在 24 個 YOLO workload configurations、RTX4090/RTX5090 範圍內，約 9–15 秒 marker-free profiling 可達 4.50% OOF MAPE。", 1.02, 2.55, 4.95, 1.45, 20, WHITE, True)
    add_text(slide, "這是同 workload family 的 unseen configuration 測試。", 1.02, 4.55, 4.85, .52, 14, MUTED)
    add_box(slide, 6.95, 1.55, 5.68, 4.75, RGBColor(62, 38, 43))
    add_text(slide, "不能宣稱", 7.27, 1.9, 2.0, .35, 17, RED, True)
    add_bullets(slide, ["任意未知 GPU 都保持 4.5%", "跨 Transformer / LLM / diffusion 泛化", "完全無 marker 收集的 domain shift 已排除", "C24 類 outlier 已得到可靠 uncertainty bound"], 7.25, 2.52, 4.75, 2.65, 16, WHITE, 12)

    # 19
    slide = new_slide(prs, "Future Work：先擴資料，再擴模型", "FUTURE WORK", 19)
    add_flow(slide, ["① 增加硬體節點", "② 增加訓練任務", "③ 多 profiler 對齊", "④ Uncertainty / OOD"], .72, 1.58, 11.9, 1.0,
             [RGBColor(24, 79, 84), RGBColor(24, 79, 84), PANEL_2, RGBColor(83, 62, 31)], 14)
    future = [["優先級", "資料擴充", "建議驗證"],
              ["P0", "至少第 3 個 GPU；跨 Ada / Blackwell / datacenter", "Leave-one-device-out"],
              ["P1", "Classification / segmentation / Transformer / LLM", "Leave-one-family-out"],
              ["P2", "Batch、precision、optimizer、I/O regime", "OOD 與 slice metrics"],
              ["P3", "完全無 NVTX 的重新收集", "Deployment domain validation"]]
    add_table(slide, future, .78, 3.08, 11.75, 2.65, [1.15, 5.45, 5.15], 11)
    add_text(slide, "原則：在第三裝置與新 workload family 驗證前，不優先增加複雜模型容量。", .88, 6.15, 11.55, .38, 14, AMBER, True, PP_ALIGN.CENTER)

    # 20
    slide = new_slide(prs, "Future Work：將 Marker-Free Period 映射到 HPCToolkit", "FUTURE WORK", 20)
    add_flow(slide, ["Nsight metrics", "Part 7 period\n+ phase", "Clock / event\nalignment", "HPCT raw trace\nor CCT", "Per-cycle / window\nHPCT features", "Runtime model"], .58, 1.58, 12.15, 1.12,
             [RGBColor(24, 79, 84), RGBColor(24, 79, 84), PANEL_2, PANEL_2, RGBColor(83, 62, 31), RGBColor(83, 62, 31)], 11)
    add_box(slide, .72, 3.25, 5.75, 2.72, PANEL)
    add_text(slide, "可加入的特徵", 1.02, 3.58, 2.2, .3, 16, TEAL, True)
    add_bullets(slide, ["CPU call-path time / samples", "GPU kernel 與 callsite composition", "Memcpy / synchronization", "PMU、stall、memory behavior", "跨週期 stability / trend"], 1.0, 4.03, 4.85, 1.7, 14, WHITE, 7)
    add_box(slide, 6.84, 3.25, 5.78, 2.72, RGBColor(62, 45, 31))
    add_text(slide, "必要條件與風險", 7.14, 3.58, 2.2, .3, 16, AMBER, True)
    add_bullets(slide, ["有 timestamp：才能做 per-cycle 對齊", "只有 hpcprof aggregate：先做 stable-window / workload-level mapping", "不同 run：以共同 kernel/activity signature 對齊", "同時 profiling 可能衝突並增加 overhead"], 7.12, 4.03, 4.88, 1.75, 13, WHITE, 7)
    add_text(slide, "建議先做三層 ablation：GPU metrics → + HPCT aggregate → + time-aligned HPCT。", .88, 6.27, 11.55, .35, 14, WHITE, True, PP_ALIGN.CENTER)

    # 21
    slide = new_slide(prs, "結論與下一個實驗決策", "TAKEAWAYS", 21)
    add_metric_card(slide, .72, 1.55, 3.65, 1.52, "~10.1 秒", "平均 profiling 到輸出 period", TEAL)
    add_metric_card(slide, 4.84, 1.55, 3.65, 1.52, "1.01%", "Marker-free period MAPE", CYAN)
    add_metric_card(slide, 8.96, 1.55, 3.65, 1.52, "4.50%", "Nested runtime MAPE", AMBER)
    add_box(slide, .72, 3.62, 11.89, 2.22, PANEL)
    add_text(slide, "核心發現", 1.04, 3.93, 1.6, .32, 16, TEAL, True)
    add_text(slide, "Period 決定時間尺度；GPU behavior 決定 workload-dependent hardware scaling。", 1.04, 4.48, 10.85, .42, 22, WHITE, True, PP_ALIGN.CENTER)
    add_text(slide, "下一步：第三硬體節點 + 新 workload family + HPCT 對齊可行性實驗 + uncertainty gate。", 1.04, 5.17, 10.85, .38, 14, MUTED, False, PP_ALIGN.CENTER)
    add_text(slide, "Q & A", 5.6, 6.35, 2.1, .42, 20, TEAL, True, PP_ALIGN.CENTER)

    prs.core_properties.title = "Part 7–8 Marker-Free Cross-GPU Runtime Prediction"
    prs.core_properties.subject = "YOLO26 heterogeneous GPU training runtime prediction"
    prs.core_properties.author = "YOLO26 Runtime Prediction Project"
    prs.core_properties.comments = "Generated from Part 7 and Part 8 experiment artifacts."
    prs.save(PPTX)
    print(PPTX)
    print(f"slides={len(prs.slides)}")


if __name__ == "__main__":
    build_deck()
