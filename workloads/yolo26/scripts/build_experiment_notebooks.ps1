param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$NotebookRoot = Join-Path $ProjectRoot "notebooks"

function New-MarkdownCell([string]$Text) {
    return [ordered]@{
        cell_type = "markdown"
        metadata = @{}
        source = @($Text -split "(?<=`n)")
    }
}

function New-CodeCell([string]$Code) {
    return [ordered]@{
        cell_type = "code"
        execution_count = $null
        metadata = @{}
        outputs = @()
        source = @($Code -split "(?<=`n)")
    }
}

function New-Notebook([string]$RelativePath, [array]$Cells) {
    $target = Join-Path $NotebookRoot $RelativePath
    $parent = Split-Path -Parent $target
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $notebook = [ordered]@{
        cells = $Cells
        metadata = [ordered]@{
            kernelspec = [ordered]@{
                display_name = "Python 3"
                language = "python"
                name = "python3"
            }
            language_info = [ordered]@{
                name = "python"
                version = "3"
            }
        }
        nbformat = 4
        nbformat_minor = 5
    }
    $json = $notebook | ConvertTo-Json -Depth 100
    [System.IO.File]::WriteAllText($target, $json, [System.Text.UTF8Encoding]::new($false))
}

function Get-ReportSnapshot([string]$RelativePath) {
    $path = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path -LiteralPath $path)) {
        return "> 建立 Notebook 時找不到報告：``$RelativePath``"
    }
    $body = [System.IO.File]::ReadAllText($path, [System.Text.Encoding]::UTF8)
    return "## 既有最終報告快照`n`n> 來源：``$RelativePath``。這是建立 Notebook 時嵌入的結果快照；重新執行分析後，請以磁碟上的最新報告為準。`n`n$body"
}

$setupCode = @'
from pathlib import Path
import json
import subprocess
import sys

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display


def find_project_root() -> Path:
    """Find the repository whether Jupyter starts at the repo or notebook folder."""
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "analysis").is_dir() and (candidate / "analysis_outputs").is_dir():
            return candidate
    raise FileNotFoundError("找不到專案根目錄；請從本 repository 內啟動 Jupyter。")


PROJECT_ROOT = find_project_root()
OUTPUT_ROOT = PROJECT_ROOT / "analysis_outputs"
RUN_EXPENSIVE = False  # 明確改成 True 才會重跑分析腳本。


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def load_csv(relative: str, **kwargs) -> pd.DataFrame:
    path = project_path(relative)
    if not path.exists():
        raise FileNotFoundError(f"缺少資料：{path}")
    return pd.read_csv(path, **kwargs)


def show_markdown(relative: str) -> None:
    path = project_path(relative)
    display(Markdown(path.read_text(encoding="utf-8")))


def inspect_inputs(paths) -> pd.DataFrame:
    rows = []
    for relative in paths:
        path = project_path(relative)
        rows.append({
            "path": relative,
            "exists": path.exists(),
            "type": "directory" if path.is_dir() else "file",
            "size_mib": round(path.stat().st_size / 1024**2, 3) if path.is_file() else None,
        })
    return pd.DataFrame(rows)


def run_commands(commands) -> None:
    """Show commands by default; run only after explicit opt-in."""
    for args in commands:
        command = [sys.executable, *args]
        print(" ".join(map(str, command)))
        if RUN_EXPENSIVE:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


print(f"PROJECT_ROOT = {PROJECT_ROOT}")
print(f"RUN_EXPENSIVE = {RUN_EXPENSIVE}")
'@

$metricsCode = @'
metrics = load_csv(METRICS_PATH)
sort_column = "mape_percent" if "mape_percent" in metrics.columns else "mean_period_ape_percent"
display(metrics.sort_values(sort_column).reset_index(drop=True))

if "direction" in metrics.columns and "model" in metrics.columns:
    plot_data = metrics.sort_values(sort_column).groupby("direction", as_index=False).first()
    ax = plot_data.plot.bar(x="direction", y=sort_column, legend=False, figsize=(8, 4))
    ax.set_ylabel(sort_column)
    ax.set_title("各方向最佳列（依表內指標排序）")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.show()
'@

$predictionCode = @'
predictions = load_csv(PREDICTIONS_PATH)
display(predictions.sort_values("ape_percent", ascending=False).head(15).reset_index(drop=True))

required = {"actual_ms", "predicted_ms", "ape_percent"}
if required.issubset(predictions.columns):
    view = predictions.copy()
    if "deployment_valid" in view.columns:
        view = view[view["deployment_valid"].astype(str).str.lower().eq("true")]
    ax = view.plot.scatter(x="actual_ms", y="predicted_ms", c="ape_percent", cmap="viridis", figsize=(6, 5))
    low = min(view["actual_ms"].min(), view["predicted_ms"].min())
    high = max(view["actual_ms"].max(), view["predicted_ms"].max())
    ax.plot([low, high], [low, high], "--", color="black", linewidth=1)
    ax.set_title("OOF prediction：actual vs predicted")
    plt.tight_layout()
    plt.show()
'@

$partSpecs = @(
    [ordered]@{
        Number = 1; Slug = "part01"; File = "01_cross_device_runtime_scaling.ipynb"
        Title = "來源 Ground-truth Runtime 的跨裝置 Scaling"
        Question = "已知來源 GPU 的真實 iteration runtime 時，能否預測另一張 GPU 的 runtime？"
        Conclusion = "最佳 combined log_source_ols：MAPE 4.21%、R² 0.9843。這是 feasibility baseline，不是 production solution，因為部署時仍需先取得來源真實 runtime。"
        Inputs = @("analysis_outputs/ground_truth.csv", "analysis_outputs/cross_device_samples.csv")
        Commands = @("analysis/run_baseline_models.py")
        Metrics = "analysis_outputs/model_experiments/part1_baselines/metrics.csv"
        Predictions = "analysis_outputs/model_experiments/part1_baselines/predictions.csv"
        Report = "analysis_outputs/model_experiments/part1_baselines/report.md"
        Steps = "1. 讀取兩張 GPU 的 baseline ground truth。`n2. 依 workload 建立雙向 source→target 樣本。`n3. 使用 direction-specific leave-one-workload-out。`n4. 比較固定比例、線性與 log-source 模型。`n5. 彙整 OOF error 與方向別結果。"
    },
    [ordered]@{
        Number = 2; Slug = "part02"; File = "01_workload_metadata.ipynb"
        Title = "加入 Workload Metadata"
        Question = "batch、image size、AMP 與 interaction terms 能否補足固定跨 GPU scaling？"
        Conclusion = "最佳 combined Huber interactions：MAPE 3.75%。仍使用來源真實 runtime，且假設 metadata 可得，因此不符合最嚴格黑箱部署。"
        Inputs = @("analysis_outputs/ground_truth.csv", "analysis_outputs/workload_features.csv", "analysis_outputs/cross_device_samples.csv")
        Commands = @("analysis/build_workload_features.py", "analysis/run_workload_models.py")
        Metrics = "analysis_outputs/model_experiments/part2_workload_models/metrics.csv"
        Predictions = "analysis_outputs/model_experiments/part2_workload_models/predictions.csv"
        Report = "analysis_outputs/model_experiments/part2_workload_models/report.md"
        Steps = "1. 從實驗設定建立 workload metadata。`n2. 與 source runtime、target label 合併。`n3. 在 outer LOOW 內做模型與超參數選擇。`n4. 比較 Ridge、Huber、main effects 與 interactions。`n5. 檢查兩個方向的 OOF error。"
    },
    [ordered]@{
        Number = 3; Slug = "part03"; File = "01_nsys_gpu_metrics.ipynb"
        Title = "Nsight GPU Metrics 跨裝置模型"
        Question = "不使用來源 baseline runtime，只使用 marker-aligned GPU behavior，能否預測 target runtime？"
        Conclusion = "compact_gpu_ridge 使用 15 個 features，combined MAPE 3.70%、R² 0.9915。它證明 GPU behavior 有訊號，但 segmentation 仍依賴 marker。"
        Inputs = @("analysis_outputs/nsys_gpu_metrics_features.csv", "analysis_outputs/workload_features.csv", "hardware/hardware_descriptors.csv")
        Commands = @("analysis/extract_nsys_gpu_metrics.py", "analysis/run_profile_models.py")
        Metrics = "analysis_outputs/model_experiments/part3_profile_models/metrics.csv"
        Predictions = "analysis_outputs/model_experiments/part3_profile_models/predictions.csv"
        Report = "analysis_outputs/model_experiments/part3_profile_models/report.md"
        Steps = "1. 從 Nsight SQLite 匯出 GPU metric samples。`n2. 以 marker 對齊 steady window。`n3. 建立 static、duration 與 compact GPU features。`n4. 在 nested direction-specific LOOW 中選 Ridge alpha。`n5. 只以 target baseline 作 label，產生 OOF predictions。"
    },
    [ordered]@{
        Number = 4; Slug = "part04"; File = "01_trace_and_hpctoolkit_ablation.ipynb"
        Title = "Nsight Trace 與 HPCToolkit Proxy 消融"
        Question = "加入 CUDA trace summaries 或 HPCToolkit artifact proxies，是否穩定優於 compact GPU metrics？"
        Conclusion = "沒有穩定超過 Part 3 的 3.70% MAPE。這不能證明 HPCToolkit 無效，只表示現有 proxy 粒度與樣本量不足。"
        Inputs = @("analysis_outputs/nsys_trace_features.csv", "analysis_outputs/hpctoolkit_postprocess_inventory.csv", "analysis_outputs/nsys_gpu_metrics_features.csv")
        Commands = @("analysis/extract_nsys_trace.py", "analysis/inspect_hpctoolkit_postprocess.py", "analysis/run_trace_hpct_models.py")
        Metrics = "analysis_outputs/model_experiments/part4_trace_hpct_models/metrics.csv"
        Predictions = "analysis_outputs/model_experiments/part4_trace_hpct_models/predictions.csv"
        Report = "analysis_outputs/model_experiments/part4_trace_hpct_models/report.md"
        Steps = "1. 彙整 CUDA kernel、API 與 memcpy trace。`n2. 盤點 HPCToolkit postprocess artifacts。`n3. 建立 trace／proxy feature families。`n4. 與 Part 3 compact GPU baseline 做同 fold 消融。`n5. 比較增量訊號與 variance。"
    },
    [ordered]@{
        Number = 5; Slug = "part05"; File = "01_finalization_bootstrap_slices.ipynb"
        Title = "模型封裝、Bootstrap 與 Slice 分析"
        Question = "如何把 marker-aligned compact model 工程化，並量化不確定性與 workload slices？"
        Conclusion = "Part 5 封裝 Part 3 模型、bootstrap interval、feature stability 與 slice metrics；定位仍是 marker-aligned oracle model。"
        Inputs = @("analysis_outputs/model_experiments/part3_profile_models/metrics.csv", "analysis_outputs/model_experiments/part3_profile_models/predictions.csv", "analysis_outputs/cross_device_samples.csv")
        Commands = @("analysis/finalize_model_experiment.py")
        Metrics = "analysis_outputs/model_experiments/part5_final/stage_comparison.csv"
        Predictions = $null
        Report = "analysis_outputs/model_experiments/part5_final/final_report.md"
        Steps = "1. 固定 Part 3 compact feature contract。`n2. 比較 Part 1–3 階段結果。`n3. 對 OOF errors 做 bootstrap uncertainty。`n4. 依 workload slice 檢查誤差。`n5. 訓練方向別 deployable bundles 並輸出 manifest。"
    },
    [ordered]@{
        Number = 6; Slug = "part06"; File = "01_short_window_profiling.ipynb"
        Title = "短時間 Marker-aligned Profiling Horizon"
        Question = "只看 warmup 後前 K 個已知 iterations，短 profiling 是否足以預測？"
        Conclusion = "K=50 的 behavior_mean_duration combined MAPE 約 3.77%。短 prefix 有效，但 K 與 iteration boundaries 仍由 marker 提供。"
        Inputs = @("analysis_outputs/short_window_profile_features.csv", "analysis_outputs/nsys_gpu_metrics_long.csv", "analysis_outputs/iteration_validation.csv")
        Commands = @("analysis/extract_short_window_features.py", "analysis/run_short_window_models.py", "analysis/summarize_short_window_results.py")
        Metrics = "analysis_outputs/model_experiments/part6_short_window/metrics.csv"
        Predictions = "analysis_outputs/model_experiments/part6_short_window/predictions.csv"
        Report = "analysis_outputs/model_experiments/part6_short_window/report.md"
        Steps = "1. 以 marker 取得 warmup 後 iteration boundaries。`n2. 截取 K=5、10、20、50、108 的 prefix。`n3. 建立 behavior、distribution 與 duration features。`n4. 在 nested LOOW 中評估各 horizon。`n5. 比較精度、profile cost 與方向差異。"
    },
    [ordered]@{
        Number = 7; Slug = "part07"; File = "01_marker_free_period_detection.ipynb"
        Title = "Marker-free 週期偵測"
        Question = "只用五條共同 GPU metric time series，能否在 5–15 秒 prefix 內找出 iteration-like period？"
        Conclusion = "Adaptive policy：48/48 在 10% period error 內；combined mean APE 1.01%、median 0.81%，平均 emission 10.06 秒。"
        Inputs = @("analysis_outputs/marker_free_timeseries_manifest.csv", "analysis_cache/marker_free_timeseries", "analysis_outputs/iteration_validation.csv")
        Commands = @("analysis/extract_marker_free_timeseries.py", "analysis/detect_marker_free_cycles.py", "analysis/evaluate_marker_free_cycles.py")
        Metrics = "analysis_outputs/model_experiments/part7_marker_free_cycles/adaptive_metrics.csv"
        Predictions = $null
        Report = "analysis_outputs/model_experiments/part7_marker_free_cycles/report.md"
        Steps = "1. 匯出 marker-free 1 ms GPU metric time series。`n2. 做 late-tail、robust normalization 與多尺度重採樣。`n3. 以 ACF peak、prominence 與 channel agreement 產生候選。`n4. 僅在 training workloads 選 adaptive threshold。`n5. 評估時才讀 oracle marker 計算 period APE。"
    },
    [ordered]@{
        Number = 8; Slug = "part08"; File = "01_marker_free_runtime_prediction.ipynb"
        Title = "Marker-free Behavior 到 Target Runtime"
        Question = "將 Part 7 偵測到的週期重整成 behavior features 後，能否預測另一張 GPU 的 baseline runtime？"
        Conclusion = "Nested direction-specific LOOW：combined MAPE 4.50%、median APE 2.68%、R² 0.9614。這是目前最接近部署情境的 YOLO 結果。"
        Inputs = @("analysis_outputs/marker_free_cycle_features.csv", "analysis_outputs/model_experiments/part7_marker_free_cycles/deployment_oof_detections.csv", "hardware/hardware_descriptors.csv")
        Commands = @("analysis/extract_marker_free_cycle_features.py", "analysis/run_marker_free_runtime_models.py")
        Metrics = "analysis_outputs/model_experiments/part8_marker_free_runtime/metrics.csv"
        Predictions = "analysis_outputs/model_experiments/part8_marker_free_runtime/predictions.csv"
        Report = "analysis_outputs/model_experiments/part8_marker_free_runtime/report.md"
        Steps = "1. 讀取 deployment-safe Part 7 detections。`n2. 依 detected period 將訊號切成固定長度 blocks。`n3. 建立 83 個 marker-free behavior features。`n4. 排除 oracle、target 與 leakage-prone 欄位。`n5. 在 nested direction-specific LOOW 中選 feature family 與 Ridge alpha。"
    }
)

$overviewCells = @(
    (New-MarkdownCell @'
# YOLO26 跨 GPU Runtime Prediction：實驗 Notebook 導覽

本目錄將既有 Part 1–8 實驗整理成可閱讀、可檢查與可選擇重跑的 Notebook。原始腳本、資料與既有輸出均保留在原位置。

## 使用原則

- Notebook 預設只讀取既有結果，不重跑昂貴分析。
- 每份 Notebook 的 `RUN_EXPENSIVE = False` 是安全開關；只有確認環境、輸入與輸出位置後才改成 `True`。
- 既有報告會以快照嵌入 Notebook，因此未執行 cell 也能閱讀主要結果。
- Part 1–6 多半是 oracle／marker-aligned benchmark；Part 7–8 才逐步接近 marker-free deployment。
'@),
    (New-CodeCell $setupCode),
    (New-MarkdownCell @'
## 研究演進

| Part | 核心問題 | 部署定位 |
|---:|---|---|
| 1 | source runtime 是否能預測 target runtime | feasibility baseline |
| 2 | workload metadata 是否改善 scaling | idealized、需要 metadata |
| 3 | GPU behavior 是否足以預測 | marker-aligned oracle |
| 4 | trace／HPCToolkit proxy 是否有增益 | marker-aligned ablation |
| 5 | 模型封裝、不確定性與 slice | oracle model engineering |
| 6 | 短 prefix 是否足夠 | short-horizon oracle |
| 7 | 無 marker 找週期 | deployment-safe detector |
| 8 | 無 marker 預測 target runtime | 目前主要部署估計 |
'@),
    (New-CodeCell @'
core_inputs = [
    "analysis_outputs/collection_inventory.csv",
    "analysis_outputs/ground_truth.csv",
    "hardware/hardware_descriptors.csv",
    "configs/experiments.csv",
]
display(inspect_inputs(core_inputs))
display(load_csv("analysis_outputs/ground_truth.csv").head())
display(load_csv("hardware/hardware_descriptors.csv"))
'@),
    (New-MarkdownCell "## 建議閱讀順序`n`n依序閱讀 Part 1–8，最後開啟 `99_final_results_summary.ipynb` 比較研究演進。若只關心目前部署結果，可直接閱讀 Part 7、Part 8 與總結。")
)
New-Notebook "00_research_overview.ipynb" $overviewCells

foreach ($part in $partSpecs) {
    $inputLiteral = ($part.Inputs | ForEach-Object { '    "' + $_ + '",' }) -join "`n"
    $commandLiteral = ($part.Commands | ForEach-Object { '    ["' + $_ + '"],' }) -join "`n"
    $pathVars = "METRICS_PATH = `"$($part.Metrics)`""
    if ($part.Predictions) {
        $pathVars += "`nPREDICTIONS_PATH = `"$($part.Predictions)`""
    }
    $intro = @"
# Part $($part.Number)：$($part.Title)

## 研究問題

$($part.Question)

## 目前結論

$($part.Conclusion)

> 本 Notebook 新增於既有研究產物之外，不會取代原始 Python scripts 或 CSV／Markdown 報告。
"@
    $inputCode = @"
INPUTS = [
$inputLiteral
]
display(inspect_inputs(INPUTS))
"@
    $commandCode = @"
COMMANDS = [
$commandLiteral
]
run_commands(COMMANDS)
"@
    $cells = @(
        (New-MarkdownCell $intro),
        (New-CodeCell $setupCode),
        (New-MarkdownCell "## 實驗步驟`n`n$($part.Steps)"),
        (New-MarkdownCell "## 輸入與 provenance`n`n下列 cell 只檢查輸入是否存在，不修改資料。"),
        (New-CodeCell $inputCode),
        (New-MarkdownCell "## 重現入口`n`n下列命令依現有 pipeline 排列。預設只列印；將 `RUN_EXPENSIVE` 明確改成 `True` 才會執行。"),
        (New-CodeCell $commandCode),
        (New-MarkdownCell "## 結果表"),
        (New-CodeCell ($pathVars + "`n`n" + $metricsCode))
    )
    if ($part.Predictions) {
        $cells += New-MarkdownCell "## OOF 預測與 worst cases`n`n顯示現有 prediction table 中 APE 最大的列，並提供 actual-vs-predicted 圖。"
        $cells += New-CodeCell ($pathVars + "`n`n" + $predictionCode)
    }
    if ($part.Number -eq 5) {
        $cells += New-CodeCell @'
for relative in [
    "analysis_outputs/model_experiments/part5_final/bootstrap_intervals.csv",
    "analysis_outputs/model_experiments/part5_final/slice_metrics.csv",
    "analysis_outputs/model_experiments/part5_final/feature_stability.csv",
    "analysis_outputs/model_experiments/part5_final/final_alpha_selection.csv",
]:
    print(relative)
    display(load_csv(relative))
'@
    }
    if ($part.Number -eq 6) {
        $cells += New-CodeCell @'
window_metrics = load_csv(METRICS_PATH)
best_by_window = (
    window_metrics.sort_values("mape_percent")
    .groupby(["direction", "window_iterations"], as_index=False)
    .first()
)
display(best_by_window)
best_by_window.pivot(index="window_iterations", columns="direction", values="mape_percent").plot(
    marker="o", figsize=(8, 4), ylabel="MAPE (%)", title="Best model by profiling horizon"
)
plt.tight_layout()
plt.show()
'@
    }
    if ($part.Number -eq 7) {
        $cells += New-CodeCell @'
detections = load_csv("analysis_outputs/model_experiments/part7_marker_free_cycles/deployment_oof_detections.csv")
evaluation = load_csv("analysis_outputs/model_experiments/part7_marker_free_cycles/oof_cycle_predictions.csv")
display(detections.head())
display(evaluation.sort_values("period_ape_percent", ascending=False).head(15))

metrics.set_index("device_id")[["mean_period_ape_percent", "median_period_ape_percent", "p90_period_ape_percent"]].plot.bar(
    figsize=(8, 4), ylabel="Period APE (%)", title="Adaptive marker-free period detection"
)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
'@
    }
    $cells += New-MarkdownCell (Get-ReportSnapshot $part.Report)
    $cells += New-MarkdownCell "## 解讀檢查清單`n`n- 區分 OOF evaluation 與重新用全部資料訓練的 production bundle。`n- 不把 target runtime、oracle marker 或 future information 當作 deployment feature。`n- 報告方向別結果、combined summary 與 worst cases。`n- 結論只適用於目前資料、GPU 節點與 workload family。"
    New-Notebook (Join-Path $part.Slug $part.File) $cells
}

$summaryCells = @(
    (New-MarkdownCell @'
# Part 1–8 最終結果總覽

這份 Notebook 集中比較研究問題如何從理想化跨裝置 scaling，逐步演進到 marker-free runtime prediction。數值以既有報告為準。

| Part | 主要結果 | 關鍵限制／定位 |
|---:|---|---|
| 1 | combined MAPE 4.21% | 需要 source ground-truth runtime |
| 2 | combined MAPE 3.75% | 仍需要 source runtime 與 metadata |
| 3 | combined MAPE 3.70% | marker-aligned steady window |
| 4 | 未穩定超越 Part 3 | HPCToolkit 只有 coarse proxy |
| 5 | bootstrap、slice、model bundle | 仍是 oracle segmentation |
| 6 | K=50 約 3.77% | iteration boundaries 由 marker 提供 |
| 7 | period mean APE 1.01%；48/48 within 10% | 目前限於既有 YOLO 分布 |
| 8 | combined MAPE 4.50%；R² 0.9614 | 目前最接近部署情境 |
'@),
    (New-CodeCell $setupCode),
    (New-CodeCell @'
metric_files = {
    "Part 1": "analysis_outputs/model_experiments/part1_baselines/metrics.csv",
    "Part 2": "analysis_outputs/model_experiments/part2_workload_models/metrics.csv",
    "Part 3": "analysis_outputs/model_experiments/part3_profile_models/metrics.csv",
    "Part 4": "analysis_outputs/model_experiments/part4_trace_hpct_models/metrics.csv",
    "Part 6": "analysis_outputs/model_experiments/part6_short_window/metrics.csv",
    "Part 7": "analysis_outputs/model_experiments/part7_marker_free_cycles/adaptive_metrics.csv",
    "Part 8": "analysis_outputs/model_experiments/part8_marker_free_runtime/metrics.csv",
}

inventory = []
for part, relative in metric_files.items():
    table = load_csv(relative)
    inventory.append({"part": part, "rows": len(table), "columns": len(table.columns), "path": relative})
display(pd.DataFrame(inventory))
'@),
    (New-MarkdownCell @'
## 最終研究判讀

1. Part 1–2 證明跨 GPU scaling 可預測，但依賴部署時不可得的來源真實 runtime。
2. Part 3–6 證明 GPU behavior 與短 prefix 有足夠訊號，但仍使用 marker／oracle segmentation。
3. Part 7 移除 marker，從 GPU metric time series 偵測 iteration-like period。
4. Part 8 使用 deployment-safe detections 建立 behavior features，是目前應優先引用的 runtime prediction 結果。
5. 外部效度仍受限於 RTX4090／RTX5090、YOLO26n 與 24 個 workloads；跨 workload family 結果應另行報告。
'@),
    (New-CodeCell @'
part8_predictions = load_csv("analysis_outputs/model_experiments/part8_marker_free_runtime/predictions.csv")
valid = part8_predictions[part8_predictions["deployment_valid"].astype(str).str.lower().eq("true")].copy()
display(valid.sort_values("ape_percent", ascending=False).head(20))

ax = valid.plot.scatter(x="actual_ms", y="predicted_ms", c="ape_percent", cmap="viridis", figsize=(7, 6))
low = min(valid.actual_ms.min(), valid.predicted_ms.min())
high = max(valid.actual_ms.max(), valid.predicted_ms.max())
ax.plot([low, high], [low, high], "--", color="black", linewidth=1)
ax.set_title("Part 8 deployment-valid OOF predictions")
plt.tight_layout()
plt.show()
'@),
    (New-MarkdownCell (Get-ReportSnapshot "analysis_outputs/model_experiments/part8_marker_free_runtime/report.md"))
)
New-Notebook "99_final_results_summary.ipynb" $summaryCells

$readme = @"
# Experiment notebooks

This directory contains a narrative notebook for each completed Part 1–8 experiment, plus an overview and final summary.

- Start with `00_research_overview.ipynb`.
- Read `part01` through `part08` in order.
- Use `99_final_results_summary.ipynb` for the cross-part conclusion.
- Expensive pipeline cells default to `RUN_EXPENSIVE = False`.
- Existing scripts and outputs remain authoritative and are not deleted or replaced.

The notebooks are generated by `scripts/build_experiment_notebooks.ps1` so their structure can be refreshed consistently after results change.
"@
[System.IO.File]::WriteAllText((Join-Path $NotebookRoot "README.md"), $readme, [System.Text.UTF8Encoding]::new($false))

Write-Output "Created experiment notebooks under $NotebookRoot"
