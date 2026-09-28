# 主要研究情境：Marker-free trace → runtime prediction

本文件是目前研究主線的權威入口。若其他 Notebook、GPU-metrics 模型、profiling-overhead 模型或早期 Part 1–8 實驗與本文件混淆，應以這裡列出的兩階段流程為主要使用情境；其他內容視為背景、比較、消融或延伸研究。

此主要情境目前聚焦 **RTX 5090 上 clean 與 high-load 兩種 condition 的 trace-based runtime prediction**，target 是該 condition 下未 profiling 的 steady-state per-iteration runtime。它不是早期 Part 1–8 所討論的 RTX 4090 ↔ RTX 5090 cross-device prediction，兩者請勿混稱。

## 主要流程

```text
Nsight Systems CUDA trace SQLite
        │
        ▼
Stage 1: marker-free period detector
analysis/high_load_01/evaluate_trace_event_periods.py
        │
        ├─ detections.csv
        ├─ adaptive_deployment.csv  ← Stage 2 的主要偵測輸入
        ├─ metrics.csv
        └─ experiment_contract.json
        │
        ▼
Stage 2: unified trace runtime model
analysis/high_load_01/run_unified_trace_model.py
        │
        ├─ samples.csv
        ├─ predictions.csv
        ├─ metrics.csv
        ├─ inner_alpha_selection.csv
        └─ report.md
```

## Stage 1：主要 marker-free detector

程式：`analysis/high_load_01/evaluate_trace_event_periods.py`

用途：從 Nsight Systems SQLite 的 CUDA kernel events 偵測 workload period，不使用 workload configuration、baseline runtime、GPU metrics、iteration counter 或 NVTX 作為 detector feature。

Detector 正式輸入：

- CUDA kernel start timestamp
- CUDA kernel short-name ID

`NVTX TRAIN_ITER` 只在 detection 完成後加入，用於離線 accuracy audit；它不是部署時的輸入，也不參與 adaptive selection。

主要交接產物是 `adaptive_deployment.csv`。Clean 與 high-load collections 應各自產生一份：

```powershell
python analysis/high_load_01/evaluate_trace_event_periods.py `
  --cache-root analysis_cache/nsys/RTX5090 `
  --output-dir analysis/clean_rtx5090/trace_event_period

python analysis/high_load_01/evaluate_trace_event_periods.py `
  --cache-root analysis/high_load_01/cache/nsys_trace_events/RTX5090 `
  --output-dir analysis/high_load_01/trace_event_period
```

## Stage 2：主要 runtime prediction model

程式：`analysis/high_load_01/run_unified_trace_model.py`

用途：讀取 Stage 1 的 clean/high-load `adaptive_deployment.csv`，結合 marker-free trace-derived features，執行 clean/high-load unified runtime prediction 評估。

```powershell
python analysis/high_load_01/run_unified_trace_model.py
```

預設輸出位置：`analysis/unified_trace_model/`

主要結果：

- `predictions.csv`：逐 workload out-of-fold 預測
- `metrics.csv`：各 scenario/model 的誤差摘要
- `samples.csv`：公開的模型輸入與 target 表格
- `inner_alpha_selection.csv`：nested validation tuning 紀錄
- `report.md`：clean/high-load 統一結果摘要
- `experiment_contract.json`：允許及禁止使用的特徵契約

這支程式目前是研究評估 pipeline，包含 grouped leave-one-workload-out、消融及 regime variants；它不是單筆線上 inference CLI。若要部署，需要再從選定 scenario 匯出固定模型與獨立 inference entrypoint。

## 與其他研究的界線

以下內容仍有研究價值，但不是本文件定義的主要使用情境：

- GPU-metrics marker-free detectors
- YOLO single-device profiling-overhead correction
- Direct dry-runtime models
- Part 1–6 的 oracle、metadata、短時間窗及 profiler ablations
- DNN、Gemma 4、ViT 的跨 workload/generalization 延伸實驗
- HPCToolkit 對齊與輔助分析

引用主要成果時，請明確寫成：**CUDA trace event-based marker-free period detection followed by unified clean/high-load trace runtime prediction**。
