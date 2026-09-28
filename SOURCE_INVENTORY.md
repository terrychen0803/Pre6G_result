# Source inventory

本文件記錄發布副本與原始研究工作區的對應關係。整理過程只複製檔案，沒有移動或刪除原始內容。

## 已納入

| 發布位置 | 原始位置 | 內容 |
|---|---|---|
| `docs/research_handoff.md` | `ACADEMIC_RESEARCH_HANDOFF.md` | 完整研究交接文件 |
| `analysis/` | `analysis/` | Python、PowerShell、報告及小型衍生表格；排除 cache、SQLite、NPZ |
| `configs/yolo26/` | `configs/` | YOLO26 實驗矩陣與軟體版本 |
| `hardware/` | `hardware/` | RTX 4090/5090 硬體與環境描述 |
| `notebooks/` | `notebooks/` | Part 1–8、研究總覽與結果摘要 |
| `analysis_outputs/` | `analysis_outputs/` | 精簡表格、圖片、報告、簡報與小型模型 |
| `workloads/yolo26/` | `scripts/` | YOLO26 收集與執行程式 |
| `workloads/dnn/` | `Pre6G_5090_dnn_gemma_vit/dnn_runtime_prediction/` | DNN 共用程式與設定 |
| `workloads/gemma4/` | `Pre6G_5090_dnn_gemma_vit/gemma4_runtime_prediction/` | Gemma 4 共用程式與設定 |
| `workloads/vit/` | `Pre6G_5090_dnn_gemma_vit/vit_runtime_prediction/` | ViT 共用程式與設定 |
| `tools/pre6g_experiment/` | `Pre6G_experiment/` | 實驗協調工具；排除其 Git metadata、測試依賴與 generated artifacts |

RTX 4090 與 RTX 5090 的 DNN、Gemma 4、ViT source-like files 經 SHA-256 比對內容相同，因此發布副本只保留一份共用程式。

## 未納入 Git 發布副本

| 原始位置 | 約略大小 | 原因 |
|---|---:|---|
| `analysis_cache/` | 27.4 GB | 可重建的分析快取、SQLite exports、工具與安裝檔 |
| `RTX4090_yolo26_collection_20260817/` | 11.0 GB | 原始實驗與 profiling 資料 |
| `RTX5090_yolo26_collection_20260817/` | 14.2 GB | 原始實驗與 profiling 資料 |
| `RTX4090_hpctoolkit_postprocess/` | 0.6 GB | HPCToolkit 衍生資料 |
| `RTX5090_hpctoolkit_postprocess/` | 0.6 GB | HPCToolkit 衍生資料 |
| `high_load_01_RTX5090_20260923/` | 1.6 GB | 高負載原始 runs；分析程式及精簡結果已另行納入 |
| `NsightSystems-2026.4.1.191-3860507/` | 1.5 GB | 第三方工具完整解包內容 |
| `Pre6G_4090_dnn_gemma_vit/` runtime content | 22.9 GB | `.venv`、Hugging Face cache、runs、資料及重複程式 |
| `Pre6G_5090_dnn_gemma_vit/` runtime content | 28.2 GB | `.venv`、Hugging Face cache、runs、資料及重複程式 |

## 目前發布副本統計

- 479 個檔案
- 約 12.55 MB
- 108 個 Python 程式
- 10 個 Notebook
- 173 個 CSV 結果表格
- 無超過 50 MB 的檔案
- 無巢狀 `.git` 目錄
