# Pre6G Runtime Prediction Research

本 repository 整理跨異質 GPU training runtime prediction 的研究程式、實驗設計、分析結果與可重現文件。

研究主線是：僅使用來源 GPU 上短時間、marker-free 的 profiling 訊號，建立 workload behavior features，預測相同 workload 在另一張 GPU 上的 steady-state per-iteration runtime。

## 研究範圍

- 硬體：NVIDIA RTX 4090、RTX 5090
- Workloads：YOLO26、DNN、Gemma 4、ViT
- Profilers：NVIDIA Nsight Systems、HPCToolkit
- 核心問題：cross-device runtime scaling、profiling overhead、marker-free period detection、runtime prediction
- 主要研究演進：Part 1 至 Part 8

完整研究背景、方法、結果與限制請先閱讀 [`docs/research_handoff.md`](docs/research_handoff.md)。

## 建議閱讀順序

1. `docs/research_handoff.md`：研究問題、方法、結果及限制。
2. `notebooks/00_research_overview.ipynb`：研究全貌。
3. `notebooks/part01` 至 `notebooks/part08`：依序查看實驗演進。
4. `notebooks/99_final_results_summary.ipynb`：最終結果摘要。
5. `analysis/README.md`：分析 pipeline 與主要指令。
6. `REPRODUCIBILITY.md`：資料到結果的重現順序。

環境安裝與外部資料設定請參考 [`ENVIRONMENT.md`](ENVIRONMENT.md)。

## 目錄

```text
analysis/       特徵抽取、模型、評估與報告程式
configs/        實驗與軟體環境設定
data/           資料取得方式、manifest 與小型範例
docs/           研究設計與交接文件
hardware/       RTX 4090/5090 硬體描述
notebooks/      Part 1–8 及最終摘要 Notebook
analysis_outputs/ 精簡表格、圖、報告、簡報與小型模型
tools/          Pre6G 實驗協調工具
workloads/      YOLO26、DNN、Gemma 4、ViT 收集程式
```

## 資料政策

大型原始 profiling reports、SQLite exports、模型下載、虛擬環境、工具安裝程式與分析快取不納入 Git。這些項目可重新產生或下載，且多數不適合一般 GitHub repository。

詳見 [`data/README.md`](data/README.md) 與 [`SOURCE_INVENTORY.md`](SOURCE_INVENTORY.md)。

## 目前狀態

這是從原始研究工作區建立的發布候選副本。程式、Notebook、精簡結果、可攜路徑、環境文件與 repository-only validation 已完成整理。需要 GPU／NumPy 等依賴的完整單元測試及原始資料端到端重跑，仍須在正式研究環境執行。

目前驗證狀態請參考 [`VALIDATION.md`](VALIDATION.md)。
