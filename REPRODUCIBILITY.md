# Reproducibility guide

## 分析順序

目前研究 pipeline 的概念順序如下：

1. 準備 workload 與實驗矩陣。
2. 分別在 RTX 4090、RTX 5090 收集 baseline 與 profiling runs。
3. 掃描並驗證 run metadata。
4. 建立 steady-state ground truth。
5. 匯出 Nsight Systems 與 HPCToolkit 資料。
6. 抽取 trace、GPU metrics 與短時間窗特徵。
7. 建立 workload 及 hardware descriptors。
8. 執行 Part 1–6 基線與消融實驗。
9. 執行 Part 7 marker-free period detection。
10. 執行 Part 8 marker-free runtime prediction。
11. 產生表格、圖、報告與最終 Notebook。

## 原始分析入口

主要命令記錄在 `analysis/README.md`。典型順序為：

執行前請依 `ENVIRONMENT.md` 建立環境，並設定 `PRE6G_DATA_ROOT` 或在命令中傳入 `--runs-root`。

```powershell
python analysis/scan_runs.py
python analysis/aggregate_ground_truth.py
python analysis/analyze_iterations.py
python analysis/analyze_profiler_overhead.py
python analysis/export_nsys.py --mode nsys_trace --mode nsys_gpu_metrics
python analysis/extract_nsys_trace.py
python analysis/extract_nsys_gpu_metrics.py
python analysis/inspect_hpctoolkit_postprocess.py
python analysis/build_workload_features.py
python analysis/build_cross_device_dataset.py
python analysis/analyze_correlations.py
python analysis/generate_figures.py
python analysis/generate_report.py
```

## 發布前待驗證項目

- 產生大型原始資料 manifest。
- 為每一個主要 pipeline 階段提供最小測試資料。
- 從乾淨環境執行主要測試與至少一條端到端 smoke test。

目前這份文件先忠實記錄研究執行順序，不宣稱發布副本已可在沒有原始資料的情況下端到端重跑。
