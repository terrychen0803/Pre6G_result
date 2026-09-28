# Validation status

驗證日期：2026-09-28

## 已通過

- 108 個 Python 檔案可由 Python AST parser 成功解析。
- 10 個 Notebook 可成功解析為 JSON，且包含 `cells`。
- 所有 JSON 設定與結果契約可成功解析。
- 4 個 PowerShell 腳本通過 PowerShell parser。
- 主要 analysis config/default-path smoke test 通過。
- Pre6G experiment 的 energy integration 與 job rendering smoke test 通過。
- 發布副本沒有巢狀 `.git`、虛擬環境、cache directory 或禁止的 runtime artifact。
- 沒有超過 50 MiB 的檔案。
- 已移除原始工作電腦的絕對路徑；原始資料改用 `PRE6G_DATA_ROOT`。
- 4090/5090 重複 workload source 已以 SHA-256 比對後合併。

執行 repository-only validation：

```powershell
python scripts/validate_publication.py
```

## 尚未執行

- DNN、Gemma 4、ViT 的 NumPy/PyTorch 單元測試。
- 需要 pandas、scikit-learn、SciPy、Matplotlib 或 python-pptx 的分析執行測試。
- 使用完整原始 collection 的端到端重現。
- GPU profiler collection 測試。

目前可用的 Nsight 內建 Python 是精簡執行器，足以執行標準函式庫的發布檢查，但不包含 `unittest`、NumPy 或研究依賴。完整測試應依 `ENVIRONMENT.md` 建立正式 Python 環境後執行。

