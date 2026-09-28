# Data availability

本 repository 不直接保存大型原始 profiling reports、Nsight SQLite exports、HPCToolkit databases、模型下載或完整 training runs。

## Repository 內提供的資料

- `../configs/`：實驗矩陣與軟體環境設定。
- `../hardware/`：兩張 GPU 的硬體及系統描述。
- `../analysis_outputs/`：可支援研究結論的精簡表格、圖與報告。
- `manifests/`：預留給原始資料檔名、大小、雜湊與來源紀錄。
- `samples/`：預留給可公開的小型輸入範例。

## 未直接提供的資料

- `.nsys-rep` 與 `.sqlite`
- HPCToolkit measurements/databases
- training checkpoints 與 Hugging Face model cache
- Python virtual environments
- Nsight Systems 安裝程式與解包檔案
- 可由分析流程重新產生的 cache

若要完整重現研究，需先取得原始 collection，依 manifest 放回指定位置，再執行 `REPRODUCIBILITY.md` 所列流程。

發布前仍需產生正式的原始資料 manifest，並決定大型資料要使用 GitHub Release、Git LFS、研究資料儲存庫或內部共享空間。
