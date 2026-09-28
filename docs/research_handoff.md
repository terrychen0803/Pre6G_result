# YOLO26 跨異質 GPU Training Runtime Prediction：研究交接文件

> **目前主要使用情境已更新：** 請優先閱讀 [`../PRIMARY_SCENARIO.md`](../PRIMARY_SCENARIO.md)。目前主線是 RTX 5090 clean/high-load 的 CUDA trace marker-free detector 與 unified runtime model；本文件中的跨 GPU Part 1–8 內容保留為研究演進、比較與延伸背景。

更新日期：2026-08-20  
原始工作區：YOLO26 runtime prediction research workspace

---

## 0. 文件用途

本文件整理目前專案的研究目標、資料收集方式、資料處理流程、Part 1–Part 8 實驗演進、Part 7 的 marker-free 週期偵測、Part 8 的跨裝置 per-iteration runtime 預測、實驗結果、資料洩漏界線、限制與後續研究方向。

這是一份可直接交給另一個 ChatGPT 或研究助理的自包含交接文件。後續討論應以本文件描述的「實際部署場景」為準，不能只依早期 Part 1–Part 6 的較理想化設定判斷系統是否可部署。

---

## 1. 執行摘要

### 1.1 最終研究問題

目標是在一個先前未見、且不能修改程式碼的 GPU training workload 進入系統後：

1. 在目前可用的來源 GPU 上短暫執行 profiling。
2. 不使用使用者程式中的 NVTX marker、Ultralytics callback、iteration counter、batch size、image size 或其他 workload metadata。
3. 僅從底層 profiler 收集的時間序列辨識訓練行為的週期。
4. 從偵測到的週期擷取 workload behavior features。
5. 結合來源／目標硬體資訊，預測同一個 workload 在另一張 GPU 上的 steady-state per-iteration runtime。

因此，本研究不是要等來源 workload 全部跑完後再做跨 GPU 換算，也不是把已知的來源 baseline iteration time 當作部署輸入。真正的輸入應是短時間、marker-free 的 profiling 訊號；輸出則是另一個裝置上的每次 iteration 時間。

### 1.2 目前最接近部署情境的成果

目前 Part 7 與 Part 8 最接近上述場景：

- Part 7：只讀 Nsight Systems GPU Metrics 的五條共同時間序列，自動偵測週期，不讀 marker 或 callback。
- Part 8：把 Part 7 的週期切成等長 blocks，建立 83 個行為特徵，再預測另一張 GPU 的 baseline per-iteration runtime。
- Part 8 使用 nested leave-one-workload-out 評估，每個測試 workload 在該次模型訓練與超參數選擇中完全未出現。
- 48 筆 out-of-fold 跨裝置預測的綜合 MAPE 約為 **4.50%**，但最大 APE 為 **23.19%**。

這表示目前已建立一個有用的初步 proof of concept，但還不能宣稱已解決一般性的 unseen workload、unseen GPU runtime prediction，原因包括：

- 只有 RTX 4090 與 RTX 5090 兩個裝置。
- 只有 24 個高度相關的 YOLO26n 設定，而非多種 model family。
- 來源與目標 GPU 分別位於不同整機，GPU 與 CPU／系統差異互相混雜。
- Part 7 雖然演算法不讀 marker，但原始 traces 是在 marker 存在時收集；仍需真正 no-marker recollection 驗證。
- Part 8 尚未訓練並序列化單一 production artifact；目前是嚴格的離線模型評估流程。

### 1.3 最重要的概念修正

Part 1–Part 6 中有些結果使用完成的來源 runtime、workload metadata 或 marker-aligned iteration boundary。這些實驗仍有價值，可當作可預測性分析、oracle benchmark 或消融研究，但不符合「不能修改使用者程式、事前不知道 iteration 邊界」的最終部署契約。

目前應把各階段定位為：

| 階段 | 主要用途 | 是否符合最終部署契約 |
|---|---|---:|
| Part 1–2 | 跨裝置 scaling 的可行性與理想化上限 | 否 |
| Part 3–5 | profiler feature 可預測性的 marker-aligned oracle benchmark | 否 |
| Part 6 | 短 profiling horizon 的 oracle segmentation benchmark | 否 |
| Part 7 | marker-free 週期偵測 | 大致符合；仍待真正 no-marker trace 驗證 |
| Part 8 | marker-free 特徵到 target runtime 的離線評估 | 目前最符合；尚未封裝 production model |

---

## 2. 部署契約與資料角色

### 2.1 部署時允許使用的資料

對一個新 workload，允許使用：

- 來源裝置短時間收集到的底層 profiling metrics。
- 從這些 metrics 自動估計的 period、confidence、profiling horizon 等資訊。
- 來源與目標裝置事先已知的硬體描述。
- 已經離線訓練好的模型參數與 preprocessing pipeline。

### 2.2 部署時禁止使用的資料

不應使用：

- NVTX iteration marker。
- Ultralytics callback 記錄的 iteration start/end。
- 由 wrapper 直接提供的 iteration count。
- 使用者 workload 的 batch size、image size、AMP 等設定，除非未來明確改變問題定義並允許使用。
- 新 workload 在來源 GPU 上跑完整個 benchmark 後得到的 ground-truth mean iteration time。
- 新 workload 在目標 GPU 上的任何 profiling 或 baseline 資料。

### 2.3 離線研究時可以使用但不可餵入模型的資料

NVTX marker、callback iteration timing 與 target baseline 可在離線資料集建立階段使用，但角色必須受限：

- marker/callback：驗證週期偵測是否正確。
- target baseline：作為 supervised learning label 與計算誤差的答案。
- workload metadata：用於資料分層、錯誤分析與消融，但目前 marker-free production feature schema 不使用。

這不是 data leakage，只要上述資訊沒有進入該測試樣本的 feature extraction、模型選擇或模型訓練。若 marker 被用來決定部署特徵的切割位置，或 target baseline 被拿來建特徵，就會違反部署假設。

---

## 3. 硬體與實驗節點

目前有兩個裝置／節點：

- NVIDIA GeForce RTX 4090 節點，CPU 為 Intel Core i7-13700K。
- NVIDIA GeForce RTX 5090 節點，CPU 為 AMD Ryzen 9 9950X。

硬體描述來自以下來源：

- `nvidia-smi`
- PyTorch CUDA device properties
- `lscpu` 或相對應的 CPU 資訊
- 系統記憶體資訊
- `nvcc`
- OS／kernel 資訊
- Python、PyTorch、CUDA、Ultralytics 等 software stack

整理後的硬體表位於：

- `hardware/hardware_descriptors.csv`

### 3.1 目前硬體設計的限制

兩張 GPU 不只 GPU 型號不同，host CPU 與整機環境也不同。因此模型看到的跨裝置 runtime 差異，可能同時包含：

- GPU compute throughput
- Tensor Core／架構差異
- GPU memory bandwidth 與容量
- PCIe link
- CPU data preparation／launch overhead
- system memory
- software／driver／CUDA stack 差異

目前不能把所有差異都因果歸因於 GPU。本文件所稱「跨 GPU」在實驗上更精確地說是「跨兩個 GPU 節點」。

此外，只有兩個硬體點時，無法嚴格驗證模型對第三個未見裝置的硬體泛化能力。Part 8 的 direction-specific model 中，硬體比例在同一預測方向內是常數，會被 variance filter 移除；所以目前的良好結果主要是 workload behavior 到固定裝置對 scaling 的學習，而不是完整的硬體性能模型。

---

## 4. Dataset 與 workload 設計

### 4.1 Synthetic object-detection dataset

為控制資料內容並降低外部變因，實驗使用合成 object-detection dataset：

- Training images：512 張
- Validation images：32 張
- 原始圖片大小：640 × 640
- 每張圖片：一個矩形物件
- 類別數：1
- 固定 seed：20260817
- 實際訓練的 `imgsz` 由 workload 設定決定，訓練時進行 resize

資料產生器可建立內容語意 hash，但目前 run metadata 沒有記錄可用的 dataset hash。現有 validation report 顯示 distinct recorded dataset hashes 為 0，因此尚無法從現有 metadata 做跨節點 dataset identity 的密碼學式確認。後續重跑應把 dataset manifest/hash 寫入每個 run metadata。

### 4.2 Workload matrix

設定檔：

- `configs/experiments.csv`

共 24 個 YOLO26n training workloads，為以下參數的 full factorial：

- batch size：4、8、16、32
- image size：320、480、640
- AMP：false、true

組合數：

```text
4 batch sizes × 3 image sizes × 2 AMP modes = 24 workloads
```

workload ID 為 C01–C24：

- C01–C12：AMP false
- C13–C24：AMP true

共同設定包括：

- model：`yolo26n.yaml`
- pretrained：false
- optimizer：AdamW
- workers：0
- cache：RAM
- validation/save/plots：disabled
- data augmentation：關閉
- deterministic seed：0
- compile：false
- nominal epochs：20
- callback 在 128 iterations 時停止
- warmup iterations：20
- steady-state iterations：108，也就是 iteration 21–128
- `nbs` 設成實際 batch size

runner 會檢查實際 batch size 是否等於設定值。

### 4.3 為什麼這樣設計

這個矩陣有意改變：

- batch size 對平行度、記憶體壓力與 kernel 組合的影響。
- image size 對 FLOPs、feature-map size 與 memory traffic 的影響。
- AMP 對 Tensor Core 使用、數值格式與吞吐量的影響。

優點是控制良好、組合完整；缺點是 24 個樣本共享同一個模型與資料型態，統計上高度相關，不能等同 24 個完全獨立的 application families。

---

## 5. 資料收集流程

### 5.1 一次 training run 的生命週期

每個 workload run 大致如下：

1. 載入指定的 YOLO26n 設定與 synthetic dataset。
2. 執行前 20 iterations 作為 warmup。
3. iteration 21 前做 CUDA synchronization，開始 steady window。
4. 執行 iteration 21–128，共 108 iterations。
5. iteration 128 後再次 CUDA synchronization，結束 steady window。
6. callback 停止訓練並輸出 timing metadata。

callback 同時可以記錄：

- host-side iteration timestamps
- CUDA event timing
- NVTX iteration ranges
- 完整 steady window wall-clock time

這些資料對 baseline label 與離線驗證有用，但 Part 7/8 的部署特徵不能依賴它們。

### 5.2 每個裝置的 run 數量

每個裝置收集：

| 類型 | 每 workload 次數 | 24 workloads 合計 |
|---|---:|---:|
| Baseline | 3 | 72 |
| Nsight Systems trace | 1 | 24 |
| Nsight Systems GPU metrics | 1 | 24 |
| Nsight Systems CPU metrics | 1 | 24 |
| HPCToolkit CUDA | 1 | 24 |
| 合計 | 7 | 168 |

兩個裝置合計 336 runs。

### 5.3 Ground truth 的建立

每個 baseline run 的主要 runtime 統計量為：

```text
steady_window_mean_iter_ms
= synchronized wall-clock duration of iterations 21–128 / 108
```

每個 workload/device 執行 3 次 baseline。最終 ground truth 是三次 `steady_window_mean_iter_ms` 的 median：

```text
ground_truth_iter_ms(workload, device)
= median(run_1_mean, run_2_mean, run_3_mean)
```

所以共有：

```text
24 workloads × 2 devices = 48 ground-truth labels
```

使用 baseline 而非 profiler run runtime 作為答案，是為了避免 profiler overhead 汙染目標值。

觀察到的 runtime 範圍：

- RTX 4090：30.278–131.249 ms/iteration
- RTX 5090：28.075–104.224 ms/iteration

RTX 5090 相對 RTX 4090 的 speedup：

- mean：約 1.119×
- median：約 1.104×
- range：約 1.036×–1.259×

speedup 隨 workload 改變，顯示單一固定 GPU ratio 不足以描述所有 workload。

---

## 6. Profiling modalities

### 6.1 Nsight Systems

Nsight runner 的共同設定包含：

```text
--trace=cuda,nvtx
--sample=none
--cpuctxsw=none
```

主要收集三種模式：

1. Trace：CUDA API、kernel、memory operations、NVTX 等 timeline 資料。
2. GPU metrics：GPU 各硬體單元的固定頻率時間序列。
3. CPU metrics：CPU PMU counters。

GPU metrics sampling frequency 為 1000 Hz，即約每 1 ms 一筆。共 144 份 Nsight reports 均可開啟與解析。

### 6.2 兩張 GPU 的 GPU metric schema 不同

RTX 4090 的 24 份 GPU metric reports 各有穩定的 35 個 metrics；RTX 5090 的 24 份 reports 各有穩定的 25 個 metrics。5090 的 25 個 metrics 全部包含在 4090 的 35 個 metrics 中，另外有 10 個 metrics 只出現在 4090。

這不是同一裝置上偶發掉資料，而是跨 GPU 架構、驅動、Nsight 版本或 auto metric-set 支援差異造成的 schema difference。若模型要跨裝置比較同一含義的訊號，必須：

- 只使用共同 channels；或
- 建立 architecture-aware metric ontology／映射；或
- 使用能處理 missing modalities 的模型。

Part 7 採保守作法，只使用兩張 GPU 都存在的五條 throughput channels：

- SM issue throughput
- SMs active throughput
- Tensor active throughput
- DRAM read bandwidth throughput
- DRAM write bandwidth throughput

### 6.3 CPU metrics 狀態

CPU PMU 資料目前不可用：

- RTX 4090 節點：24/24 reports 解析錯誤。
- RTX 5090 節點：report 內沒有預期資料表。
- 合計可用率：0/48。

這不代表所有 HPC/CPU 資料都不可用，而是這批 Nsight CPU metrics 無法進入目前模型。若後續要納入 CPU bottleneck，需要重新確認 PMU 權限、kernel 設定、Nsight 支援與 CPU 架構的 counter schema。

### 6.4 CUDA trace backend 差異

- RTX 4090：legacy software backend
- RTX 5090：hardware backend

這可能影響 trace overhead 與可見事件細節。跨裝置比較 profiler timing 時應保留此 caveat。

### 6.5 HPCToolkit

收集方式為 `hpcrun -e gpu=cuda`，兩張裝置各有 24 份 measurement，共 48 份。因原始資料需要 architecture-specific postprocessing，`hpcstruct`／`hpcprof` 分別在對應節點處理後，再下載到：

- `RTX4090_hpctoolkit_postprocess`
- `RTX5090_hpctoolkit_postprocess`

整理結果顯示：

- 48/48 有可用的 metadata/profile/CCT database。
- 沒有 trace database。
- 每個裝置整理出 65 種 metric taxonomy，catalog 合計 130 筆。
- 報告中存在 missing-correlation warnings。
- PC samples 為 0。

因此 HPCToolkit 並非「全部不可用」；它目前適合做 aggregated call-path、CUDA activity 與結構性特徵，但缺乏可直接按時間和 Part 7 每個週期對齊的 trace DB。Part 4 只使用了 artifact-level proxies，尚未充分開採 numeric CCT metrics。

---

## 7. Profiling 資料的整理與標準化

原始 profiler output 不能直接拿來建模，主要處理包括：

1. 驗證每個 device/workload/mode 的檔案是否存在。
2. 把 `.nsys-rep` 匯出或解析為 SQLite。
3. 盤點各 SQLite 的 tables、metric names、timestamp span 與 sample count。
4. 找出兩張 GPU 共同存在且所有 workload 都穩定提供的 channels。
5. 將硬體描述整理為統一 CSV schema。
6. 將 baseline 三次重跑彙整為 median ground truth。
7. 建立 run manifest，連結 device、workload、profiler mode、artifact 與 label。
8. 對 HPCToolkit postprocess 結果建立 inventory 與 metric taxonomy。
9. 在 Part 7 將 irregular／原生 metric samples 插值到 1 ms uniform grid。
10. 在 Part 8 將偵測區段重整為週期 blocks，建立定長 tabular features。

重要原則是：資料整理可保留 marker 供離線 audit，但 marker-free feature extractor 必須有明確程式邊界，不能讀取 marker table、callback JSON 或 target label。

---

## 8. Part 1–Part 8 實驗演進

## 8.1 Part 1：用來源 ground-truth runtime 做跨裝置 scaling

輸入包含來源裝置已完成 baseline 後的 ground-truth iteration time；輸出為另一張裝置的 ground-truth runtime。使用 direction-specific leave-one-workload-out。

最佳 combined `log_source_ols`：

- MAPE：4.21%
- Median APE：3.58%
- Max APE：11.97%
- R²：0.9843

意義：來源 runtime 對目標 runtime 有很強預測力，跨裝置 scaling 不是隨機的。

限制：部署時必須先在來源裝置把 baseline protocol 跑完並知道真正 iteration 平均值，與「只做短 profiling 且不知道 iteration」的目標矛盾。因此 Part 1 是 feasibility baseline，不是 production solution。

## 8.2 Part 2：加入 workload metadata

輸入包含來源 runtime、batch、image size、AMP 與 interaction terms，測試 Ridge／Huber 等模型。

最佳 combined Huber interactions：

- MAPE：3.75%
- RTX4090 → RTX5090：約 3.11%
- RTX5090 → RTX4090 的最佳設定為 ridge main：約 4.27%

意義：workload configuration 可以補充固定 scaling ratio 無法描述的差異。

限制：仍使用來源真實 runtime，且假設系統知道使用者的 workload metadata。這不符合目前最嚴格的黑箱部署設定。

## 8.3 Part 3：Nsight GPU metrics 模型

從 marker-aligned steady window 擷取 Nsight GPU metrics，使用 compact feature set 和 Ridge 預測 target baseline。來源 baseline runtime 不當 feature，target baseline 只作 label。

最佳 `compact_gpu_ridge` 使用 15 個 input features：

- Combined MAPE：3.70%
- Median APE：2.58%
- Max APE：10.74%
- R²：0.9915
- 4090 → 5090：3.14%
- 5090 → 4090：4.26%

意義：底層 GPU behavior 本身已包含可用的跨裝置 runtime 訊息。

限制：steady window 是靠 marker／callback 精準對齊，並非由 profiler signal 自動找到。應將它視為 oracle segmentation benchmark。

## 8.4 Part 4：Nsight trace 與 HPCToolkit proxy 消融

在 compact GPU metrics 之外，加入 Nsight trace summaries 與當時可取得的 HPCToolkit artifact proxies。

結果：沒有穩定超過 Part 3 的 3.70% MAPE。

解讀：不能直接推論 HPCToolkit 對此問題沒有價值。較合理的結論是當時使用的 proxy 粒度不足，且沒有 numeric CCT／time-aligned cycle features。增加高維或弱相關特徵，在只有 24 workloads 的情況下也容易增加 variance。

## 8.5 Part 5：模型封裝、bootstrap 與 slice 分析

Part 5 將 Part 3 的 compact model 做較完整的報告、bootstrap uncertainty 與 workload slice 分析。

目前定位：這是 marker-aligned oracle model 的工程化與穩健性分析，不應再描述為符合黑箱 workload 的 production model。

## 8.6 Part 6：短時間、marker-aligned profiling horizon

只取 warmup 後前 K 個已知 iterations，K = 5、10、20、50、108，建立 behavior／duration features，研究短 profiling 是否足夠。

最佳 combined `behavior_mean_duration` 在 K=50：

- MAPE：約 3.77%

部分方向結果：

| K | 4090→5090 MAPE | 5090→4090 MAPE |
|---:|---:|---:|
| 5 | 3.20% | 7.39% |
| 20 | 3.10% | 5.17% |
| 50 | 3.06% | 4.49% |
| 108 | 3.60% | 4.32% |

意義：不必等 workload 完整訓練完，短 prefix 的 profiling behavior 已能支持預測。

限制：K 與每個 iteration boundary 由 marker 提供，實務黑箱情境下不知道「已經跑了 K 個 iterations」。所以 Part 6 是 short-horizon oracle benchmark，促成 Part 7 的 marker-free detector。

## 8.7 Part 7：marker-free 週期偵測

只使用五條共同 GPU metric time series，在 5–15 秒 profiling prefix 中估計 iteration-like period。詳細方法見第 9 節。

adaptive policy 的 marker-blind detection 結果：

- Combined mean period APE：約 1.01%
- Median period APE：約 0.81%
- P90：約 1.86%
- Max：約 5.61%
- 48/48 在 10% 以內
- Mean emission time：約 10.06 秒

## 8.8 Part 8：marker-free behavior 到 target runtime 預測

把 Part 7 偵測到的區段依 period 重整成多個固定長度 blocks，建立 83 個 marker-free behavior features，再以 Ridge 預測另一裝置 baseline runtime。詳細方法見第 10 節。

Nested direction-specific LOOW combined 結果：

- MAPE：4.50%
- Median APE：2.68%
- MAE：2.37 ms
- RMSE：4.34 ms
- Max APE：23.19%
- R²：0.9614

這是目前最應被引用的部署情境估計，但仍只適用於目前兩張 GPU、YOLO26n workload family 與現有資料分布。

---

## 9. Part 7：marker-free period detection 詳解

### 9.1 輸入

每個 device/workload 的 Nsight Systems GPU metrics SQLite。Detector 不讀：

- NVTX tables
- callback timing
- baseline label
- batch/imgsz/AMP
- 已知 iteration count

### 9.2 共同時間軸與重採樣

對五條 channels 取共同可用的時間範圍，插值到 1 ms uniform time grid。這樣不同 channel 的 sample timestamps 才能一起做 correlation。

### 9.3 Profiling horizons

診斷用固定 horizons：

- 5 秒
- 7 秒
- 9 秒

最終 adaptive policy：

```text
先看 9 秒 → confidence 不足則看 12 秒 → 再不足則看 15 秒
```

### 9.4 Late-tail heuristic

每個 horizon 只保留末端的一部分，以減少最初 warmup/transient 的影響。候選 tail fractions：

- 0.50
- 0.65
- 0.80

需要強調：這是「假設後段較穩定」的 heuristic，不是真正學習式的 stationarity/change-point detector。未來可用變點偵測替代。

### 9.5 Robust normalization

每條 channel 分別標準化：

```text
z(t) = (x(t) - median(x)) / IQR(x)
```

使用 median/IQR 是為降低 spikes 或少數 outliers 對尺度的影響。

### 9.6 Autocorrelation

對每條 channel 以 FFT 高效率計算 normalized autocorrelation function，然後將五條 ACF 聚合為 consensus ACF。

搜尋 period 範圍：

```text
15 ms ≤ period ≤ 250 ms
```

候選 bin sizes：

- 1 ms
- 2 ms
- 4 ms

在 consensus ACF 上找 peaks：

- prominence 至少 0.02
- peaks 的最小距離約 8 ms

候選週期依 consensus ACF peak height 與 prominence 綜合排序。

### 9.7 多 channel agreement

對每個候選 period，再檢查每條 channel 自己的 ACF 是否也在附近有 peak。單 channel peak prominence 至少 0.01，容許誤差：

```text
tolerance = max(2 × bin_ms, 0.08 × candidate_period)
```

`channel_agreement` 是五條 channels 中支持候選週期的比例。

### 9.8 Confidence

目前 confidence 定義為：

```text
cycles_observed = segment_duration / detected_period

confidence = clip(prominence, 0, 1)
             × channel_agreement
             × min(1, cycles_observed / 5)
```

三個因子分別表示：

- peak 是否明顯。
- 多條硬體訊號是否一致支持。
- 區段內是否至少觀察到約五個週期。

這是 heuristic score，不是校準後的「正確機率」。例如 confidence 0.8 不能解讀為有 80% 機率正確。

### 9.9 Adaptive threshold 如何避免 leakage

在每個 outer fold 中，只用其他 23 個 workloads 的 detector 結果選 threshold。選擇目標是在 training workloads 上增加 coverage，同時要求被接受的 training detections period APE 均不超過 10%。

所有 outer folds 最後選到的 threshold 都是 0.4。測試 workload 的 marker answer 不參與 threshold 選擇。

### 9.10 Period ground truth 與 evaluation

Detector 完成後，才在獨立 evaluation code 中讀 NVTX，計算 steady iterations 21–128 的相鄰 start timestamps 差值中位數，作為 oracle period：

```text
oracle_period = median(start_time[i+1] - start_time[i])
```

Period APE：

```text
abs(detected_period - oracle_period) / oracle_period × 100%
```

marker 只用於回答「detector 準不準」，沒有被 detector 當輸入。

### 9.11 結果

Forced fixed-horizon：

| Horizon | Mean APE | Median APE | P90 | Max | ≤10% |
|---:|---:|---:|---:|---:|---:|
| 5 s | 94.44% | 5.75% | 262.54% | 286.56% | 52.1% |
| 7 s | 62.76% | 2.74% | 247.65% | 325.21% | 68.8% |
| 9 s | 13.21% | 1.13% | 32.79% | 212.49% | 83.3% |

短 horizon 的 median 已可能很好，但少數 harmonic aliasing 造成極端錯誤，因此 mean 與 max 很差。Adaptive confidence policy 的作用是讓不可靠的 9 秒偵測延長觀察：

| Device | N | Mean APE | Median | P90 | Max | Mean emission |
|---|---:|---:|---:|---:|---:|---:|
| RTX4090 | 24 | 1.17% | 0.99% | 2.65% | 5.61% | 10.625 s |
| RTX5090 | 24 | 0.86% | 0.77% | 1.56% | 2.25% | 9.500 s |
| Combined | 48 | 1.01% | 0.81% | 1.86% | 5.61% | 10.063 s |

Emission 分布：

- 9 秒：33 profiles
- 12 秒：13 profiles
- 15 秒：2 profiles

### 9.12 仍需注意的風險

- Detector 看不到 marker，但收集 trace 時 marker 確實存在。NVTX 本身一般不應改變 GPU metric pattern 太多，仍應以真正移除 marker/callback 的 run 做 domain validation。
- 週期可能是 iteration、半個 iteration、兩倍 iteration或某個 dominant sub-cycle。多 channel agreement 可降低但不能完全消除 harmonic aliasing。
- 目前 period search range 15–250 ms 是依現有 workload 設定；更快或更慢的模型可能超出範圍。
- Late-tail 不是穩態保證；更一般的 workload 需要在線 change-point／stationarity detection。

---

## 10. Part 8：marker-free runtime model 詳解

### 10.1 Part 7 的 period 如何轉成週期資料

Part 8 讀取 Part 7 的 deployment-safe detection 與 marker-free 1 ms NPZ time series。對 detector 選定的 segment：

```text
period_samples = detected_period_ms / raw_sampling_interval_ms
cycle_count = floor(segment_sample_count / period_samples)
```

保留末端恰好 `cycle_count × period_samples` 個 samples，再 reshape：

```text
[channel, cycle, sample_within_cycle]
```

目前並不知道真實 iteration 的 phase zero，因此這些 blocks 是依偵測 period 從固定端點反向排列，具有任意 phase offset。它們不是 callback 定義的真正 iteration slices。

目前特徵多採分布、頻譜與重複性統計，對固定 phase offset 相對不敏感；但 period 有小誤差時，多個 blocks 仍會逐漸 phase drift。未來可改用 fractional-period resampling、circular alignment 或 learned phase estimator。

### 10.2 83 個 behavior features

83 是 marker-free behavior predictor 的數量，不是 CSV 的全部欄位。

對五條 channels，每條建立 14 個特徵，共 70 個：

1. mean
2. standard deviation
3. coefficient of variation
4. P10
5. median
6. P90
7. active fraction（值大於 0.05 的比例）
8. cycle-mean CV
9. cycle-mean P90−P10 span
10. normalized cycle trend
11. mean cycle-template amplitude
12. first-harmonic energy fraction
13. second-harmonic energy fraction
14. cycle repeatability correlation

再加上：

- 五條 channels 兩兩相關係數：C(5,2) = 10 個
- 三個跨 channel summary features

總計：

```text
5 × 14 + 10 + 3 = 83 behavior features
```

這些特徵試圖描述：

- 平均資源利用程度。
- burstiness／變異。
- 各週期間穩定度。
- 週期內波形強度。
- dominant harmonics。
- SM、Tensor 與 DRAM 行為之間的協同關係。

完整 feature table 每列有 93 個欄位，因為還包括 device、workload、period、confidence、horizon、audit fields 等；不能把 93 全部誤認為模型 predictors。

### 10.3 Hardware features

候選硬體資訊使用 11 個 target/source log ratios：

- GPU compute capability
- SM count
- VRAM GiB
- default power limit
- maximum SM clock
- maximum memory clock
- negotiated PCIe generation
- negotiated PCIe width
- CPU physical core count
- CPU maximum MHz
- system memory

概念形式：

```text
hardware_feature_j = log(target_hardware_j / source_hardware_j)
```

但在 direction-specific model 內，所有 4090→5090 rows 的 hardware ratios 完全相同，所有 5090→4090 rows 也完全相同，因此 variance filter 會移除這些常數。它們目前沒有真正學到「硬體規格如何映射 runtime」。要驗證硬體 conditioned generalization，至少需第三個、最好更多 GPU 節點，並以 leave-one-device-out 評估。

### 10.4 明確排除的 features

Part 8 的 deployment-valid model 不使用：

- batch size
- image size
- AMP flag
- model name/config
- source baseline runtime
- target GPU metrics
- target profiling duration
- NVTX/callback boundaries
- target baseline，除作 label 外

### 10.5 目標值

每筆方向性樣本的 label 是同一 workload 在另一張 GPU 的 baseline ground truth：

```text
X = source-device short profiling features
y = target-device baseline ground_truth_iter_ms
```

例如：

```text
C10 / RTX4090 → RTX5090
input: C10 在 RTX4090 的 marker-free profiling behavior
label: C10 在 RTX5090 的 baseline median iteration ms
```

反方向則使用 RTX5090 profiling 預測 RTX4090 label。

### 10.6 Feature families

測試的主要 families：

- `behavior_only`
- `period_only`
- `period + compact behavior`（43 個 compact predictors）
- `period + full behavior`（83 個 behavior predictors）
- `period + compact behavior + quality`（加入 confidence、horizon 等 quality fields）
- oracle period-only：只作診斷對照，不是部署模型

### 10.7 Ridge pipeline

模型預測 log runtime：

```text
SimpleImputer(median)
→ VarianceThreshold
→ StandardScaler
→ Ridge(alpha)
→ predict log(target_runtime)
→ exp(prediction)
```

候選 alpha：

```text
0.01, 0.1, 1, 10, 100, 1000
```

使用 log runtime 是因為硬體 speedup／slowdown 更接近乘法 scaling，且可避免直接產生負 runtime。

### 10.8 Nested leave-one-workload-out 評估

Nested CV 的用途是誠實估計 unseen workload error，同時避免用測試 workload 選 feature family 或 alpha。

#### Outer loop

共 24 個 outer folds。以 C01 為例，整個 C01 在兩個方向都被排除：

```text
Outer training workloads: C02–C24
Outer test workload: C01
```

然後訓練兩個獨立方向模型：

1. 4090 source features → 5090 target runtime
2. 5090 source features → 4090 target runtime

所以每個 workload 產生兩筆完全 out-of-fold 預測：

```text
24 workloads × 2 directions = 48 OOF predictions
```

精確地說，primary evaluation 有 24 outer folds × 2 directions = 48 個最終 outer evaluation models，另外還有內層暫時訓練的 candidate models。先前若簡化說成「24 個模型」，指的是 24 個 held-out workload folds，而不是實際 fit 次數。

#### Inner loop

當 C01 是 outer test，inner GroupKFold 只能使用 C02–C24。這 23 個 workloads 分成 5 folds，每次約 18–19 個 inner training、4–5 個 inner validation。

對每個 feature family 與 alpha：

1. 只在 inner training fit imputer、variance filter、scaler、Ridge。
2. 預測 inner validation。
3. `exp()` 轉回 milliseconds。
4. 計算 validation APE/MAPE。
5. 合併五個 inner folds，選 validation MAPE 最低的 alpha。
6. 比較各 family 的最佳 inner MAPE，選 family。
7. 使用選定 family/alpha 在全部 C02–C24 重新 fit。
8. 最後只預測一次 C01。

C01 的 target runtime 從未用於選 alpha、feature family 或 preprocessing。

#### 一個實際例子

C01 / 4090→5090：

- Inner 選到 quality family、alpha = 0.1
- Inner MAPE：約 2.762%
- Actual target：29.0603 ms
- Prediction：27.2676 ms
- APE：約 6.169%

C01 / 5090→4090：

- Inner 選到 quality family、alpha = 10
- Inner MAPE：約 5.993%
- Actual target：30.2781 ms
- Prediction：32.4685 ms
- APE：約 7.234%

這也說明 inner validation 最佳設定不保證該單一 outer workload 誤差最低；inner loop 只能依其他 workloads 推測哪個設定較可能泛化。

### 10.9 評估與真正部署的差別

Nested LOOW 是模型表現評估流程，不是線上部署時會重跑 24 次。

研究評估階段重複留一法，是因為目前只有 24 個 workloads，想讓每個 workload 都有一次「完全未見」預測。

真正部署前應：

1. 根據 nested CV 結果凍結 feature schema 與 model family。
2. 使用全部 24 個 training workloads 訓練每個方向的 final model。
3. 序列化 imputer、feature filter、scaler、Ridge、feature order 與版本資訊。
4. 新 workload 到來時只在來源 GPU profile 一次。
5. Part 7 在 9/12/15 秒自動停止並輸出 period。
6. Part 8 擷取 features。
7. final model 呼叫一次 `predict()` 得到 target ms/iteration。

目前第 1–3 步尚未正式完成，所以 Part 8 應稱為「production-like model evaluation」，而不是已部署的 production model。

### 10.10 誤差定義

單筆 Absolute Percentage Error：

```text
APE_i = abs(pred_i - actual_i) / actual_i × 100%
```

整體 MAPE：

```text
MAPE = mean(APE_i)
```

其他指標：

```text
MAE  = mean(abs(pred_i - actual_i))
RMSE = sqrt(mean((pred_i - actual_i)^2))
R²   = 1 - SSE/SST
```

例如 actual = 100 ms、prediction = 95 ms：

```text
APE = |95 - 100| / 100 × 100% = 5%
```

### 10.11 Primary 結果

| Direction | N | MAPE | Median APE | Max APE | R² | MAE |
|---|---:|---:|---:|---:|---:|---:|
| 4090→5090 | 24 | 3.25% | 2.00% | 14.01% | 0.9810 | 1.63 ms |
| 5090→4090 | 24 | 5.75% | 3.36% | 23.19% | 0.9480 | 3.11 ms |
| Combined | 48 | 4.50% | 2.68% | 23.19% | 0.9614 | 2.37 ms |

Combined RMSE 為 4.34 ms。

結果不對稱，反向 5090→4090 較難。可能原因包括 source behavior observability、不同架構 metric dynamics、兩節點 host 差異、樣本量太小與 Ridge bias；現階段不能單憑此資料確定因果。

### 10.12 Ablation 結果

Direction-specific combined：

| 方法 | MAPE |
|---|---:|
| Fixed period ratio baseline | 10.61% |
| Detected period only | 7.53% |
| Behavior only | 6.09% |
| Period + compact | 5.11% |
| Period + full | 5.40% |
| Period + compact + quality | 4.35% |
| Oracle period only | 7.56% |

4.35% 是事後固定看某 family 的 aggregate 結果；正式 primary nested family-selection 結果是 4.50%。學術報告應以 4.50% 作主要 estimate，4.35% 作 post-hoc ablation。

Oracle period-only 仍約 7.56%，與 detected period-only 相近，顯示主要瓶頸不只是 period 判斷誤差。Period 告訴模型一個週期多長，但不同 workload 在兩張 GPU 的 scaling 仍與 SM/Tensor/DRAM behavior 有關。這支持「period detection + behavior mapping」而非只用 period ratio。

### 10.13 Worst cases

- C24 / 5090→4090：actual 101.094 ms，prediction 77.646 ms，APE 23.19%。
- C20 / 4090→5090：actual 約 62.050 ms，prediction 53.357 ms，APE 14.01%。

所以 MAPE 4.50% 並不表示每筆都在 4.50% 內；它是 48 筆百分比誤差的平均。模型對多數樣本準確，但仍有值得研究的 systematic outliers。

---

## 11. Data leakage audit

### 11.1 Baseline 是否在訓練時使用

有，但只作 label：

- Training rows 的 target baseline 用來教模型。
- Validation rows 的 target baseline 用來選模型。
- Outer test row 的 target baseline 只在 prediction 產生後計算誤差。

這是正常 supervised learning，不是 leakage。

### 11.2 會構成 leakage 的情況

- 把 test workload 的 target baseline 放進 feature。
- 用 test workload 的誤差挑 feature family、alpha、confidence threshold。
- 先用全資料 fit scaler/imputer，再做 cross-validation。
- 兩個方向拆成獨立 row 後只做普通 row-wise CV，導致同一 workload 的反方向資料留在 training。
- 使用 marker 來切 test profiling cycles，卻宣稱 marker-free。
- 使用 source baseline runtime，卻宣稱只需短 profiling metrics。

### 11.3 目前 Part 8 的保護措施

- Outer split 以 workload group 為單位，同一 workload 的兩方向一起被 hold out。
- Inner CV 只用 outer training workloads。
- Preprocessing 在每個 fold 的 training subset 內 fit。
- Detector threshold 也在 outer training workloads 選。
- Marker-free extractor 不讀 NVTX/callback。
- Target baseline 只作 label/evaluation。

---

## 12. 目前可以支持與不能支持的研究結論

### 12.1 可以支持

- 兩個現有節點之間的 speedup 具有 workload dependence，不宜只用單一固定 ratio。
- 短時間 GPU metric time series 含有足以辨識 iteration-like period 的週期訊號。
- Adaptive 9/12/15 秒策略在現有 48 profiles 上能顯著降低短 horizon 的 harmonic failures。
- 不使用 marker、workload metadata 與 source baseline 的 behavior features，在 nested unseen-workload evaluation 中可達約 4.50% MAPE。
- Behavior features 對 runtime scaling 有超越 period-only 的資訊。

### 12.2 目前不能支持

- 對任意深度學習模型都有 4.50% MAPE。
- 對第三張未見 GPU 也能準確預測。
- 誤差完全源自 GPU，與 CPU／系統環境無關。
- 83 個 features 全部都有穩定因果意義。
- HPCToolkit 對此問題沒有幫助。
- Confidence 是校準機率。
- 真正完全不含 marker/callback 的生產 workload 已被驗證。

---

## 13. Threats to validity

### 13.1 External validity

- 只使用 YOLO26n。
- dataset 是合成且簡化的單物件資料。
- 只變更 batch/imgsz/AMP。
- 只有 NVIDIA 4090/5090，無其他世代、資料中心 GPU 或 AMD GPU。

### 13.2 Statistical validity

- 只有 24 workload groups。
- full factorial rows 共享大量結構，並非完全獨立樣本。
- Nested CV 減少模型選擇偏誤，但不能增加資料本身的多樣性。
- Feature 數相對樣本數高，Ridge regularization 很重要。
- 尚缺 workload-level bootstrap／confidence interval 的最終統一報告。

### 13.3 Measurement validity

- Source traces 在 marker 存在時收集。
- 4090/5090 的 Nsight backend 與 metric schema 不同。
- CPU metrics 無效。
- HPCToolkit 無 trace DB 且有 correlation warning。
- Dataset identity 尚缺每 run hash。
- 尚未完整確認每次量測時 GPU 是否 exclusive、是否有其他 process。
- 執行順序、randomization、cooldown、clock/thermal state 需更完整記錄。

### 13.4 Construct validity

Part 7 偵測的是 profiler signal 的 dominant period。現有資料顯示它接近 iteration period，但更複雜 workload 可能存在：

- gradient accumulation
- alternating phases
- data-loader stalls
- validation/checkpoint intervals
- dynamic shapes
- multi-GPU collectives
- compilation/recompilation

屆時「一個 dominant period = 一個 training iteration」可能不成立，需要 multi-period 或 hierarchical temporal model。

---

## 14. HPCToolkit 與 Part 7 對齊的 future work

這個方向是合理的，但要分層進行。

### 14.1 目前可先做：workload/stable-window aggregate alignment

現有 hpcprof database 沒有 trace DB，所以可先把每個 workload 的 aggregated HPCT metrics 與 Part 8 每筆 workload features 合併，例如：

- CUDA kernel/activity 類型比例
- call-path aggregated time
- memory copy/activity summaries
- GPU operation taxonomy
- CCT depth／hot-path summaries

然後做消融：

```text
GPU metrics only
vs. GPU metrics + HPCT aggregate
```

### 14.2 真正 per-cycle 對齊

若原始 HPCT measurement 含可恢復 timestamps，或未來改用能輸出 trace database 的收集方式，可用 Part 7 推測的週期區間：

```text
[t0 + kP, t0 + (k+1)P)
```

聚合每個 cycle 內的：

- kernel call-path distribution
- CUDA activity duration/count
- memcpy direction/bytes
- synchronization events
- CPU call-path or PMU counters

再建立 cycle-to-cycle stability 與 phase behavior features。

### 14.3 分開 profiler run 的對齊問題

Nsight 與 HPCToolkit 若是分開 run，絕對 timestamp 不能直接對齊。可考慮：

- 以 workload ID 與 steady-window aggregate 合併。
- 以 kernel/activity signature 做相對 phase alignment。
- 在可接受 overhead 時進行同步收集，但要先驗證 profiler coexistence 與 perturbation。
- 讓兩個 profiler 都記錄一個不依賴使用者程式修改的外部 host timestamp anchor。

建議消融順序：

1. GPU metrics only。
2. + HPCT aggregate CCT features。
3. + time-aligned HPCT features。
4. 比較 accuracy gain、profiling overhead 與 feature stability。

---

## 15. 建議的後續研究優先順序

### Priority 1：增加硬體節點

至少增加第三張 GPU，理想上涵蓋不同世代與性能區間。評估方式需加入：

- leave-one-workload-out：測 unseen workload。
- leave-one-device-out：測 unseen hardware。
- leave-one-workload-and-device-out：最嚴格的雙重泛化。

只有這一步完成後，才適合聲稱 hardware descriptors 支持跨裝置泛化。

### Priority 2：增加 workload diversity

加入不同模型與任務：

- classification
- object detection 的其他 model sizes/families
- segmentation
- transformer／LLM fine-tuning
- CNN、ViT、RNN 等不同算子組合
- 不同 optimizer、gradient accumulation、data pipeline

應以 model family 作 group split，避免同家族設定洩漏造成過度樂觀。

### Priority 3：真正 no-marker recollection

在 runner 中完全關閉 NVTX/callback iteration markers，再收一批 GPU metrics，確認：

- metric series distribution 是否一致。
- Part 7 detection accuracy 是否維持。
- Part 8 prediction error 是否維持。

ground truth 可用獨立、低侵入方式在另一批 reference runs 建立，但不能進入 deployment collection path。

### Priority 4：production artifact

- 凍結 feature schema。
- 使用全部目前 training data 訓練 final direction models。
- 序列化完整 pipeline。
- 建立 CLI：輸入 source profiler artifact、source/target ID，輸出 predicted ms/iteration、confidence 與 out-of-distribution warning。
- 加 schema/version checks 與 missing-channel handling。

### Priority 5：改進 period detector

- 線上 change-point／stationarity detection。
- 多尺度 spectral/autocorrelation analysis。
- harmonic disambiguation。
- multi-period detection。
- calibrated confidence。
- fractional-period estimation 與 phase alignment。

### Priority 6：HPCToolkit integration

先做 aggregate ablation，再做 time-aligned per-cycle features；不要在尚無 trace DB 時假設能直接精準對齊每個 iteration。

### Priority 7：不確定性與 OOD

部署輸出除 point estimate 外，應加入：

- prediction interval
- distance-to-training-distribution
- detector confidence
- insufficient cycles warning
- missing metrics warning
- unsupported hardware warning

---

## 16. 建議的新增實驗

1. **No-marker A/B test**：相同 workload 收集 marker-on 與 marker-off GPU metrics，比較 time-series distribution、period APE 與 prediction APE。
2. **Horizon-cost curve**：以 profiling 秒數為成本，畫 coverage、period error、runtime MAPE 曲線。
3. **Feature stability**：同 workload 重複 profiler 3–5 次，測 83 features 的 coefficient of variation。
4. **Thermal/clock robustness**：控制冷機、熱機、power cap、background load。
5. **Workload-family holdout**：整個 model family 留出，而非單一 config 留出。
6. **Device holdout**：增加設備後完整 hold out 一張 GPU。
7. **HPCT ablation**：aggregate CCT、kernel mix、memory activity 分別加入。
8. **Nonlinear models**：在資料量增加後再比較 Gradient Boosting、Random Forest、Gaussian Process、small MLP；目前 n=24/方向不適合複雜模型。
9. **Physics-informed model**：把 workload behavior 分成 compute-bound、memory-bound、launch-bound，再用硬體 ratios 預測 scaling。
10. **Worst-case analysis**：針對 C20/C24 檢查 detector、feature distribution、Ridge leverage、baseline variance 與 profiler variance。

---

## 17. 主要程式與產物導覽

後續接手者應優先閱讀：

### Collection/config

- `configs/experiments.csv`
- `scripts/generate_dataset.py`
- `scripts/train_yolo26.py`
- `scripts/run_baseline.py`
- `scripts/run_nsys.py`
- `scripts/run_hpctoolkit.py`
- `scripts/check_environment.py`
- `scripts/discover_nsys_capabilities.py`
- `scripts/discover_hpctoolkit_capabilities.py`

### Validation and manifests

- `analysis_outputs/collection_validation_report.md`
- `analysis_outputs/collection_inventory.csv`
- `analysis_outputs/ground_truth.csv`
- `hardware/hardware_descriptors.csv`

### Part 7

- `analysis/extract_marker_free_timeseries.py`
- `analysis/detect_marker_free_cycles.py`
- `analysis/evaluate_marker_free_cycles.py`
- `analysis_outputs/model_experiments/part7_marker_free_cycles/report.md`
- `analysis_outputs/model_experiments/part7_marker_free_cycles/deployment_oof_detections.csv`
- `analysis_outputs/model_experiments/part7_marker_free_cycles/oof_cycle_predictions.csv`
- `analysis_outputs/model_experiments/part7_marker_free_cycles/detector_contract.json`

### Part 8

- `analysis/extract_marker_free_cycle_features.py`
- `analysis/run_marker_free_runtime_models.py`
- `analysis_outputs/marker_free_cycle_features.csv`
- `analysis_outputs/model_experiments/part8_marker_free_runtime/report.md`
- `analysis_outputs/model_experiments/part8_marker_free_runtime/predictions.csv`
- `analysis_outputs/model_experiments/part8_marker_free_runtime/experiment_contract.json`

### Earlier experiment reports

- `analysis_outputs/model_experiments/part1_baselines/report.md`
- `analysis_outputs/model_experiments/part2_workload_models/report.md`
- `analysis_outputs/model_experiments/part3_profile_models/report.md`
- `analysis_outputs/model_experiments/part4_trace_hpct_models/report.md`
- `analysis_outputs/model_experiments/part5_final/final_report.md`
- `analysis_outputs/model_experiments/part6_short_window/report.md`

檔名若因後續整理略有不同，可用以下方式搜尋：

```powershell
rg --files | rg "part[1-8]|marker_free|ground_truth|hardware|validation"
```

---

## 18. 重現 Part 7–8 的概念順序

應依序執行：

```text
Nsight GPU metric SQLite
→ extract_marker_free_timeseries.py
→ marker-free 1 ms NPZ/time-series manifest
→ detect_marker_free_cycles.py
→ deployment-safe detections
→ evaluate_marker_free_cycles.py
→ period accuracy report（這一步才讀 oracle marker）
→ extract_marker_free_cycle_features.py
→ 83-feature table
→ run_marker_free_runtime_models.py
→ nested LOOW OOF predictions and report
```

執行前應檢查各 script 的 `--help` 與目前路徑參數，不要假設無參數執行一定指向正確 artifact。

---

## 19. 尚待原研究者確認／補記的 collection provenance

以下資訊目前未在現有 metadata 中得到完整、可稽核答案：

- 兩個節點是否確定使用完全相同 dataset contents；需要 hash/manifest。
- 每個 profiler run 時 GPU 是否 exclusive，是否有其他 process。
- workload 執行順序是否 randomize。
- runs 間是否有固定 cooldown。
- 是否鎖定 GPU clocks／power mode，或至少記錄 temperature/clock trace。
- Nsight GPU metrics 使用的 exact metric-set override。
- 兩節點的精確 Nsight、driver、CUDA 與 HPCToolkit versions。
- HPCToolkit missing-correlation warning 的具體來源與影響。

這些不是推翻現有結果的證據，但會影響學術論文的 reproducibility 與 threats-to-validity 敘述。

---

## 20. 給下一個 ChatGPT 的建議提示詞

可將本文件連同 repository 交給另一個 ChatGPT，並使用以下提示詞：

```text
你正在接手一個「YOLO26 跨異質 GPU Training Runtime Prediction」研究。
請先完整閱讀 ACADEMIC_RESEARCH_HANDOFF.md，以及其中列出的 Part 7/Part 8 scripts、reports、OOF CSV 和 hardware descriptors。

研究的部署契約是：新 workload 不能修改程式碼；只允許在來源 GPU 做約 9–15 秒的 marker-free profiling；不能使用 NVTX、callback、已知 iteration count、batch/imgsz/AMP、source baseline 或 target profiling；輸出是 target GPU 的 steady-state per-iteration runtime。

請嚴格區分：
1. Part 1–2 的 idealized feasibility baseline；
2. Part 3–6 的 marker-aligned oracle benchmark；
3. Part 7–8 的 marker-free production-like evaluation。

目前主要結果是 Part 7 adaptive period mean APE 約 1.01%，以及 Part 8 nested direction-specific LOOW MAPE 約 4.50%。不要把 post-hoc 4.35% 當 primary，也不要宣稱對 unseen GPU 已驗證，因為目前只有 RTX4090/RTX5090 兩個節點。

下一步請先做方法與 leakage audit，再提出可投稿的研究問題、實驗矩陣、統計檢定、第三裝置/新 workload 設計、真正 no-marker validation，以及 HPCToolkit aggregate/time-aligned feature integration 計畫。任何新模型都必須維持 workload-group split；若測硬體泛化，還要做 device holdout。
```

---

## 21. 建議論文敘事主線

若寫成論文，較合理的主線不是「使用 Ridge 達到 4.5%」，而是：

1. 定義無法修改 workload、未知 iteration boundary 的實務問題。
2. 證明固定硬體 ratio 不足以描述 workload-dependent scaling。
3. 提出 marker-free、multi-channel、adaptive-horizon 的週期偵測。
4. 提出 phase-light cycle behavior representation。
5. 用 nested group evaluation 避免同 workload 反方向 leakage。
6. 證明 behavior features 相對 period-only baseline 有增益。
7. 分析兩方向不對稱、worst cases、profiling cost 與 confidence。
8. 在更多 workload families 與 devices 上檢驗泛化。

真正的學術貢獻潛力主要在「黑箱短 profiling 的週期發現與跨硬體 behavior mapping」，Ridge 本身只是目前小樣本下合適、可解釋且低 variance 的預測器。

---

## 22. 簡短術語表

- **Baseline**：沒有外部 profiler 的 reference training run。
- **Ground truth iteration time**：三次 baseline steady-window mean 的 median。
- **Source device**：新 workload 實際進行短 profiling 的裝置。
- **Target device**：要預測 runtime、部署時不執行該 workload 的裝置。
- **Marker-free**：特徵抽取與推論不讀 NVTX、callback 或 iteration counter。
- **Oracle**：只在離線驗證中可取得、部署時不知道的答案或邊界。
- **Period**：GPU metric signal 中估計出的 dominant iteration-like 週期。
- **OOF prediction**：由未使用該 workload 訓練／選模的 outer-fold model 產生的預測。
- **LOOW**：leave-one-workload-out。
- **Nested CV**：outer loop 估泛化誤差，inner loop 選 feature family/hyperparameter。
- **APE/MAPE**：單筆／平均絕對百分比誤差。
- **Production artifact**：凍結並序列化、可對新 workload 單次推論的完整 pipeline。

---

## 23. 當前總結

目前資料並非不可用。相反地，它已足以建立兩節點、同一模型家族下的 marker-free 初步模型，並以嚴格的 unseen-workload nested evaluation 得到約 4.50% MAPE。最關鍵的研究進展是從「使用已知 iteration runtime 或 marker-aligned window」修正為「先從純 GPU metric 訊號偵測週期，再擷取 behavior features」。

但目前研究仍處於 proof-of-concept：硬體數、workload family、真正 no-marker validation、measurement provenance、HPCToolkit temporal integration 與 production packaging 都尚需補強。後續研究應優先增加裝置與 workload diversity，並將評估提升到 unseen workload + unseen device，才能回答真正的跨異質硬體泛化問題。
