# Tesla P4 的 Qwen3-ASR 模型量化與推論最佳化研究

研究日期：2026-11-01（Asia/Taipei）。追蹤：[Issue #14](https://github.com/DF-wu/CapsWriter-Offline-Container/issues/14)。
分支：`research/tesla-p4-asr-model-quantization`；程式基準：`95912cf66f384aa752f3dda291e02c893070beff`。

本次交付是深度來源研究、模型清單與實驗設計，**沒有執行 P4 ASR benchmark，因此沒有已證實的最快模型或加速倍數**。讀取了一次本機 GPU 身分；未替換正式模型、調整時脈、安裝推論套件或修改容器。

相關工作：[既有 runtime 研究](tesla-p4-qwen3-asr-runtime-20261031.md)、[Issue #12 的同模型延遲計畫](../superpowers/plans/2026-11-01-p4-qwen3-asr-dictation-latency.md)。本 issue 把模型量化與縮小模型納入比較，與 #12 保持原 Q5_K 的 runtime 實驗分開。

## 1. 直接回答：有可用量化版，但沒有查到可驗證的「P4 最強版」

最值得先做的組合是：**原生 FP32 encoder 對照現有 encoder，搭配 SM61 CUDA decoder，再在同一個 1.7B checkpoint 上比較 Q5_K、Q4_K_M／Q4_0、Q8_0**。這是依相容性、整合成本與 P4 架構排出的實驗優先序，並非速度排名。

- **現成且最接近本專案：上游 Qwen3-ASR-1.7B Q4_K ZIP。** 上游同時提供目前 Q5_K 與 Q4_K；有下載 URL、大小與 SHA-256，可作低成本候選。但兩個 ZIP 的 encoder／檔案布局尚未逐項比對，不能直接把整包 A/B 的結果全歸因於 decoder 位元數。[S1][S2]
- **更值得研究的模型方向：混合精度。** P4 有 DP4A INT8 與 FP32，沒有 Tensor Cores；因此浮點 encoder 與低位元 decoder 應分開選精度。把整個模型改 FP32，或看到 INT8 檔名就預期加速，都缺少關鍵的 kernel 證據。[H1][H2]
- **有完整 FP32／INT4 ONNX，以及多個 Q4／Q5／Q8 GGUF。** 但分屬 split ONNX+GGUF、llama.cpp 的 text+mmproj、transcribe.cpp 的單檔模型、CrispASR 等不同契約；GGUF 是容器格式，不是跨 runtime 的即插即用保證。[S3][S4][S5][S6]
- **找到 Pascal 專案，但主要是 P40，沒有本機 P4 成績。** `uaysk/vllm-pascal` 作者記錄了 SM61 P40 上 Qwen3-ASR-1.7B 的可用環境。這推翻「Pascal 完全不能跑任何 vLLM」的絕對說法，卻不證明標準 vLLM 支援 P4，也沒有證明 8GB P4 更快或能沿用同樣記憶體設定。[S7]
- **確實找到專屬 Pascal 核心與 P4 調校案例。** 現行 llama.cpp 有獨立 `mmq-config-pascal-dp4a.cuh`；另有 P4 使用者提出 Q8_0 tile 調整，在兩個 BGE 模型的 pp512 測試約提升12%。這是可研究的 P4 kernel 最佳化，**不是已量得的 Qwen3-ASR 加速**。[K1][K2]
- **Jetson Orin INT8 SmoothQuant 確實存在，但不是 P4 成品。** 其 checkpoint 是 TensorRT-Edge-LLM 匯出的中間產物，模型卡明言不能直接用標準 Transformers 推論；encoder 與 lm_head 仍為 FP16。不能把 Orin／RTX 5090 的成績移植為 P4 成績。[S8]
- **沒有找到帶 P4、精確模型 revision、kernel／runtime、中文品質與可重現計時的勝出證據。** 「未找到」是本次搜尋範圍內的負結果，不等於證明網路上不存在。查詢及失敗紀錄保存在[來源與 artifact 清單](tesla-p4-asr-model-artifacts-20261101.json)。

## 2. 你的 P4 與現有系統，實際知道到哪裡

本機只讀命令：

```sh
rtk proxy nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader
```

觀察結果：`Tesla P4, 6.1, 580.196.07, 7680 MiB`。這是 driver 回報總顯存，不是可用顯存、峰值推論記憶體或負載測試。P4 廣告規格為 8GB、192GB/s、最高 5.5 TFLOPS FP32、22 TOPS INT8；規格峰值不可當成 ASR 吞吐。[H1]

P4 的 GP104／SM61 有 DP4A，可將多個 8-bit 整數乘積累加到 INT32。GP104 原生 FP16 算術吞吐低於 FP32；P100 的高速 FP16 不適用於 P4。但是 FP16 **儲存**、FP32 **累加**及 FP16 算術是不同層次，不能由這一點推導「所有 FP16 模型都慢 64 倍」。實際由 cuBLAS／cuDNN／GGML 選擇的 kernel 與 shape 決定。[H2]

從目前程式可確定的是「預期配置」：

| 項目 | 程式證據 | 本輪未觀察的部分 |
|---|---|---|
| 模型 | Qwen3-ASR-1.7B；下載器 pin Q5_K ZIP 與三個檔案 hash | 正式 volume 的 hash／GGUF metadata |
| Encoder | 兩個 ONNX session；輸入 dtype 由 frontend ONNX type 決定 | 現行 ONNX 的 tensor／算子精度、各 node 的 EP 分配 |
| Decoder | llama.cpp b7798；NVIDIA 預設下載 Vulkan libraries | 正式部署是否真的全 GPU、是否 fallback CPU |
| 容器 | CUDA 11.8、cuDNN 8、ORT GPU 1.18.0 | 正式容器映像 digest／實際載入版本 |
| 上屏 | 需保留 client 後處理、服務排程、prompt/context 與字幕對齊 | 各階段占 release→final 的比例 |

本機程式證據：[模型 manifest][L1]、[entrypoint][L2]、[encoder][L3]、[ASR embedding 輸入][L4]、[Dockerfile][L5]、[依賴][L6]。不能只由檔案大小推定 encoder 是 FP16 或 INT4，也不能把 env 中的 `CUDA` 當作每個 node 都在 GPU 的證據。

### 為何不是整個模型 FP32

官方 HF API 在 revision `7278e1e70fe206f11671096ffdd38061171dd6e5` 回報 2,349,217,408 個 BF16 儲存參數，包含完整語音模型，不能只拿名稱中的 1.7B 算顯存。逐參數轉為 FP32 是 **9,396,869,632 bytes，約 8.75 GiB**，尚未計 workspace／KV／activations；這是按現存 tensor 數量的算式，未假設載入器的權重 sharing／去重。對這張回報 7.5GiB 的 P4，完整 FP32 全 GPU 不是合理的預設路線；CPU offload 又需另測傳輸代價。[S9]

**FP32 encoder + 量化 decoder** 的意義不同：只讓適合 FP32 的階段用 FP32，同時降低逐 token 讀取的 decoder 權重流量。這有架構依據，但仍須測 encoder 增加的記憶體／頻寬是否抵銷運算收益。

## 3. 搜到的模型與 runtime，逐項判定

| 候選 | 真正的格式／特性 | 對 P4 與本系統的判定 |
|---|---|---|
| HaujetZhao 1.7B Q4_K ZIP | split ONNX + GGUF；上游現成包 | **優先評測**；最接近現有系統，先拆開確認 encoder 是否相同，不直接覆寫正式模型。[S1] |
| 同一原始權重重新匯出 Q8_0／Q4_0／Q4_K_M | 保留本專案 decoder 架構、tokenizer、embedding 路徑 | **優先評測**；比較少量格式，不能由 Q5_K 重新量化成 Q8 宣稱恢復原品質。Q8 與 Q4 的速度順序未定。[S2][L4] |
| FP32 split encoder | 上游匯出腳本原生 `torch.float32` | **優先評測**；沿用原本輸入／mask／輸出契約；不只是把現行 FP16 檔案加 Cast，也不保證更快。[S2] |
| andrewleech 1.7B ONNX | FP32 encoder + FP32／INT4 decoder init/step；外部權重與獨立 embedding | **完整替代引擎候選**；不是只換本專案兩個 ONNX。CPU WER／RTF、DirectML／WebGPU 測試不能當作 P4 CUDA 成績。[S3] |
| ggml-org 1.7B Q8_0 + Q8 mmproj | llama.cpp 原生 audio encoder + text decoder | **值得新增評測**；可把 encoder 與 decoder 放入同一 GGML runtime。現行 b7798 ctypes 路線沒有載入 mmproj；須獨立 build/adapter，不能任意換 `.so`。[S4][S10] |
| handy-computer 1.7B Q4_K_M／Q5_K_M／Q6_K／Q8_0 | transcribe.cpp 單檔 GGUF | **第二批優先候選**；量化與英語品質證據較完整，CUDA CMake 有 SM61 路徑；P4 仍未測，prompt／串流／timestamps 要補驗。[S5][S11] |
| cstr/CrispASR Q4_K + Q8 audio tower | 單檔、encoder 保留較高精度 | **混合精度的設計參考／次級 runtime 候選**；作者曾修復低位元 encoder 引發空白／重複，不採單段「完美」轉錄作全面品質證據。[S6] |
| s0me-0ne 1.7B ONNX INT8 | dynamic INT8 decoder；encoder FP32 | **CPU 對照，非 P4 最佳化成品**；作者只測單段德語、無完整 benchmark suite；還要處理 signedness、EP 與 KV I/O。[S12] |
| sherpa 0.6B INT8／社群 1.7B INT8 | conv frontend + encoder + decoder ONNX | **CPU／全 ONNX 比較候選**；官方例子為 CPU，不代表 P4 DP4A。1.7B 社群卡明載 CPU export、dynamic INT8；須重新接 adapter。[S13][S14] |
| vrfai 1.7B INT8 SmoothQuant／INT4 AWQ | ModelOpt → TensorRT-Edge-LLM；主要針對 Orin | **排除直接替換**；新 runtime 的 target、plugins、算子與 build 均不同。INT8 卡連結的 VLAOpt 匯出 repo 本輪 HTTP 404，重現性另有缺口。[S8][S15] |
| Blackwood416 1.7B BNB-NF4 | Transformers／bitsandbytes 4-bit checkpoint | **低優先記憶體候選**；Pascal 可有 NF4 硬體支援，但計算 dtype、未量化 audio tower、PyTorch wheel 與本機速度未證實；不是 INT8 DP4A 模型。[S16][H3] |
| uaysk/vllm-pascal | 作者在 P40 SM61 驗證 Qwen3-ASR-1.7B | **可行性備選，非首選速度路線**；固定舊 PyTorch/CUDA、eager 執行、關閉多個快路徑；P4 容量、品質與效能均須獨立確認。[S7] |
| antirez/qwen-asr 原生 CUDA | Safetensors；Q8 token matvec、FP16 batched weights、FP32 activations/cache | **原型候選**；公開測試是 GB10。1.7B Q8 保留兩份權重，作者報 GPU 權重約5.60GiB，另需 buffers；未證實 P4 能以相同參數舒適常駐。[S17] |
| Luigi／vieenrose Nano gen1 Q8 | 為 SM53／CUDA10.2 改過的 0.6B 引擎 | **舊卡移植參考**；Maxwell 沒有 DP4A，與 P4 不同。其 Q8 勝 Q4、繁體字與 CPU decode 單段結論，不可套到 SM61 或 1.7B。[S18] |
| faster-qwen-asr、雲端 FP8 最佳化 | Torch compile／CUDA Graph 或新卡 BF16/FP8 核心 | **看設計、不移植其速度承諾**；faster-qwen-asr 的公開環境為 GB10 + CUDA13；與 P4 的工具鏈及硬體不相容。[S19] |

### 新發現：可直接研究的全 GGML encoder 路線

先前報告重點是 ONNX + llama.cpp，這次查到 ggml-org 已發佈 Qwen3-ASR text decoder／mmproj 配對。所讀 llama.cpp commit `0ec4027a2a0183de87c007069adddc06e665f7de` 的 `tools/mtmd/models/qwen3a.cpp` 直接引用 Qwen3-ASR reference，`mtmd.cpp` 有 QWEN3A audio preprocessing。[S4][S10]

這使「**同一 runtime 內的 Q8 audio encoder + 低位元 text decoder**」成為值得加入的候選：可能避免目前 NumPy 介面中的部分往返搬移。但這只是整合機會，未證明消除了所有 host copy，也未證明該 Q8 encoder 的所有算子都走 DP4A。首先以發佈的 Q8 text+Q8 mmproj 驗正確性，再比較自原始浮點權重產生的 Q4／Q5 text；不要將首次載入、量化與 runtime 遷移混在一組實驗。

官方卡只有最小啟動示例，未給 P4 benchmark。套件下載、讀到 audio、產出非空文字，都仍不足以證明辨識正確；必須跑中文與混語固定樣本。[S4]

## 4. INT8 與 FP32 最容易踩到的三個坑

### 4.1 「INT8」可能是三種不同的計算

1. **Weight-only**：W8A32／W4A32，低位元權重載入後解量化，再以浮點做部分或全部運算。可能省頻寬，但不是廣告中的 22 INT8 TOPS。
2. **真正整數點積**：量化 activations 與 weights，再用 DP4A／INT32 accumulate；量化、scale、zero point、packing 的成本要一起量。
3. **Tensor Core INT8／FP8／NVFP4**：使用較新架構專用核心；P4 沒有這條硬體路線。

本專案的 GGUF Q4／Q5 能否用 DP4A，取決於 llama.cpp CUDA 內 MMVQ／MMQ 的實際分派與輸入形狀；「不是 Q8」不表示不能用整數點積，「是 Q8」也不表示整個模型都是 INT8。原始碼與版本邊界詳見[硬體／kernel 證據附錄](p4-asr-hardware-kernel-evidence-20261101.md)。[H2]

**P4 專用調校的新線索：** llama.cpp issue #30235 作者在 Tesla P4／CUDA12.6／Windows 上，僅把 Pascal config 的 Q8_0 `I` tile 寬度由64改128，兩個 BGE encoder 的 `pp512` 分別得到約12.1%／11.6%提升。作者沒有測 Q8 decoder、Q5_K、ASR，也沒有 batch／prompt sweep；把同樣改動擴到所有量化型別，部分 Q4／Q6 工作負載反而顯著退步。這是後續 **Q8 大矩陣乘法** 的窄範圍實驗，不是所有 Qwen token decode 的通用加速。[K1]

所查 current commit 的 Q8_0 tile 仍為64；有明確來源可以建立獨立 patch A/B，但 b7798 與 current 的 kernel／ABI 不同，要放在完整 runtime 版本實驗之後。Pascal 預設也已能選 MMQ，小 batch 還可先走 MMVQ，所以 `GGML_CUDA_FORCE_MMQ=ON` 不會自動讓所有工作更快。[K2][K3]

### 4.2 上游 INT8 encoder 腳本不能直接套進 ORT 1.18

上游 `04-Quantize-ASR-Encoder.py` 的 INT8 設定是 `MatMulNBitsQuantizer(bits=8, block_size=128, accuracy_level=4)`，只挑 MatMul，沒有把整個 CNN／attention 變成 INT8。[S2]

但**本專案 pin 的 ORT 1.18.0**，CPU `MatMulNBits` 明確只允許 4-bit；CUDA 實作也走 `TryMatMul4Bits`／`Dequantize4Bits`，不是能正確執行任意 bits=8 的通用核心。不能因工具叫 NBits 就假定新版 exporter 的 W8 artifact 可用。[O1][O2]

同時不能過度概括成「CUDA EP 完全沒有 MatMulInteger」：ORT 1.18 的 kernel 表確實列 **S8×S8** CUDA MatMulInteger；CPU dynamic export 常見的 DynamicQuantizeLinear 輸出則是 **U8**，CUDA 表沒有相同的完整配對。應檢查實際圖、signedness、fusion 與 node placement，否則可能 CPU fallback 或建立 session 失敗。[O3]

結論是：**現成 CPU INT8 ONNX 不是目前 P4 容器的一鍵 INT8 加速包**。若做 encoder 的真 S8S8 DP4A 路線，需要特定圖／校正／kernel 的工程與品質驗證，而不是改副檔名。

### 4.3 舊 TensorRT 可行，但不能安裝「最新版」代替

TensorRT **8.6.1** 的歷史矩陣仍列 P4、FP32／FP16／INT8，沒有 Tensor Cores；所以「P4 不能做任何 TensorRT INT8」也不正確。[H4]

值得研究的窄題是 **舊 TensorRT 8.6.1 的 encoder-only FP32／INT8 build**：先由相容 ONNX 圖開始，準備代表實際中文／混語／噪音的 calibration，觀察哪些層真的使用 INT8，必要時對敏感層保留 FP32。現有程式的 TRT 分支會啟用 `trt_fp16_enable=True`，也必須納入 P4 專用配置審查；它不是一個已做好的 INT8 模式。[L3]

但 8.6.1 的歷史硬體支援不證明 Qwen encoder 的動態 mask／算子全部可解析；更不等於 TensorRT-LLM／TensorRT-Edge-LLM 在 P4 可用。ORT、TensorRT、CUDA、cuDNN 的版本須一起固定，新 TensorRT 與 CUDA13 工具鏈不能沿用此硬體結論。這是後排實驗，成本高於先測原生 FP32 encoder。

尤其 ORT 官方的 EP 配對表把 ORT1.18 對到 TensorRT10.0，ORT1.17／1.16 才對到8.6。不能把「現有 ORT1.18 CUDA EP 可用」理解成「直接在同映像加 TRT8.6 EP 可用」；需要另建相容環境。[O4]

## 5. 可下載與重現的具體 artifact

以下大小皆為**檔案／壓縮包大小**。沒有下載多 GB 模型；hash 是發佈方 API 的值，尚未在本機重算。精確 revision、全部候選檔名、SHA-256 與 pinned download URL 收錄在 [JSON manifest](tesla-p4-asr-model-artifacts-20261101.json)。

| 發佈方／格式 | 已核對的檔案 | 大小 bytes | 用途 |
|---|---|---:|---|
| CapsWriter 上游 | `Qwen3-ASR-1.7B-q5_k.zip` | 1,951,570,656 | 現行下載器基準；SHA 與 repo manifest 相同 |
| CapsWriter 上游 | `Qwen3-ASR-1.7B-q4_k.zip` | 1,410,584,449 | 最接近現有引擎的較低位元候選 |
| ggml-org | `Qwen3-ASR-1.7B-Q8_0.gguf` | 2,165,034,944 | 全 GGML 路線的 text decoder |
| ggml-org | `mmproj-Qwen3-ASR-1.7B-Q8_0.gguf` | 355,709,344 | 上一檔配對的 audio encoder/projector |
| handy-computer | `Qwen3-ASR-1.7B-Q4_K_M.gguf` | 1,319,830,496 | transcribe.cpp 單檔候選 |
| handy-computer | `Qwen3-ASR-1.7B-Q5_K_M.gguf` | 1,517,290,464 | 同 runtime 中間品質候選 |
| handy-computer | `Qwen3-ASR-1.7B-Q8_0.gguf` | 2,185,030,624 | 同 runtime 高精度對照 |
| andrewleech | `encoder.onnx` | 1,270,241,779 | 完整 ONNX runtime 的 FP32 encoder；不是本專案 split encoder |

注意同檔名不代表同 tensor：ggml-org 與 handy 的 Q8_0 不同大小／hash，前者還需 mmproj。andrewleech 的 `encoder.int4.onnx` 與 `encoder.onnx` 在本輪 API **大小及 SHA-256 完全相同**，其文件也說 encoder 保留 FP32；`.int4` 名字表示選取的整組 variant，不表示每一個 component 都量化成 INT4。[S3]

另一個常被忽略的差別：slyusarev 的卡記錄 Q4_K_M 自 Q8 requantize，Q5_K_M 則自原始 BF16 轉出，並去掉重複 output embedding；因此它的 Q5 檔可以比 Q4 還小。這是 tensor 選擇／sharing 不同，不能拿位元名稱推導品質、速度或嚴格單變因實驗。[S20]

## 6. 品質證據：哪些可以參考，哪些不能推廣

handy-computer 的一手模型卡給出 2,620 段 LibriSpeech test-clean、batch=1、Apple M4 的 WER：Q8_0 1.62%、Q5_K_M 1.65%、Q4_K_M 1.81%。這支持「有可重現的量化品質對照」，不支持「繁體中文／混語無損」或「P4 Q4 最快」。不同匯出 pipeline 的小樣本結果也不必一致。[S5]

andrewleech 的資料來自 200 段 test-other／CPU：1.7B FP32 3.79%、INT4 4.20%，而該作者的 per-tensor AWQ INT8 1.7B 路線有 special-token 預測失敗。這是**特定量化 recipe 的負結果**，不能擴大為所有 1.7B INT8 都不能用，也不應因檔名 INT8 比 INT4 高就預期品質必然更好。[S3]

cstr 作者記錄了 Q4 audio tower 造成空白或重複，後來保留 Q8 tower。這使「encoder 先保 FP32／Q8、decoder 才試更低位元」有實際工程依據，但 encoder cosine similarity、LLM perplexity、單段正確輸出仍不是中文 CER／WER 的替代品。[S6]

同理，Nano gen1 模型卡宣稱 Q4 影響繁體字，是值得納入測試的風險案例，未有足夠資料可宣稱 Q4 會普遍把所有中文變簡體。應分開記錄原始 ASR script、繁簡後處理與內容正確率，避免用 OpenCC 遮掉詞彙辨識回歸。[S18]

## 7. 實驗矩陣：如何在這張 P4 找出真正最快的組合

### 第一批：保留 1.7B，先測最接近現有系統的路線

| ID | 唯一主要變因 | 固定條件與目的 | 進入下一步的條件 |
|---|---|---|---|
| B0 | 無，建立現況 | 現有 Q5_K／encoder／Vulkan，記錄有效 EP、模型 hash、全 GPU／fallback | 有可信 stage timing、品質與 client latency |
| B1 | decoder backend | 同一 Q5_K、同一 b7798 ABI，Vulkan → CUDA SM61 | 能正確常駐、沒有非預期 CPU fallback |
| B2 | encoder precision | 固定勝出 decoder，現有 encoder → 從原始 checkpoint 原生匯出的 FP32 split encoder | tensor/mask/長度契約正確；stage 與全程均量 |
| B3 | decoder quantization | 固定 encoder／CUDA build，以同一浮點 decoder 匯出 Q5_K、Q8_0、Q4_K_M、Q4_0 | 同組品質條件下比較 p50/p95，保留負結果 |
| B4 | 低位元 kernel 設定 | 只在 profiler／來源顯示有意義時，逐個測 default／FORCE_MMQ／FORCE_CUBLAS | 不將強制參數預設為快；量 prefill 與 decode 差異 |

B3 至少保留 Q5_K 基準；若現成 Q4 ZIP 的 encoder 不同，先作 package-level feasibility，或只取已驗相容的 decoder，不能把不同 encoder 當成同條件。不要把 Q4_K、Q4_K_M、Q4_0 混稱為相同格式。

### 第二批：模型不變，替換完整推論實作

1. **G1：llama.cpp 原生 Q8 text + Q8 mmproj，編譯 SM61。** 正確性通過後再測較低位元 text／較高精度 encoder；重新核對 audio preprocessing、prompt 模板、語言 token 及上下文位置。
2. **G2：transcribe.cpp 1.7B Q8_0／Q5_K_M／Q4_K_M。** 在常駐 C API／Python adapter 中量測；保留 CapsWriter 外層分段／字幕 aligner。CLI 每次載入模型的數字另列。
3. **G3：舊 TensorRT encoder-only FP32／校正後 INT8。** 先建立相容工具鏈及 parser／node placement 證據；敏感層保高精度，完整品質不合格即淘汰。
4. **G4：antirez CUDA 與 Pascal vLLM fork。** 僅在前項仍不足時投入；先核對 7.5GiB 實際總顯存下的峰值配置，避免雙份權重／cache 壓力。這是較高維護成本方案。
5. **G5：Q8_0 Pascal MMQ tile64／128。** 僅在 G1/G2 選定的版本實際含該 config、Q8 matmul 是顯著瓶頸時另測；保留同模型／音訊與交錯 A/B，不將 BGE 的12%當作預期 ASR 收益。[K1][K2]

### 第三批：明確接受能力／品質取捨的效能上限比較

- **Qwen3-ASR-0.6B**：可比較 FP32 encoder + Q8 decoder，或全 GGML mixed quant。這次模型端研究允許評估它，但它不是「同 1.7B 無損加速」；官方公開數據亦存在品質差異。[S22]
- **SenseVoice／Paraformer**：若主要是中文／英文短句，非自回歸或並行解碼模型值得當速度對照。本 repo 有引擎程式，但容器下載器只支援 qwen_asr／fun_asr_nano，需另外整合模型與 bootstrap。SenseVoice 的快速辨識設計是一手資料，不能當成本 P4 的延遲結果。[S21][L1]
- **Whisper + CTranslate2 INT8/FP32**：P4 硬體路線值得當跨模型對照；CTranslate2 的 Qwen3 文字支援不能等同完整 Qwen3-ASR 語音轉換。更換模型的中文、方言、熱詞／context 與標點代價要單列，不能計為保留所有能力的勝出。[S23][S24]
- **蒸餾／剪枝／speculative decoding**：這次搜尋沒有找到可重現、已在 P4 驗證的 Qwen3-ASR 成品。0.6B 不能直接當 1.7B 的 drop-in draft；audio embedding 尺寸、KV、tokenizer 與驗證路徑都需額外工程。先不排在低成本的量化／encoder 實驗之前。

### 共同比較規則

- 主指標：同一個 client monotonic clock 的「放開錄音 → 最終文字上屏」p50／p95；server 各階段獨立記 duration，不相減跨主機時鐘。
- 樣本：固定 2／5／10 秒短句；至少每組 30 個成功 latency 樣本，baseline/candidate 交錯重複；少量樣本的 p95 只作探索，勝出再擴樣。長檔／字幕獨立回歸。
- 區分 cold load、warm、閒置後第一句，保持單請求 batch=1；另列錯誤、取消、OOM 與空白轉錄，不能把失敗當作快。
- 固定模型原始 revision、quantizer／runtime SHA、CUDA／driver／ORT、CPU threads、prompt、語言、context 與生成參數；測量用編譯配置／容器 digest 也保存。
- 記錄 mel、frontend、backend encoder、audio embeddings 交接、prefill、逐 token decode、aligner、queue、client 後處理；有 CUDA profiling 時檢查 CPU fallback、反覆 H2D/D2H、是否實際使用整數核心。
- 品質集包含台灣繁中、英文混語、數字／單位、專名、方言、噪音、靜音、長段邊界。保存去識別樣本 ID、CER/WER、關鍵詞、幻覺／重複／空白與字幕時間戳；正式選模前依 baseline 分布先訂非劣化門檻，不能量完再改。
- RTF 統一成 `wall_seconds / audio_seconds`，越小越快；有些專案把倒數叫 RTF／RTFx，不直接混排。
- 峰值 VRAM／RAM 用量實測；不以模型磁碟大小、量化位元或 P40 能載入推定 P4 能載入。
- 若 decoder 不是主要耗時，先解 encoder／搬移；若已達 GPU 計算瓶頸，錄音期間增量處理與尾段收斂可能比再降一個量化位元更影響體感。仍需保留 final 的穩定性。

## 8. 重現用起點與版本固定

以下是**下一階段實驗範例，未執行建置、匯出或推論**。在獨立來源／模型目錄進行，不能取代正式 downloader 的 hash／marker／ABI 審查。

```sh
# 同 ABI 的 Pascal decoder 建置起點；使用能編譯 SM61 的 CUDA 11.8/12.x。
rtk proxy git clone --branch b7798 --depth 1 https://github.com/ggml-org/llama.cpp.git /work/llama-b7798
rtk proxy cmake -S /work/llama-b7798 -B /work/llama-b7798/build-sm61 \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_BUILD_TYPE=Release
rtk proxy cmake --build /work/llama-b7798/build-sm61 --config Release -j4

# 輸入必須是保留現有模型語意的原始浮點 decoder，不是現有 Q5_K 反量化。
rtk proxy /work/llama-b7798/build-sm61/bin/llama-quantize \
  /work/asr-export/qwen3_asr_llm.f16.gguf /work/asr-export/qwen3_asr_llm.q8_0.gguf Q8_0
```

上游 exporter 檔案順序為 `01-Export-ASR-Encoder-Frontend.py`、`02-Export_ASR-Encoder-Backend.py`、`03-Optimize-ASR-Encoder.py`、`04-Quantize-ASR-Encoder.py`、`05-Export-ASR-Decoder-HF.py`、`06-Convert-ASR-Decoder-GGUF.py`、`07-Quantize-ASR-Decoder-GGUF.py`。先 pin commit、依其設定調整路徑，保留 FP32 encoder 產物，Linux 另指定對應的 `llama-quantize`；其中第07腳本原樣是 Windows `.exe` 路徑，不能直接複製執行。[S2]

既有 b7798 只作 B1／B3 同 ABI 比較；原生 mmproj 路線用另一個固定完整 llama.cpp commit，重新建置並驗證 adapter，不能把這兩個版本混用。

## 9. 搜尋方式、證據邊界與負結果

以使用者指定的 **Tavily advanced search／extract** 做 discovery，接著讀作者模型卡、官方文件、GitHub pinned source、HF／GitHub API。搜尋摘要、DeepWiki、推測型 GPU fit 網站、P4 雲端機型名及 Qwen3 文字模型成績不作結論證據。

JSON 保存了主研究的 24 組查詢及 3 次改寫重試，另有多批 extract；完整 query、回傳 URL、request ID、HTTP 550／554 失敗一併保留。背景硬體研究另附查詢紀錄。曾呼叫 Tavily Pro research，工具只回傳 `pending`；當前 gateway 只暴露提交、未暴露取得該 report 的工具，所以**不宣稱已讀取其完成報告，亦不以該 pending 工作支持任何結論**。

HF API discovery 每種尺寸最多取100筆；因此模型清單是針對候選的深入核對，不是全 Hub 窮盡清單。沒有對所有 artifact 下載、執行或驗證作者品質宣稱。VLAOpt repo 404、Nano／Apple／Orin／GB10／P40 與 P4 的硬體差異、CPU INT8 與 CUDA kernel 差異，均保留在判定中。

下一步應先完成 **B0 → B1 → B2 → B3**，並把 **G1 全 GGML Q8 encoder** 作為值得新增的獨立候選；這能回答 P4 最關鍵的「kernel、encoder 精度、decoder 位元」三個變因。只有完成這些實測，才有資格把某個組合稱為你系統中的最快配置。

## 一手來源

[H1]: https://images.nvidia.com/content/pdf/tesla/184457-Tesla-P4-Datasheet-NV-Final-Letter-Web.pdf
[H2]: https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html#fp16-arithmetic-support
[H3]: https://huggingface.co/docs/bitsandbytes/main/en/installation
[H4]: https://archive.docs.nvidia.com/tensorrt/tensorrt-861/support-matrix/
[S1]: https://api.github.com/repos/HaujetZhao/CapsWriter-Offline/releases/tags/models
[S2]: https://github.com/HaujetZhao/Qwen3-ASR-GGUF/tree/e790e3bfe9c1c52453da1166df236950a8a8b036
[S3]: https://github.com/andrewleech/qwen3-asr-onnx/blob/4aee2ae0c355c7bc33bd5c09befec9e9eabd4b50/README.md
[S4]: https://huggingface.co/ggml-org/Qwen3-ASR-1.7B-GGUF/tree/36a678687ba7d07a74ca70ccb0e36902e005fb80
[S5]: https://huggingface.co/handy-computer/Qwen3-ASR-1.7B-gguf/blob/0e0a596e3959c3f3f2ab7d443a5b6f0f4a4bba6f/README.md
[S6]: https://huggingface.co/cstr/qwen3-asr-1.7b-GGUF/blob/674df5d44b50a63e7102a18895ed20e3f91de301/README.md
[S7]: https://github.com/uaysk/vllm-pascal/blob/36e14f2c10081d020c4b372884ad67345804c6b7/README.md
[S8]: https://huggingface.co/vrfai/Qwen3-ASR-1.7B-int8/blob/12c3f1a1797f3900deac244ffa8289e515845248/README.md
[S9]: https://huggingface.co/api/models/Qwen/Qwen3-ASR-1.7B/revision/7278e1e70fe206f11671096ffdd38061171dd6e5?blobs=true
[S10]: https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/tools/mtmd/models/qwen3a.cpp
[S11]: https://github.com/handy-computer/transcribe.cpp/blob/5700f127e38dd53c4735ea23ef743b4b538eacd4/docs/models/qwen3-asr.md
[S12]: https://huggingface.co/s0me-0ne/Qwen3-ASR-1.7B-ONNX-int8/blob/cc8f9f1a5fda2f22966b0f1814785c2526101b07/README.md
[S13]: https://k2-fsa.github.io/sherpa/onnx/qwen3-asr/pretrained.html
[S14]: https://huggingface.co/thieunv-asilla/sherpa-onnx-qwen3-asr-1.7B-int8/blob/69eb686fd94a4a865bb5340a3d6ac0d7f1fec0d5/README.md
[S15]: https://nvidia.github.io/TensorRT-Edge-LLM/0.7.0/user_guide/getting_started/supported-models.html
[S16]: https://huggingface.co/Blackwood416/Qwen3-ASR-1.7B-BNB-NF4/blob/1098afde6281085f55c90d62937c638e1baef05d/README.md
[S17]: https://github.com/antirez/qwen-asr/blob/924694251d9e0f18e5d86bbd06aa3ab5f870002d/README.md
[S18]: https://huggingface.co/Luigi/qwen3-asr-0.6b-rapidspeech-gguf/blob/afdbe99228fe9f82b6fb54ea05d06b1dec485056/README.md
[S19]: https://github.com/andimarafioti/faster-qwen-asr/blob/e8d4a8ac336561e11ee4952ff04bf33142a74678/README.md
[S20]: https://huggingface.co/slyusarev/Qwen3-ASR-1.7B-GGUF/blob/3f45ea4f5f84f3159a561c9ece8063d56e740648/README.md
[S21]: https://github.com/FunAudioLLM/SenseVoice
[S22]: https://github.com/QwenLM/Qwen3-ASR
[S23]: https://opennmt.net/CTranslate2/quantization.html
[S24]: https://opennmt.net/CTranslate2/guides/transformers.html
[O1]: https://github.com/microsoft/onnxruntime/blob/v1.18.0/onnxruntime/contrib_ops/cpu/quantization/matmul_nbits.cc#L90
[O2]: https://github.com/microsoft/onnxruntime/blob/v1.18.0/onnxruntime/contrib_ops/cuda/quantization/matmul_nbits.cc
[O3]: https://github.com/microsoft/onnxruntime/blob/v1.18.0/docs/OperatorKernels.md
[O4]: https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#requirements
[K1]: https://github.com/ggml-org/llama.cpp/issues/30235
[K2]: https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/ggml/src/ggml-cuda/mmq-config-pascal-dp4a.cuh
[K3]: https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-cuda/mmq.cu#L262
[L1]: https://github.com/DF-wu/CapsWriter-Offline-Container/blob/95912cf66f384aa752f3dda291e02c893070beff/docker/server/download_models.py
[L2]: https://github.com/DF-wu/CapsWriter-Offline-Container/blob/95912cf66f384aa752f3dda291e02c893070beff/docker/server/entrypoint.sh
[L3]: https://github.com/DF-wu/CapsWriter-Offline-Container/blob/95912cf66f384aa752f3dda291e02c893070beff/core/server/engines/qwen_asr_gguf/inference/encoder.py
[L4]: https://github.com/DF-wu/CapsWriter-Offline-Container/blob/95912cf66f384aa752f3dda291e02c893070beff/core/server/engines/qwen_asr_gguf/inference/asr.py
[L5]: https://github.com/DF-wu/CapsWriter-Offline-Container/blob/95912cf66f384aa752f3dda291e02c893070beff/docker/server/Dockerfile
[L6]: https://github.com/DF-wu/CapsWriter-Offline-Container/blob/95912cf66f384aa752f3dda291e02c893070beff/requirements/server-docker.txt
