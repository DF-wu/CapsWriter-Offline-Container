# splitQwen3-ASR：GPU 廠牌、runtime 與模型優化政策

研究日期：2026-11-01（Asia/Taipei）。範圍：Linux Docker Compose、現有 split ONNX encoder＋GGUF decoder；llama.cpp 固定 **b7798 / `c301172f660a1fe0b42023da990bf7385d69adb4`**。研究起點 checkout HEAD：`63216157dc324683c34b45ec3b9bab433166f475`，工作區有父層並行修改；下述本機程式狀態是讀取時快照，不表示 master 或已發布映像的狀態。

本輪只做一手文件、作者 source 與既有研究紀錄的閱讀，未執行 benchmark、GPU 探測、Docker、服務操作、模型下載、套件安裝或編譯；**沒有新增任何硬體驗證結果**。既有 P4 測量另行標示，Intel／AMD 的整套 ASR 路徑均未在本輪實測。

## 可直接採納的結論

1. **一份 Compose 可以作共同設定入口，但不能把一個現有 CUDA image 宣稱為三廠原生 runtime 已齊備。** 建議以同一 Compose 的 image 變數或互斥 profiles 選擇經固定版本驗證的 CPU、NVIDIA、Intel、AMD 映像。裝置授權與映像選擇仍需在容器啟動前完成；容器內 fallback 救不了 Docker 建立容器前的裝置／runtime 錯誤。[D1]
2. **NVIDIA 優先 CUDA 是合理政策；現有 CUDA 11.8＋cuDNN 8＋ORT 1.18.0＋b7798 SM61 是 P4 路徑，不是所有 NVIDIA 型號的加速保證。** 可沿用現有權重，不必先重新量化。新舊世代需要各自的 CUDA、ORT、driver、編譯架構契約。[N1–N3]
3. **Intel 原生候選是 encoder 用 ORT OpenVINO、decoder 用 llama.cpp SYCL；Vulkan 是 decoder 的另一候選。** OpenVINO 不會替 GGUF decoder 自動加速；SYCL／Vulkan 也不會替兩份 ONNX encoder 自動加速。現有 encoder adapter 尚未接 OpenVINO。[I1、I2、R1]
4. **AMD 原生候選是 encoder 用 ORT MIGraphX、decoder 用 llama.cpp HIP；Vulkan 可作 decoder 候選。** 舊 ORT ROCm EP 可維持固定版本的歷史相容路線，但不應當作新版預設：上游自 ORT 1.23 移除 ROCm EP，AMD 文件列 ROCm 7.0 為最後含它的官方發行，7.1+ 移除。[A1–A3]
5. **社群新版 patch 要逐套判讀。** TheRock／PyTorch 或 Ollama 成功不等於本模型可用；但 Schaka 的新版 Polaris 專案確實已包含 ORT＋MIGraphX，不能錯稱所有 Polaris patch 都只支援 decoder／PyTorch。其 RX 470 作者測試仍不等於本專案完整 splitQwen3-ASR 驗證。[A4–A7]
6. **沒有找到一個「完全免改模型且對所有 NVIDIA GPU 保證加速」的合格方案。** 這是本輪證據結論，不是聲稱數學上不存在任何未來方案。硬體相容、算子落點、音訊長度、搬移成本、顯存與品質都必須成立；能載入不代表更快。

## 必須分開的四層契約

| 層次 | 可以證明什麼 | 不能據此宣稱什麼 |
| --- | --- | --- |
| GPU brand／PCI vendor／裝置節點 | 有 NVIDIA、Intel、AMD 裝置或 render node | 正確 driver、EP、GPU kernel 已可用 |
| 後端存在 | ORT 列出 EP；llama 載入 backend／枚舉 GPU | 本模型所有需要的 op、dtype、shape 能執行 |
| 整體模型契約 | 同一套 encoder、embedding、decoder、tokenizer、prompt、ABI 與功能可跑 | 一定全 GPU、品質相等、一定低延遲 |
| 實測速度 | 指定硬體／版本／音訊／精度／量測範圍下的結果 | 同品牌全系列、其他語音負載、未測硬體也同樣加速 |

ORT 的 provider priority 是**每個可支援節點**的選路規則；`['CUDAExecutionProvider', 'CPUExecutionProvider']` 不代表全圖都在 CUDA。`get_available_providers()` 也不等於指定模型已成功執行。llama 的 backend 是另一個 registry，兩者不可混為同一個 GPU 開關。[O1]

### 本 repo 目前實際怎麼接

- [encoder.py](../../core/server/engines/qwen_asr_gguf/inference/encoder.py) 建立 frontend、backend 兩個 ORT sessions，只明確處理 TensorRT、DML、CUDA，否則保留 CPU；加入字串 `OPENVINO`／`MIGRAPHX` 環境值不會自動接通。目前沒有 OpenVINO／MIGraphX／ROCM 分支。[R1]
- 同檔 frontend 按 100 mel frames 分塊、`sess.run()` 後以 NumPy concatenate，再交給 backend；backend 序列／mask 長度可變。這是要核對的實際 shape／dtype 契約，不能用簡單 MatMul smoke 代替。現有 host NumPy 邊界也表示切換 GPU backend 不會自動得到跨兩套 runtime 的零拷貝。[R1]
- [asr.py](../../core/server/engines/qwen_asr_gguf/inference/asr.py) 把 audio embeddings 與 GGUF token embedding table 按現有 prompt 拼接成 FP32 embedding sequence，再交給 llama batch。要保持 embedding 維度、特殊 token、位置編碼、KV／sampling 與 streaming rollback 行為；能跑一般文字 Qwen GGUF 並不足夠。[R2]
- [llama.py](../../core/server/engines/qwen_asr_gguf/inference/llama.py) 用 ctypes 手寫 `llama_model_params`、`llama_context_params`、`llama_batch` 結構；[build_cuda.sh](../../docker/server/build_cuda.sh) 明確因 ABI 相容而固定 b7798。新版 llama 或社群預編譯庫不能只換 `.so` 就假定相容。[R3]
- [requirements/server-docker.txt](../../requirements/server-docker.txt) 固定 `onnxruntime-gpu==1.18.0`。OpenVINO／MIGraphX 是各自的 ORT build／套件與相依庫；不是現有 CUDA wheel 的 provider 名稱切換。[N1、I1、A2]
- 讀取時 [entrypoint.sh](../../docker/server/entrypoint.sh) 的 `auto` 在沒有 P4 optimization 時會選 Vulkan；CUDA bundle 明確以 SM61 建置，CUDA 選擇也受 P4 flag 保護。**不能把目前程式描述成「任意 NVIDIA 已自動選原生 CUDA」。** 父層若正在改政策，需另以最終程式確認。

## NVIDIA：先修 runtime，不必先改權重

### 已有合格來源與既有 P4 證據

ORT 官方相容表列 **ORT 1.18.x 的 CUDA 11.8 build 對 cuDNN 8.x**；同為 1.18 的 CUDA 12 build，1.18.0 對 cuDNN 8，1.18.1 對 cuDNN 9。cuDNN 8／9 不可互換，不能只憑版本前綴決定 image。[N1]

b7798 CUDA CMake source 明列 CC61 的 DP4A 架構，CUDA <13 的預設 targets 含 61；本 repo 則顯式建置 `CMAKE_CUDA_ARCHITECTURES=61`。這證明有 SM61 build 路線，沒有證明所有 GGUF op 或所有 NV 架構皆已涵蓋。[N2、R3]

既有 [P4 CUDA 部署報告](p4-cuda-runtime-deployment-20261101.md) 記錄：原 FP16 ONNX encoder 與原 Q5_K_M GGUF SHA-256 不變，將 decoder 從 CPU fallback 改成 b7798 CUDA SM61；10 秒片段 HTTP p50 從 4.1497s 降至 0.7902s（每長度／方案 3 次正式請求，另有暖機）。這是**該 P4／該環境的小樣本歷史測量**，不是本輪重測，也不是 CUDA 對所有 GPU 的倍率。該報告另揭露 HTTP temperature 行為與完整 image 尚未重建的限制；本輪未查核其服務現況。[P1]

### 真正阻礙

- **編譯架構覆蓋**：SM61-only bundle 沒有驗證其他 CC。b7798 同檔對 Blackwell 明列 CUDA 12.8+ 的架構處理；保留 CUDA11.8 對 P4 的路徑，不等於覆蓋新卡或新卡最佳 kernel。[N2]
- **工具鏈代際**：NVIDIA 說明 CUDA13 不再提供 CC<7.5 的 offline compilation；既有舊 toolkit binary 是否繼續跑由 driver 支援決定。不得用「全部升最新 CUDA」取代 Pascal 支援政策，也不得因此宣稱舊 binary 立刻不能跑。[N3]
- **encoder 與 decoder 分別成功**：兩個 ORT sessions 要有有效 CUDA placement；decoder 要有實際 GPU layer／buffer／kernel 使用。只有 NVIDIA 裝置、ORT provider 名單或 llama 的庫檔不夠。[O1、R1–R3]
- **模型／負載特性**：P4 沒有可由旗標創造的 Tensor Cores；FP16 檔案大小、FP16 運算吞吐、GGUF 低位元 kernel 是不同事。既有 [P4 實驗](p4-exclusive-container-experiments-20261101.md) 中 FP32 promoted encoder、Q6／Q8 未成為採納方案，不能只按精度名稱選最快。[P2]

**本次安全落地建議**：保留已知 P4 三份模型與 b7798 ABI；把 P4 CUDA 11.8／SM61 當明確相容範圍。NVIDIA auto 可優先「符合已支援版本／架構的 CUDA」，探測失敗才依政策退到其他合格 backend 或 CPU。新 NV 型號另補包含目標 CC 的 runtime image 與實機模型 smoke；未完成前標記待驗，不給通用加速承諾。

## Intel：OpenVINO encoder 與 SYCL decoder 是兩項整合

ORT OpenVINO EP 官方支援 Intel CPU／iGPU／dGPU／NPU；目前文件版本表列 ORT1.24.1＋OpenVINO2025.4.1、ORT1.23.0＋2025.3、ORT1.22.0＋2025.1。它有獨立 `onnxruntime-openvino` 套件及 `device_type=GPU`／`GPU.0`，並提供動態 shape 相關選項。這是 **ONNX 執行能力**，沒有宣告本 repo 的 split Qwen encoder 通過。[I1]

固定 b7798 的 SYCL 文件列 Intel 11th Gen Core 起的 iGPU、內建 Arc、Arc／Flex／Data Center Max；舊 iGPU 不能只因品牌相同而視為支援。需要 oneAPI runtime／oneDNN／oneMKL 與 Intel driver。文件含其他 LLM 的速度例子，本文不把那些數字當 Qwen3-ASR 成績。[I2]

可行候選順序：

1. **原生完整候選**：兩份 ONNX 改接 ORT OpenVINO GPU＋同 b7798 的 SYCL decoder。先保持原權重，確認 graph op／FP16／動態 mask 及 embedding 注入，再決定是否需重新匯出／固定 shape。
2. **較小 decoder 整合候選**：經測試可用的 CPU encoder＋b7798 Vulkan decoder。這是混合運算，不可顯示「encoder 已用 Intel 原生 runtime」。Vulkan 不是本輪 ORT 官方 provider 清單中的 Linux encoder EP。[L1、O1]
3. **CPU**：兩個 stage 都用經驗證的 CPU model／runtime 契約。

阻礙包括現有 adapter 缺 OpenVINO 分支、ORT 套件／版本與依賴需換映像、iGPU 世代及共享記憶體限制、SYCL b7798 shared library 封裝與選 device、兩套 GPU runtime 的記憶體競爭。OpenVINO 的 AUTO 只處理 OpenVINO 可見裝置，**不會替 llama 選 SYCL 或替整個 ASR 作 fallback**。[I1、I2、R1–R3]

**本次可安全承諾**：Intel Vulkan／CPU 的既有政策可保留，但原生 OpenVINO＋SYCL 只能列另 image／另實機驗證工作，不能僅改 Compose 或文案就宣稱整合完成。

## AMD：新版以 MIGraphX＋HIP 為候選，舊 ROCm EP 必須註明版本

官方 ROCm EP 文件與 ORT1.23.0 release（2025-09-26）均確認 EP 已自 source tree 移除；官方指向 MIGraphX／Vitis AI。對本 GPU ONNX 路線，MIGraphX 是直接相關的候選；不能把 EP 名稱從 `ROCMExecutionProvider` 改字串就完成遷移。[A1、A3]

MIGraphX EP 官方矩陣列 ORT1.23.2＋ROCm7.2／7.2.1、ORT1.23.1＋ROCm7.1，另列歷史 ORT1.18＋ROCm6.2.x wheels。這些是具體已列出的配對，**不是聲稱它們是 2026-11-01 最新版**，也沒有本 encoder 的支援宣告。MIGraphX 的 FP16／INT8 選項屬額外轉換／量化能力，INT8 有 calibration 契約；不應自動啟用並假定品質相同。[A2]

b7798 build 文件提供 `GGML_HIP=ON`、GPU target 選擇與 ROCm 依賴；同檔也有 Vulkan 路徑。因此候選是 encoder MIGraphX＋decoder HIP，或明示 CPU encoder＋Vulkan decoder。現有 CUDA ORT wheel 不會因此取得 MIGraphX EP。[L1、A2]

### 社群 patch 核對：不能一律說有，也不能一律說只有 decoder

| 一手專案／版本快照 | 已能支持的判斷 | 仍缺的本專案證據 |
| --- | --- | --- |
| AMD 官方 ROCm/TheRock；SUPPORTED_GPUS 最近文件 commit `9ad5d988a39d3afb4f27451b18cdfe9574fbcc49`，2026-10-28 | ROCm/HIP 建置與 ROCm、PyTorch、JAX 發布基礎；支援表明確區分 Build Passing／Sanity Tested／Release Ready。讀取表未列 gfx803 | 不能由「TheRock 有此 gfx」推論 ORT MIGraphX build、b7798 HIP、兩份 encoder 或 ASR 品質已驗。需逐個 artifact／gfx／版本核對。[A4、A5] |
| `scottt/rocm-TheRock` fork，查閱 2026-11-01 | README／releases 有 ROCm／PyTorch 及 gfx1151 等目標的建置／wheel 證據；不是 Qwen3-ASR 套件 | PyTorch 可用不會自動替 ORT 加 provider，也不證明 b7798 C ABI／audio embedding 路徑。[A8] |
| `robertrosenbusch/gfx803_rocm`，README ROCm6.4.0／Torch2.6／Ollama0.9 | 作者明列 Polaris 的 rocBLAS、PyTorch、Ollama、WhisperX／CTranslate2 重建路徑 | WhisperX 是不同模型／runtime；README 不足以支持本 split ONNX encoder＋b7798 ASR 即插即用。[A6] |
| **Schaka/rocm-gfx803**，README 固定 `8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78`，2026-10-06 | 作者新版包含 ROCr／CLR、rocBLAS、MIOpen、MIGraphX、ORT、PyTorch、Triton 與 gfx803 patches；作者記錄 ROCm10.1 image 在 RX470 於2026-10-06通過 imgvalidate／verify。**它確實超過「只有 PyTorch／decoder」的範圍** | 原 repo 的兩個 ONNX graphs／動態 shapes／FP16 op／b7798 HIP ABI／streaming／aligner／完整 ASR 品質與速度，均未由這份紀錄證明；本文也未重測。[A7] |

Schaka 作者特別說明有些 patch 在換 pin 後尚待原始 repro 重驗、不同 ROCm minor 的 component image 混搭可能 import 成功卻算錯；Polaris 甚至有 runtime coherence、GEMM／convolution 正確性 patch。這些是**選擇固定版本專用 experimental image 的直接理由**。不得把作者針對其 host 的 BIOS／kernel／VBIOS 建議自動套用到使用者主機，也不在本輪做這些操作。[A7]

**本次建議**：AMD 支援矩陣同時記錄 gfx、host driver/kernel、ROCm、ORT/MIGraphX、llama commit 與模型 hashes。主線 AMD 原生 image 和 Polaris community image 分開驗證；先採上游支援 target，Polaris 明確 opt-in、固定作者 commit／image digest。任何一個 backend 的存在都不作整體 ASR 支援宣告。

## CPU fallback 的真實底線

「最後回 CPU」必須連 **兩個 encoder sessions、decoder、可選 aligner 與模型 dtype／op** 一起驗證。現有預設 schema 有 INT4 encoder，而 P4 報告採 FP16 encoder，兩者不能默認是同一套 CPU fallback 資產。ORT 官方 FP16 文件提醒 CPU float16 op 支援限制；不能只把 provider 改 CPU 就保證原 FP16 graph 全部可跑。[O2、R1–R3]

fallback 應分清三件事：

- ORT 單圖不支援節點交 CPU，是圖分割。
- encoder 用 CPU、decoder 用 GPU，是明示的混合方案。
- 整個 engine 用 CPU，是完整 fallback，需經模型構造＋實際短音訊成功確認。

建議 auto 模式：探測候選 encoder／decoder 組合；失敗則釋放／重建 engine，在新 worker 中改用已知可行的下一組；最終 CPU 也失敗便回報不健康，不以「有 CPUExecutionProvider」偽裝服務可用。native crash／hang 要靠有期限的獨立程序探測隔離，不能只靠 Python exception。這是後續設計建議，本輪未實作。

若使用者明確指定 strict CUDA／HIP／SYCL，可保留失敗即停止；一般 auto 才允許 fallback。狀態需揭露 requested／resolved encoder EP、decoder backend、實際裝置、fallback 原因與模型識別。不能把舊 P4 嚴格模式的 refusal 與新 auto fallback 政策混寫。

## 是否還需要針對模型優化

**要做模型相容性驗證與負載調校；不必先做廠牌專用權重或重新訓練。** 同一份 GGUF 常可交給不同 llama backend，但仍取決於該 backend 的量化型別／算子實作、模型架構與 b7798 adapter；同一份 ONNX 也要確認 EP 的 dtype、opset、fusions、動態 shape 與 CPU partition。[R1–R3、O1]

建議順序：

1. **保持三份原模型與 ABI**，先讓相符 runtime 真正執行，記錄 encoder／prefill／decode 與整體 HTTP 範圍。這已在既有 P4 案例中證明「不改權重也可能改善」，但本輪不跑測量。[P1]
2. 再調 offload layers、batch／ubatch、context、threads、KV／flash-attention 能力、記憶體預算及 encoder 搬移／分塊；不同 iGPU／dGPU 的最佳值不預設相同。
3. 只有出現具體 op／FP16／動態形狀障礙或顯存／速度瓶頸，才評估同 checkpoint 的 FP32 export、固定 shape、QDQ／INT8、weight-only INT4 等。它們都增加 graph／量化／校準及品質回歸契約，不能叫做純 runtime 切換。[O2、A2]
4. 若改成全 GGUF ASR、新 llama mtmd／其他完整 engine，視為換整體 runtime 與 adapter；不是把 decoder GGUF 換檔。保留語言、context／hotwords、streaming、timestamps／aligner 與中英混語品質門檻。[R2、P2]

## 本次落地範圍建議

| 項目 | 本次可安全採用的政策／標示 | 仍需另 image 或實機證據 |
| --- | --- | --- |
| 一份 Compose | 共用 volumes／ports／設定；明確 image／profile 選擇；auto 後端探測與 CPU fallback 的設計 | 非三廠自動加速證書；GPU 掛載不能無條件要求缺席的裝置。[D1] |
| P4 NVIDIA | 保留 CUDA11.8／cuDNN8／ORT1.18.0／b7798 SM61、既有模型；引用既有報告範圍 | 完整最終 image／服務回歸與當前部署狀態另驗；不能外推所有 NV。[N1–N3、P1] |
| 其他 NVIDIA | 相符 CUDA 為 auto 首選；保留架構／driver 失敗的明確 fallback | 多架構 binary／現代 CUDA image、各 CC／實際 graphs 與效能 |
| Intel | 原生路線明列 OpenVINO encoder＋SYCL decoder；Vulkan decoder 與 CPU 可另列混合模式 | 接 adapter、固定 ORT/OpenVINO/oneAPI、driver／iGPU 世代、整體模型 smoke／品質／速度 |
| AMD 主線 | MIGraphX encoder＋HIP decoder 為原生候選；ROCm EP 僅歷史 pin | ROCm／gfx／ORT 配對 image、b7798 HIP、兩圖相容與完整 ASR |
| AMD Polaris 社群 | 有完整 ORT/MIGraphX stack 的作者證據，明確 experimental／opt-in | pin 所有 component／patch；本模型在指定 gfx803 卡的完整 correctness／服務驗證 |
| CPU | 終端可用路徑是 engine＋CPU 相容模型成功 | FP16／INT4 資產、aligner 及服務功能；不是 provider list 檢查 |

## 來源與查核方法

所有網路查核經 `smart-search` CLI；`doctor --format json` 回傳 `ok: true`。搜索命中只作定位，核心結論回到官方文件、固定 source 或 patch 作者 repo。原始 URL 可由以下命令重查（均以 rtk 開頭）：

```sh
rtk proxy smart-search doctor --format json
rtk proxy smart-search exa-search 'ONNX Runtime ROCm execution provider removed 1.23 MIGraphX' --num-results 4 --include-domains onnxruntime.ai github.com --include-text --format json
rtk proxy smart-search exa-search 'ROCm TheRock gfx-runtimes Polaris gfx803 llama.cpp PyTorch' --num-results 5 --include-domains github.com --include-text --format json
rtk proxy smart-search fetch https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html --format markdown
rtk proxy smart-search fetch https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html --format markdown
rtk proxy smart-search fetch https://raw.githubusercontent.com/Schaka/rocm-gfx803/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78/README.md --format markdown
```

一筆限定 xAI provider 的 `gfx-runtimes` 搜索在60秒後回 `request timed out`，未把其內容作證據；採已取得的官方／作者來源與 GitHub API 做 source-first 核對。未定位名稱保留上述缺口，未改用其他網路工具。

下列官方動態文件均查閱於 **2026-11-01**；不是把頁面現行全部內容追溯套用到舊版本。社群 README 的聲稱屬作者證據，與本輪親測明確分開。

- **N1**：[ORT CUDA EP 相容矩陣](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)：採用 CUDA11.8／ORT1.18.x／cuDNN8 與 CUDA12的1.18.0／1.18.1差別。
- **N2**：[llama b7798 CUDA CMake source](https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-cuda/CMakeLists.txt)：固定 commit 的架構／toolkit gates。
- **N3**：[NVIDIA GPU architecture support 公告](https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers/)：採正文 offline compilation／driver 區分，不採頁面 AI-generated summary 作主證。
- **I1**：[ORT OpenVINO EP](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)：獨立套件、版本表、device 與 shape 選項。
- **I2**：[llama b7798 SYCL 文件](https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/docs/backend/SYCL.md)：固定版本 Intel GPU 世代、runtime、容器／driver 條件。
- **L1**：[llama b7798 build 文件 HIP／Vulkan](https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/docs/build.md)：decoder backend build 能力，不當作 ASR 通過證據。
- **A1**：[ORT ROCm EP 退場公告](https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html)：ORT1.23、ROCm7.0／7.1界線。
- **A2**：[ORT MIGraphX EP](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html)：具體配對、套件、量化條件。
- **A3**：[Microsoft ORT v1.23.0 release](https://github.com/microsoft/onnxruntime/releases/tag/v1.23.0)：2025-09-26發行，明載移除 ROCm EP。原 release 另經 CLI fetch 核對，並與 A1交叉核對。
- **A4**：[AMD TheRock README](https://github.com/ROCm/TheRock/blob/main/README.md)、[RELEASES.md](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)：ROCm／框架與按 gfx 分發的基礎建設。
- **A5**：[TheRock GPU 支援表固定快照](https://github.com/ROCm/TheRock/blob/9ad5d988a39d3afb4f27451b18cdfe9574fbcc49/SUPPORTED_GPUS.md)、[文件 commit API](https://api.github.com/repos/ROCm/TheRock/commits?path=SUPPORTED_GPUS.md&per_page=1)：Build／Sanity／Release分級；日期2026-10-28。
- **A6**：[robertrosenbusch/gfx803_rocm 作者 README](https://github.com/robertrosenbusch/gfx803_rocm/blob/main/README.md)：ROCm6.4、PyTorch／Ollama／WhisperX範圍。
- **A7**：[Schaka Polaris README 固定快照](https://github.com/Schaka/rocm-gfx803/blob/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78/README.md)、[verify.py](https://github.com/Schaka/rocm-gfx803/blob/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78/verify.py)、[commit 日期 API](https://api.github.com/repos/Schaka/rocm-gfx803/commits?path=README.md&per_page=1)：作者RX470／ROCm10.1紀錄與完整 stack／patch 範圍。
- **A8**：[scottt/rocm-TheRock README](https://github.com/scottt/rocm-TheRock/blob/main/README.md)、[作者 releases API](https://api.github.com/repos/scottt/rocm-TheRock/releases?per_page=10)：用於區分 ROCm／PyTorch artifact 與本 ASR 契約。
- **O1**：[ORT provider 架構與 priority](https://onnxruntime.ai/docs/execution-providers/)：provider存在／逐節點 placement 的區分；清單未列一般 Linux Vulkan EP。
- **O2**：[ORT float16／mixed precision](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html)：FP16轉換、blocked ops及CPU限制；不是本圖 CPU 不可用的實測判定。
- **D1**：[Docker Compose GPU access 官方文件](https://docs.docker.com/compose/how-tos/gpu-support/)：host／daemon 前置條件與 device reservation。
- **R1–R3**：上文相對連結的本機 encoder／asr／llama source；另 fetch [原作者 encoder.py](https://github.com/HaujetZhao/CapsWriter-Offline/blob/master/core/server/engines/qwen_asr_gguf/inference/encoder.py) 核對同樣 provider 分支。動態 upstream source 不替代本機快照。
- **P1**：[本 repo P4 CUDA 部署紀錄](p4-cuda-runtime-deployment-20261101.md)；**P2**：[P4 獨占容器實驗](p4-exclusive-container-experiments-20261101.md)、[P4 kernel／硬體證據](p4-asr-hardware-kernel-evidence-20261101.md)。均為既有紀錄，本輪只讀，未重做或延伸其測量結論。
