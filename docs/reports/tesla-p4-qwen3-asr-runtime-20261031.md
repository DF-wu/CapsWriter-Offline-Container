# Tesla P4 的 Qwen3-ASR 效能路線研究

開發分支：`perf/p4-qwen3-asr-dictation-latency`。
過程追蹤：[Issue #12](https://github.com/DF-wu/CapsWriter-Offline-Container/issues/12)；
後續工作依[分階段計畫](../superpowers/plans/2026-11-01-p4-qwen3-asr-dictation-latency.md)記錄假設、commit、實驗與驗收結果。

研究日期：2026-10-31。程式基準：CapsWriter fork `a674e40cb58c5c51991b0e20600a575b7db27944`。
這是來源與相容性研究，沒有測試 GPU、讀取正式容器狀態、修改部署或執行鎖頻。
目前沒有證據可以承諾任何候選在使用者的 P4 上快幾倍。

2026-11-01 補充：使用者已指定首要情境為**錄音放開後更快出字**；以下
評測以短句互動延遲優先，長檔與多人吞吐只作回歸與次要比較。

## 結論與建議順序

有可評測的優化空間，但未找到已證實能在 P4 上更快、完整保留目前功能、
而且可直接替換的現成 runtime。第一優先是保留 Qwen3-ASR **1.7B** 與
CapsWriter 服務層，拆量各階段，再比較 Pascal 相容的 CUDA 解碼器與目前
Vulkan 解碼器；編碼器則評估 I/O Binding、固定前端的小圖重用與資料搬移。
這是工程優先順序，並非已測得的效能排名。

第二階段才評測專用 Qwen ASR 實作。`antirez/qwen-asr` 有原生 CUDA 與圖重播，
但 P4 未獲該專案驗證；`transcribe.cpp` 有同尺寸模型與 CUDA 後端，卻不能
把其 Qwen 的文字輸出介面當成現有串流、字幕、prompt 等功能已全部相容。
主流 vLLM／SGLang／新 TensorRT 的標準 GPU 路線則受 P4 硬體門檻限制。

## 針對「放開後出字」的優先順序

主指標是 `放開按鍵 → 最終可使用文字出現` 的 p50／p95；另外記錄首次
預覽文字時間，以及 ASR 回覆後到實際上屏的時間。提早顯示不穩定 token
不能取代最終文字的指標，也不以大 batch 吞吐或長檔 RTF 決定勝負。

1. **先確認有效 GPU 後端、常駐模型與尾段耗時。** 用同一組 2／5／10 秒
   短句記錄等待、編碼、prefill、生成、合併、用戶端熱詞／規則，以及有啟用
   時的 LLM 後處理與上屏時間。優先改善實際占大頭的階段；不以關閉既有
   校正或降成 0.6B 作為捷徑。
2. **優先評測短請求的資料搬移、launch overhead 與 CUDA 解碼器。** 保留
   1.7B／Q5_K，先比較 warm 與閒置後第一句的 p95；增加 API concurrency
   或檔案 batch 並不等於單句更快。額外鎖頻仍保持使用者選定的關閉狀態。
3. **更大的體感改善方向是錄音期間先處理。** 預設桌面 mic 分段為 60 秒、
   overlap 4 秒，一般短句未達分段門檻，所以主要辨識仍在放開後執行。
   增量 encoder／解碼與最終尾段收斂可把工作前移，但這需要引擎與服務層
   整合。不能只把分段改到極短，否則可能增加重算、排隊與跨句錯誤；必須
   比較中文／混語／專有名詞品質及最終文字穩定性。
4. **桌面與 Web 分開計時。** Web 目前停止錄音只建立完整 Blob，之後由
   使用者提交 HTTP 辨識，不是錄音期間上傳／推論。其提交後還有上傳與
   FFmpeg 等路徑；停止與手動提交之間的人為等待不能歸因於 runtime。
   Web 純文字請求的按需對齊值得評估，但桌面 `mic` 原本就跳過 aligner，
   這項優化不能當作桌面快捷鍵的主要收益。

本節依據：
[桌面分段預設](https://github.com/DF-wu/CapsWriter-Offline-Container/blob/a674e40cb58c5c51991b0e20600a575b7db27944/config_client.py)、
[分段提交條件](https://github.com/DF-wu/CapsWriter-Offline-Container/blob/a674e40cb58c5c51991b0e20600a575b7db27944/core/server/connection/ws_recv.py)、
[Web 錄音生命週期](https://github.com/DF-wu/CapsWriter-Offline-Container/blob/a674e40cb58c5c51991b0e20600a575b7db27944/client/web/src/App.tsx)、
[用戶端後處理](https://github.com/DF-wu/CapsWriter-Offline-Container/blob/a674e40cb58c5c51991b0e20600a575b7db27944/core/client/output/result_processor.py)。

## 現行流程與「保留能力」的基準

- 模型設定指向 **Qwen3-ASR-1.7B**，音訊編碼器為兩個 ONNX session，
  解碼器由 GGUF／llama.cpp 載入。Docker downloader 明確固定
  `Qwen3-ASR-1.7B-q5_k.zip` 及其檔案 hash，故預期解碼器為 **Q5_K**。
  Manifest 中 decoder 約 1.47 GB、兩個 encoder 合計約 0.64 GB；這是
  磁碟檔案大小，不能直接當成推論 VRAM 用量。
  實際使用中的模型仍需核對 hash／metadata，不能僅憑通用 GGUF 檔名推定；
  本研究沒有讀取模型 volume。
  [模型設定](https://github.com/DF-wu/CapsWriter-Offline-Container/blob/a674e40cb58c5c51991b0e20600a575b7db27944/config_server.py)、
  [固定下載來源](https://github.com/DF-wu/CapsWriter-Offline-Container/blob/a674e40cb58c5c51991b0e20600a575b7db27944/docker/server/download_models.py)
- NVIDIA + `default` 的**預期配置**是 ONNX CUDA、GGUF Vulkan；不是全 CUDA。
  `low_vram_gpu` 則把解碼器放 CPU。`auto` 的 GPU bootstrap／probe 失敗時，
  會改成完整 CPU 後端並重驗，因此「選了 GPU」不能證明實際正在用 GPU。
  本研究未查正式部署的有效後端。
  [入口的選擇與回退](https://github.com/DF-wu/CapsWriter-Offline-Container/blob/a674e40cb58c5c51991b0e20600a575b7db27944/docker/server/entrypoint.sh)
- 目前 Docker 使用 CUDA **11.8 / cuDNN 8**，`onnxruntime-gpu==1.18.0`，
  不應直接用新卡的 CUDA 13／BF16 安裝示例覆蓋。
  [Dockerfile](https://github.com/DF-wu/CapsWriter-Offline-Container/blob/a674e40cb58c5c51991b0e20600a575b7db27944/docker/server/Dockerfile)、
  [依賴](https://github.com/DF-wu/CapsWriter-Offline-Container/blob/a674e40cb58c5c51991b0e20600a575b7db27944/requirements/server-docker.txt)
- 服務實際呼叫 `QwenASREngine.decode_stream()`，以外層音訊分段提供互動。
  它保留語言與 context，內部生成完成才交回結果；不是將每個 decoder token
  直接串流到 CapsWriter 用戶端。不能用獨立 CLI 的 token streaming 宣稱
  已等價，也不能拿獨立 `asr()` 的 padding／memory 設定推定服務路徑。
  Qwen adapter 的動態 hotwords 方法目前為空；須保留的是實際存在的
  context／prompt 與服務層能力，而非假定已有 CTC 熱詞搜尋。
  [Qwen adapter](https://github.com/DF-wu/CapsWriter-Offline-Container/blob/a674e40cb58c5c51991b0e20600a575b7db27944/core/server/engines/qwen_asr_gguf/asr_engine.py)
- Qwen 沒有原生 token 時間戳；檔案路徑由外掛 aligner 補齊後供字幕使用。
  更換 ASR runtime 時應保留這個階段、文字格式化、分段合併、HTTP／WebSocket
  契約、取消與 deadline、隱私與佇列規則。多個 HTTP request 可排隊，不代表
  目前單 worker 會做 GPU continuous batching。
  [Pipeline](https://github.com/DF-wu/CapsWriter-Offline-Container/blob/a674e40cb58c5c51991b0e20600a575b7db27944/core/server/worker/pipeline.py)、
  [Worker 排程](https://github.com/DF-wu/CapsWriter-Offline-Container/blob/a674e40cb58c5c51991b0e20600a575b7db27944/core/server/worker/task_handler.py)

另外，HTTP 上傳（包含 Web 錄音）都會成為 `file` 任務，目前即使只要求
純文字也會走對齊嘗試。Docker downloader 沒有自動下載 forced-aligner
模型；若外掛不可用，pipeline 會以均分時長產生 fallback timestamps。
所以不能假定正式部署已具備真正字級強制對齊，更不能把「略過不存在的
對齊計算」算成實測加速。應先查明實際是否載入了 aligner。
[HTTP 任務](https://github.com/DF-wu/CapsWriter-Offline-Container/blob/a674e40cb58c5c51991b0e20600a575b7db27944/fork_server/http_api/transcription_tasks.py)、
[對齊與 fallback](https://github.com/DF-wu/CapsWriter-Offline-Container/blob/a674e40cb58c5c51991b0e20600a575b7db27944/core/server/worker/pipeline.py)

## P4 的硬體邊界

P4 是 Pascal **SM 6.1、8 GB**，不是 T4。NVIDIA 矩陣列其沒有 Tensor Cores。
這會排除不少以新卡 tensor-core attention 為前提的標準套件。
[NVIDIA GPU 清單](https://developer.nvidia.com/cuda-legacy-gpus)、
[P4 規格](https://images.nvidia.com/content/pdf/tesla/184457-Tesla-P4-Datasheet-NV-Final-Letter-Web.pdf)、
[TensorRT 8.6.1 硬體矩陣](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/support-matrix/)

Pascal GP104 類型的原生 FP16 算術吞吐只有 FP32 的 1/64，但支援 DP4A
整數點積；P100 的高速 FP16 不能套到 P4。這不表示 FP16 儲存沒有價值，
也不代表所有標成 FP16 的函式都使用同一種運算；實測要分開看權重儲存、
運算精度、累加與 kernel 實作。[Pascal tuning guide](https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html#fp16-arithmetic-support)

CUDA 13 不再為 SM <7.5 離線編譯，舊架構建置需保留合適的 CUDA 12.9 或更早
工具鏈；這不等於既有舊 CUDA 程式立即不能執行。編譯工具鏈、執行映像、
driver 與 wheel 內含架構需分別核對。
[NVIDIA 架構支援說明](https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers/)

## 候選 runtime 比較

| 路線 | P4／同模型依據 | 保留能力的成本與判斷 |
|---|---|---|
| 現有 ONNX + Pascal CUDA llama.cpp | 固定 b7798 的 CUDA CMake 明列 `61`／DP4A，CUDA <13 的預設架構也包含 61 | **優先 A/B**。維持同一模型／量化與服務層，但需建置、封裝 CUDA runtime；不是改一個 env 就能切換 |
| ONNX encoder I/O Binding／CUDA Graph | ORT 明確提供減少輸入輸出複製與 kernel launch overhead 的機制 | **優先評測**。先做固定形狀 frontend；必須驗證節點都在 CUDA、固定記憶體位址與形狀，動態 backend 不能直接一鍵開啟 |
| `antirez/qwen-asr` | 真正支援 0.6B／1.7B、原生 CUDA、Q8 decoding、圖重播；公開 CUDA 測試是 GB10 SM121 | **第二階段候選**。有語言／prompt／增量模式，但 P4 未驗證、目前 CLI 每次載入模型，需常駐整合及既有 aligner |
| `handy-computer/transcribe.cpp` | 完整 Qwen3-ASR 1.7B port，GGML 有 CUDA／Vulkan，提供量化與準確率驗證資料 | **第二階段候選**。Qwen 路線目前無原生即時 streaming／timestamps；現有外層分段與 aligner 可能保留，但 prompt、語言控制與結果格式仍須驗收 |
| sherpa-onnx Qwen3-ASR | 官方已提供 ONNX 路線與 hotwords；主範例是 0.6B INT8，另連 1.7B 轉換檔 | **需先驗模型／算子**。不能把 0.6B 的速度當成 1.7B；也不能把 CPU INT8 成績當成 P4 CUDA 成績 |
| vLLM | 官方 Qwen ASR 有真正語音支援；套件所用 0.14.0 要 SM≥7.0，現行 main 要 ≥7.5 | **不選作 P4 標準替換**。兩者都排除 SM6.1；社群 Pascal fork 是另一個需維護及驗證的移植項目 |
| SGLang／FlashInfer／FlashAttention | 有 ASR 實作，但目前安裝線及預設 attention kernel 以較新 GPU 為前提 | **不選作 P4 標準替換**。換 attention flag 不足以證明整套依賴在 SM6.1 可用 |
| TensorRT／TensorRT-LLM | TensorRT 11.4.1 要 SM≥7.5；8.6.1 還有已棄用 P4 支援；TRT-LLM 的 Qwen3 文字支援不能等同 ASR | **舊 TRT encoder 是較高成本研究案**，需相容 ORT／TRT／CUDA 組合；新 TRT-LLM 不是現成 P4 解法 |
| CTranslate2／faster-whisper | CT2 的 SM6.1 量化表確實有 INT8/FP32；但 Qwen3 文件是文字 Generator，沒有完整 Qwen3-ASR 轉換流程 | **不當作現成替換**。硬體可用不等於完整語音模型可用；改 Whisper 也不是保留同一模型 |

上述矩陣來源：
[llama.cpp b7798 CUDA 建置原始碼](https://github.com/ggml-org/llama.cpp/blob/b7798/ggml/src/ggml-cuda/CMakeLists.txt)、
[ORT CUDA EP](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)、
[antirez/qwen-asr](https://github.com/antirez/qwen-asr#cuda--nvidia-gpus)、
[transcribe.cpp Qwen contract](https://github.com/handy-computer/transcribe.cpp/blob/main/docs/models/qwen3-asr.md)、
[sherpa-onnx 預訓練 Qwen 模型](https://k2-fsa.github.io/sherpa/onnx/qwen3-asr/pretrained.html)、
[vLLM 0.14.0 requirements](https://github.com/vllm-project/vllm/blob/v0.14.0/docs/getting_started/installation/gpu.cuda.inc.md)、
[vLLM main requirements](https://github.com/vllm-project/vllm/blob/main/docs/getting_started/installation/gpu.cuda.inc.md)、
[SGLang 安裝](https://docs.sglang.ai/get_started/install.html)、
[TensorRT 新版矩陣](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)、
[TensorRT-LLM 矩陣](https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html)、
[CT2 量化](https://opennmt.net/CTranslate2/quantization.html)、
[CT2 模型轉換](https://opennmt.net/CTranslate2/guides/transformers.html)。

## 三個最有價值的實驗

### A：不改模型，先減少現有 pipeline 開銷

先區分純文字辨識與需要時間戳的請求。若實際有載入 aligner，可評估把
對齊改成依 response format／timestamp 需求執行：只求文字時省掉計算，
字幕／時間戳請求仍保留完整對齊。這是保留功能的按需執行設計，需要
明確傳遞任務需求，不能一律關閉 aligner。

目前 ONNX frontend 每約 1 秒 mel 分片呼叫一次 `session.run()`，回傳 NumPy
再接後續階段。值得分別量 frontend launch、GPU→CPU→GPU 複製、backend
encoder、LLM prefill 與 token decode。ORT 建議用 I/O Binding 避免不必要
的複製；固定 frontend 可評估穩定位址的 CUDA Graph。保留有效長度、mask
與 mel／tokenizer 語意，不能單純刪 padding 或改分片以換取表面速度。
[現行 Encoder](https://github.com/DF-wu/CapsWriter-Offline-Container/blob/a674e40cb58c5c51991b0e20600a575b7db27944/core/server/engines/qwen_asr_gguf/inference/encoder.py)、
[ORT 條件與限制](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#using-cuda-graphs-preview)

### B：同 checkpoint／同 GGUF，Vulkan 對 Pascal CUDA

先以目前 b7798 ABI 建置 SM61 CUDA 解碼器，與現行 Vulkan 比較，再考慮升
llama.cpp 版本，避免把 ABI、量化、runtime 和服務重構一次混在一起。
觀察 DP4A／quantized kernel、prefill batch 與短音訊 p95，不能預設 CUDA
一定贏 Vulkan。CUDA 工具鏈與執行映像必須匹配；現有下載、bootstrap marker
與 runtime loader 也要納入候選映像，不能直接替換任意 `.so`。
[b7798 build](https://github.com/ggml-org/llama.cpp/blob/b7798/docs/build.md)、
[b7798 SM61 支援](https://github.com/ggml-org/llama.cpp/blob/b7798/ggml/src/ggml-cuda/CMakeLists.txt)

### C：只在 A／B 後，評測獨立 ASR runtime

`qwen-asr` 有值得測試的原生 CUDA graph 實作，但其 README 指出 1.7B Q8
光 GPU 權重就約 **5.60 GiB**，另需 workspace／KV／runtime；P4 的 8 GB
還必須留給 encoder、可能的字幕 aligner 與其他程式。Q8 同時保留兩份權重
表示不能把它當成單純節省顯存的方案。它的 GB10 warm benchmark 不含模型
載入，不能與 CapsWriter 端到端延遲直接相比。
[專案的記憶體與計時條件](https://github.com/antirez/qwen-asr#cuda--nvidia-gpus)

`transcribe.cpp` 的同尺寸 Qwen GGUF 值得當獨立候選，但其完整模型格式與目前
「split ONNX + decoder GGUF」不同。應保留外層服務與 aligner，再驗證語言、
context／prompt、檔案分段與音訊長度；官方英語 WER 或 Apple GPU 成績不是
P4 上中文／台語／混語品質與延遲的驗證。
[Qwen 模型說明](https://github.com/handy-computer/transcribe.cpp/blob/main/docs/models/qwen3-asr.md)

## 驗收方式：功能與品質先當硬條件

1. 記錄現在的完整映像 digest、模型 hash／GGUF 量化、有效 ONNX provider、
   解碼器實際裝置、driver、可用 VRAM，以及是否發生完整 CPU fallback。
   這一步是後續獲授權的診斷工作，本研究沒有執行。
2. 使用同一組 1.7B checkpoint、音訊、語言／prompt、解碼參數與字幕 aligner；
   先保留現有量化。更換 quantization 另列實驗，不宣稱品質天然無損。
3. 分短錄音 2／5／15 秒、長檔 1／10 分鐘及字幕模式；量 warm/cold、端到端
   p50/p95、RTF、各階段時間、最高 VRAM/RAM。單人延遲與多人吞吐分開。
   現有 pipeline 的 `time_complete` 在外掛對齊前賦值，不能用它代表完整字幕
   request 的耗時；以 client 完成時間另行量測。
4. 中文、繁體用詞、英文混語、數字、專有名詞、噪音、靜音與長段邊界都要
   比較；以 CER/WER、關鍵詞、幻覺／重複、丟字、時間戳偏差與語意正確性
   驗收，不能只用一段 JFK 或英文 test-clean。
5. 功能回歸要涵蓋 Web／HTTP、桌面 mic/file、取消、deadline、字幕格式與
   隱私。新 runtime 若只有 text CLI，透過 adapter 補齊後才是可比較的候選。

不把縮成 0.6B、改用 Whisper／Fun-ASR、關閉字幕對齊、縮短 context 或
略過冷啟動計入「保留能力的加速」。也不假定新 runtime 必須 byte-identical：
浮點運算與 kernel 差異可能改變輸出，應以預先定義的品質條件判定。

## 研究證據與重現

外部資料只使用專案／廠商的一手文件與原始碼。搜尋用 `smart-search
exa-search`，結論建立在隨後 `fetch` 的頁面，不採搜尋模型的總結作證據。
主要本機證據為 `/tmp/p4-*-20261031.md`，分項研究為
`/tmp/tesla-p4-runtime-hardware-20261031.md` 與
`/tmp/tesla-p4-qwen-current-audit-20261031.md`；`/tmp` 不視為可攜式發布附件。

使用的命令形式：

```sh
rtk proxy smart-search exa-search 'Qwen3-ASR runtime CUDA Pascal' --num-results 6 --format json --output /tmp/p4-discovery.json
rtk proxy smart-search fetch 'PRIMARY_SOURCE_URL' --format markdown --output /tmp/p4-source.md
rtk proxy git show fork/master:core/server/engines/qwen_asr_gguf/asr_engine.py
```

本次僅新增研究文件，沒有實作 runtime、安裝依賴、執行 benchmark 或變更
Portainer／容器／模型 volume／GPU 狀態。
