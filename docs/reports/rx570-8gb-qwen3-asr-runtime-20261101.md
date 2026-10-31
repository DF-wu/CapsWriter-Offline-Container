# RX 570 8GB：CapsWriter Qwen3-ASR 1.7B Q5_K 執行路徑查核

本次分支：`perf/qwen3-asr-four-gpu-runtime`（基於 `perf/p4-qwen3-asr-dictation-latency`）。研究追蹤 [#13](https://github.com/DF-wu/CapsWriter-Offline-Container/issues/13)，主追蹤 [#12](https://github.com/DF-wu/CapsWriter-Offline-Container/issues/12)；[四卡綜合比較](qwen3-asr-four-gpu-runtime-20261101.md)。

查核日：2026-11-01。僅查閱現有原始碼與第一手文件；未部署、未啟動 GPU／模型、未改動執行設定，也未測試延遲或穩定性。本文是支援範圍研究，不是效能測試。

## 判斷

**現有 Linux CapsWriter 最接近既有架構的 RX 570 路徑，是 ORT CPU encoder＋llama.cpp b7798／Mesa RADV Vulkan decoder。** 上游證據支持這條路徑具備工程可行性，尚無本機端到端驗證。RX 570 不在目前 ROCm／vLLM AMD 官方支援清單，不能當成現有 CUDA encoder 的直接替換。若主要目標是縮短放開聽寫鍵後的等待，單獨加速 decoder 的效果仍受 CPU encoder 與音訊前處理限制，沒有依據承諾秒數。

## 硬體：必須指定板卡版本

- RX 570 是 Polaris／第 4 代 GCN；以下依 Polaris／gfx803 架構家族評估。SAPPHIRE 官方 8GB 頁確認 Polaris／GCN 世代；LLVM 23.1.0 的架構表將 Polaris10 歸於 gfx803，但該表產品欄主要列 RX 470／480，並非本機 PCI ID 的查驗。[SAPPHIRE](https://www.sapphiretech.com/en/consumer/pulse-rx-570-8g-g5)、[LLVM 架構表](https://releases.llvm.org/23.1.0/docs/AMDGPUUsage.html)
- **MSI RX 570 ARMOR 8G OC** 官方規格：8GB GDDR5、256-bit、7000 MHz 記憶體速率、2048 單元、最高 1268 MHz、150W。由規格推算頻寬 `7 Gbit/s × 256 ÷ 8 = 224 GB/s`；FP32 理論峰值約 `2048 × 2 × 1.268 GHz = 5.19 TFLOPS`。這兩項是理論推算，不能換算 ASR 延遲，亦非持續可達吞吐量。[MSI 8GB 規格](https://www.msi.com/Graphics-Card/Radeon-RX-570-ARMOR-8G-OC/Specification)
- **不要將「120–150W」寫成全部 RX 570 8GB 的已證實範圍**：本次可直接確認 MSI 為 150W，SAPPHIRE PULSE RX 570 8GB（SKU 11266-36）卻標示 `<180W`，且 boost 1284 MHz。不同板卡的時脈、功耗標示不同；120W 未在本次取回的 8GB 官方頁獲證實。規格功耗也不是聽寫時實測耗電。[SAPPHIRE 規格](https://www.sapphiretech.com/en/consumer/pulse-rx-570-8g-g5)

## 現有 Linux Docker／Portainer 路徑

本地唯讀證據：`docker/server/Dockerfile:1` 固定 CUDA 11.8.0／cuDNN 8／Ubuntu 22.04；`:17` 安裝 Mesa Vulkan 驅動；`requirements/server-docker.txt:4` 固定 `onnxruntime-gpu==1.18.0`；`docker/server/download_models.py:416`、`:425` 固定 llama.cpp b7798 CPU／Vulkan 套件。`core/server/engines/qwen_asr_gguf/inference/encoder.py:130–159` 使用兩個 ORT ONNX sessions，候選為 CPU／CUDA／DML／TensorRT，沒有 Vulkan encoder。`docker/server/entrypoint.sh:40–49` 在非 NVIDIA 的 Vulkan 分支停用 CUDA；裝有 CUDA 底層映像，不代表 AMD 能執行 CUDA EP。[ORT CUDA 官方文件](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) 明確以 NVIDIA CUDA-enabled GPU 為對象，並列出 ORT 1.18.x／CUDA 11.8／cuDNN 8.x 的版本組合。

### Vulkan decoder：有依據的候選，非 RX 570 保證

- Mesa 官方 RADV 文件支持 Linux kernel 所支援的 GCN／RDNA 顯卡，從 GCN 1 起；查核日文件列 GFX8 以上可提供 Vulkan 1.4。**不能以 Polaris 老舊為由斷言「不支援 Vulkan 1.2」**。這是目前上游文件能力，不保證 Ubuntu 22.04 容器實際安裝的 Mesa 版本或主機驅動狀態。[RADV 支援硬體](https://docs.mesa3d.org/drivers/radv.html#supported-hardware)
- **精確到 b7798**：`ggml-vulkan.cpp` 檢查 loader／instance API 至少 Vulkan 1.2（約 5412–5416 行）；必要裝置功能包括 `storageBuffer16BitAccess`（約 4990–4995、15209 行）。`shaderFloat16` 則僅決定 FP16 運算路徑是否啟用（約 4714–4716、4875 行）；缺少原生 FP16 運算不等於缺少 16-bit storage，也不能據此宣告整個後端不可用。[b7798 Vulkan 原始碼](https://github.com/ggml-org/llama.cpp/blob/b7798/ggml/src/ggml-vulkan/ggml-vulkan.cpp)
- Cooperative matrix 由 shader 編譯器偵測與裝置 extension／feature 決定，程式具有非 coopmat 路徑。b7798 的根 GGML CMake 沒有可照搬的 `GGML_VULKAN_SHADERS_FLOAT16`／`GGML_VULKAN_COOPMAT` 開關；勿把其他版本的 build 指令套在此版本。此處的 FP16／coopmat 可選性，主要由原始碼中的能力分支與 `GGML_VK_DISABLE_F16`／`GGML_VK_DISABLE_COOPMAT` 佐證，本文沒有設定它們。[b7798 GGML CMake](https://github.com/ggml-org/llama.cpp/blob/b7798/ggml/CMakeLists.txt)、[Vulkan CMake](https://github.com/ggml-org/llama.cpp/blob/b7798/ggml/src/ggml-vulkan/CMakeLists.txt)
- 工程推論：既有套件與 Mesa 讓 decoder 具備較少改動的候選路線；仍需實際驅動功能、容器 render node 權限、GPU 選擇與記憶體配置均成立。`/dev/dri` 存在只是現有啟動腳本的偵測條件，不能證明模型已成功 offload。

### Encoder：CPU 有既有程式路徑，ROCm 不是現成替代

- ORT 官方 EP 清單沒有通用 Vulkan EP；現有 encoder 也沒有這條分支。因此 decoder 的 Vulkan 支援不會自動加速兩個 ONNX encoder。CPU EP 是現有 fallback，但本次未載入 ONNX，未驗證其算子、資料型別與完整執行相容性。[ORT EP 清單](https://onnxruntime.ai/docs/execution-providers/)
- AMD 現行 ROCm 相容矩陣（抓取頁標示 **10.1.0**）以及固定的 **6.3.3** 矩陣都沒有 RX 570／gfx803。這是「沒有目前官方支援」，不是宣稱所有社群 patch、自編 HIP 或舊軟體都不可能運作；它們需要另一套建置與維護責任。[現行矩陣](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)、[6.3.3 矩陣](https://rocm.docs.amd.com/en/docs-6.3.3/compatibility/compatibility-matrix.html)
- ORT ROCm EP **自 1.23 移除**；AMD 官方最後提供此 EP 的 ROCm 發行版是 **7.0**，7.1 起移除，官方建議遷移 MIGraphX。這不代表專案固定的 ORT 1.18.0 已移除 ROCm EP；但目前安裝的 CUDA wheel 並非 ROCm wheel，現有 encoder 也未選用 ROCm。[ORT ROCm 退役邊界](https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html)
- 現行 vLLM 文件列 ROCm ≥6.3，AMD GPU 為 gfx90a／gfx942／gfx950／gfx1100、1101／gfx1200、1201／gfx1150、1151 等；RX 570／gfx803 不在清單。改用 vLLM 不能視為 RX 570 的官方支援捷徑。[vLLM GPU 安裝與需求](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#amd-rocm)

## Windows：DirectML encoder＋Vulkan decoder

ORT DirectML 官方要求 DirectX 12 裝置，列 AMD 第 1 代 GCN 起的相容範圍；RX 570 符合硬體世代條件。這是 Windows 候選路線，不能直接移植成目前 Linux Docker 的 DML 路線。DirectML 已進入 **sustained engineering**：仍受支援，但新功能開發移到 WinML，不能稱它已停止支援。[ORT DirectML](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)

官方要求 `enable_mem_pattern=False`、`execution_mode=ORT_SEQUENTIAL`；同一 session 不可多執行緒同時 `Run`，不同 sessions 才可各自同時執行。現有 Qwen encoder 建立 SessionOptions 時**沒有明確停用 memory pattern**；因此有 DML provider 分支，仍不足以認定可直接使用。確切 ONNX opset／算子、動態長度與 padding、Windows 原生函式庫及 worker 排程都需核對。這次未作任何修正，也未把此靜態發現稱為已重現的崩潰。

## 完整模型 Vulkan 替代路線：辨別專案與證據強度

- **antirez/qwen-asr**：查到的主線 commit `924694251d9e0f18e5d86bbd06aa3ab5f870002d`（2026-09-30）提供 CPU、ROCm、CUDA；ROCm 文件限定實測 ROCm 7.1／Radeon 8060S gfx1151，明言不代表每張 ROCm 卡皆支援。未取得其 Vulkan 後端證據，不可把這個 qwen-asr 寫成 RX 570 全 Vulkan 路線。[該專案 README](https://github.com/antirez/qwen-asr/tree/924694251d9e0f18e5d86bbd06aa3ab5f870002d)
- **handy-computer/transcribe.cpp**：證據超過「有 GGUF」：Qwen3-ASR 家族移植文件明確描述 audio encoder＋causal LM，記錄 CPU／Metal／Vulkan 中間張量與端到端驗證；1.7B 專屬模型頁列出 RADV RENOIR 的 Vulkan 執行記錄。固定查閱的 `cd0ea568` 模型頁亦列 Q5_K_M 格式。故可列為**有上游模型＋後端證據的完整 Vulkan 引擎候選**，本文不引用其跨硬體時間數字。[家族移植文件](https://github.com/handy-computer/transcribe.cpp/blob/main/docs/porting/families/qwen3_asr.md)、[固定 commit 的 1.7B 模型頁](https://github.com/handy-computer/transcribe.cpp/blob/cd0ea568/docs/models/qwen3-asr-1.7b.md)、[建置文件](https://github.com/handy-computer/transcribe.cpp#build)
- 仍無 RX 570 專屬驗證；RENOIR 不是 Polaris。transcribe.cpp 家族文件也明確未提供 Qwen3-ASR streaming、timestamps、VAD、speaker diarization。完整 GGUF 與 CapsWriter 的「兩個 ONNX＋decoder-only GGUF」不是已證實可互換格式，不能直接沿用現有 Q5_K 權重或 llama ABI。需要 adapter／模型產物／提示詞與功能回歸，工程量高於保留既有 split pipeline。

## 放開按鍵延遲、穩定性與使用難度的界線

**工程推論**：放鍵到顯示文字包含尚未完成的上傳／排隊、前處理與 encoder、decoder prefill／逐 token 生成、後處理／回傳；已在按鍵期間完成的工作不應重複計入。224 GB/s、約 5.19 TFLOPS、8GB 或 decoder 能 offload，都無法單獨推出這段端到端秒數。8GB 是容量條件，仍須加上 KV cache、暫存、driver／顯示占用；不能保證任意音長都不 OOM。

就改造負擔而言：既有 Linux CPU encoder＋Vulkan decoder 最接近目前程式；Windows DML 需額外 EP／session／原生 runtime 整合；transcribe.cpp 全 Vulkan 需替換引擎；gfx803 ROCm／HIP 社群路線不具目前官方支援保證。這是程式與支援邊界的比較，不是實測穩定性排名。**本次沒有 crash test、長時間運作、重複聽寫、並行與 OOM 測試，不能稱 RX 570 已穩定或必定不穩定。**

## 證據留存與缺口

所有網頁均經 smart-search CLI 取回；代表指令：

```text
rtk proxy smart-search exa-search 'qwen-asr transcribe.cpp Vulkan Qwen3 ASR encoder github' --num-results 4 --include-text --format json --output /tmp/rx570-full-vulkan-search.json
rtk proxy smart-search fetch https://raw.githubusercontent.com/ggml-org/llama.cpp/b7798/ggml/src/ggml-vulkan/ggml-vulkan.cpp --format markdown --output /tmp/rx570-b7798-vulkan.md
```

主要快照：`/tmp/rx570-msi-8gb.md`、`rx570-sapphire-8gb.md`、`rx570-mesa-radv.md`、`rx570-b7798-vulkan.md`、`rx570-b7798-vulkan-cmake.md`、`rx570-b7798-ggml-cmake.md`、`rx570-ort-ep.md`、`rx570-ort-cuda.md`、`rx570-ort-dml.md`、`rx570-ort-rocm.md`、`rx570-rocm-matrix.md`、`rx570-rocm-633.md`、`rx570-vllm-gpu.md`、`rx570-transcribe-port.md`、`rx570-transcribe-17b-pinned.md`、`rx570-antirez-qwen.md`、`rx570-llvm-amdgpu.md`。除明列 tag／commit／版本者，其餘是查核日浮動文件快照；它們不是目前機器的安裝版本證明。

缺口：實際板卡 SKU／VBIOS、主機 kernel／Mesa／Vulkan feature、容器裝置權限、實際 ONNX dtype／opset／算子、本機 CPU encoder 時間及放鍵事件時間線、RX 570 Q5_K 端到端穩定性皆未驗證。Mesa GitLab `amd_family.h` 的 fetch 回傳驗證頁，沒有當成原始碼證據；誤猜的 transcribe.cpp 原始碼路徑回傳 404，亦未採用。
