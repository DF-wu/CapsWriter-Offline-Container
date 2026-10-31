# 四卡 Qwen3-ASR：效能、runtime 穩定性與易用性

研究日期：2026-11-01。本次分支：`perf/qwen3-asr-four-gpu-runtime`，基於 `perf/p4-qwen3-asr-dictation-latency` 的 `95912cf66f384aa752f3dda291e02c893070beff`。主追蹤 [#12](https://github.com/DF-wu/CapsWriter-Offline-Container/issues/12)，本次研究 [#13](https://github.com/DF-wu/CapsWriter-Offline-Container/issues/13)。

比較範圍：Tesla P4 8GB、RTX 2080 8GB（非 Super／Ti）、RTX 2060 **6GB**、RX 570 **8GB**。目標仍是同 Qwen3-ASR 1.7B／目前 Q5_K、保留功能，縮短放鍵至最終上屏；本次沒有硬體 benchmark 或穩定性實測。

## 建議與證據界線

**以目前 Linux Docker／Portainer 架構、短句延遲及維護方便為前提，優先評測 RTX 2080，其次 RTX 2060 6GB；已有 P4 則保留它作低功耗 baseline。RX 570 8GB 適合已有卡的隔離實驗，不列為這個服務省事的優先替換。** 這是支援範圍與整合成本的工程判斷，不是四卡實測速度／故障率排名。2060 的建議以完整工作負載能容納於 6GB 為條件。

穩定性分成三件事：上游是否正式支援、專案是否已整合、實機是否通過長時間與錯誤恢復測試。下文只能判斷前兩項；不存在可據以宣稱「四卡都穩」或「RX 570 必定會崩」的測試資料。

## 硬體相對規格

| 項目 | P4 8GB | RTX 2060 6GB | RTX 2080 8GB | RX 570 8GB |
| --- | --- | --- | --- | --- |
| 架構 | Pascal／SM6.1 | Turing／SM7.5 | Turing／SM7.5 | Polaris／GCN |
| 記憶體頻寬 | 192 GB/s | 336 GB/s | 448 GB/s | 224 GB/s，以 MSI ARMOR 8G OC 為例 |
| 相對 P4 頻寬 | 1.00× | 1.75× | 2.33× | 1.17× |
| FP32 峰值 | 約 5.5 TFLOPS | 約 6.45 TFLOPS | 約 10.07；FE 約 10.60 TFLOPS | 約 5.19 TFLOPS，同上板卡推算 |
| 相對 P4 FP32 | 1.00× | 1.17× | 1.83–1.93× | 0.94× |
| NVIDIA Tensor Core | 無 | 有 | 有 | 不適用 |
| 板卡功耗標示 | 最大 75W | 160W | 215W；FE 225W | MSI 150W；其他板卡不同 |

NVIDIA 數據及計算見[三卡規格來源](p4-rtx2080-rtx2060-qwen3-asr-20261101.md)。RX 570 採用 [MSI 的確切 8GB SKU](https://www.msi.com/Graphics-Card/Radeon-RX-570-ARMOR-8G-OC/Specification)：頻寬 `7 Gbit/s × 256 / 8 = 224 GB/s`，FP32 `2048 × 2 × 1.268 GHz = 5.19 TFLOPS`。不能把這些比值當成 ASR 吞吐或出字加速倍數，也不能把一款板卡的功耗推廣到所有 RX 570。

## 目前容器：完整路徑決定易用性

| 顯卡 | 目前預期 encoder＋decoder | 使用／維護判斷 | 尚需驗證的主要風險 |
| --- | --- | --- | --- |
| RTX 2080 | ORT CUDA＋llama b7798 Vulkan | 在這個專案中改動最少的優先候選；8GB 比 2060 有餘裕 | 有效 providers、實際 offload、VRAM 峰值及 idle 後延遲 |
| RTX 2060 6GB | 同上 | 同一 NVIDIA 配置；安裝步驟與 2080 接近 | 6GB 容量；OOM 後可能完整回退 CPU，而非只少加速一部分 |
| P4 | 同上，沿用 Pascal 相容的固定版本 | 舊配置可繼續驗收，不必為了新 runtime 強行換棧 | Pascal 工具鏈／驅動支援終點與後續升級維護 |
| RX 570 8GB | ORT CPU＋llama b7798 Vulkan／RADV | 有既有程式路徑；需要 AMD render node／群組權限，不是 NVIDIA reservation 原樣套用 | encoder 未 GPU 加速；Mesa／Vulkan 功能與解碼器穩定性未實測 |

唯讀原始碼證據：[entrypoint](https://github.com/DF-wu/CapsWriter-Offline-Container/blob/95912cf66f384aa752f3dda291e02c893070beff/docker/server/entrypoint.sh) 在 `/dev/dri` 可見、無 NVIDIA 時啟用 Vulkan 並停用 CUDA；[env mapping](https://github.com/DF-wu/CapsWriter-Offline-Container/blob/95912cf66f384aa752f3dda291e02c893070beff/fork_server/env_config.py) 因而把 encoder 選為 CPU。GPU bootstrap／probe 失敗會完整回退 CPU。`/dev/dri` 存在或 health 為綠，都不是 GPU 推論已成功的證明。

[Dockerfile](https://github.com/DF-wu/CapsWriter-Offline-Container/blob/95912cf66f384aa752f3dda291e02c893070beff/docker/server/Dockerfile) 固定 CUDA 11.8／cuDNN 8 並裝 Mesa；[requirements](https://github.com/DF-wu/CapsWriter-Offline-Container/blob/95912cf66f384aa752f3dda291e02c893070beff/requirements/server-docker.txt) 固定 ORT 1.18.0。[ORT 官方相容表](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#cuda-11x) 列 1.18.x／CUDA 11.8／cuDNN 8.x，但 cuDNN 8 與 9 不相容。版本固定有助重現，不代表舊版永遠免維護；不能直接以最新版 wheel 覆蓋。

## Runtime 選擇：先保留能力，再考慮更換引擎

| 路線 | 支援與成熟度依據 | 易用性／本專案整合成本 | 本次定位 |
| --- | --- | --- | --- |
| 現有 ORT＋llama Vulkan | 已有 adapter、下載、probe、CPU fallback 與 transport | 四卡中 NVIDIA 路線改動較少；RX 570 僅 decoder 有現成 GPU 路線 | **優先 baseline**，量有效後端與端到端延遲 |
| ORT CUDA＋llama CUDA | b7798 有 CUDA 後端；P4 需保留 SM61 及 CUDA <13 的建置線 | 要補建置／下載 checksum／ABI／probe；目前不是改 env 就能換 | **第一批效能實驗**；同模型驗證 CUDA 是否勝過 Vulkan，不預設必勝 |
| Qwen 官方 vLLM | 官方支援 ASR、streaming 與另用 aligner 的時間戳；現行 NVIDIA 最低 SM7.5 | 2080／2060 符合架構門檻，仍須驗證模型格式、精度、attention、VRAM；不是現有 split Q5_K 的 drop-in | **較後面的獨立 runtime 候選**，不因有官方範例就認定比現有部署省事 |
| RX 570 上的 ROCm／HIP | 現行官方矩陣沒有 RX 570／gfx803；ORT ROCm EP 1.23 起移除 | 社群 patch／舊版／自編要自行維護，且不自動提供完整 ASR 支援 | **不作穩定易用的預設方案**；這不等於所有自編路線都不可能 |
| Windows DirectML＋Vulkan | DirectML 支援 DX12／AMD GCN，仍受支援但進入 sustained engineering | 另建 Windows 原生環境、DML wheel／session options；不能放進目前 Linux 容器直接用 | **條件候選**，不是當前 Portainer 的直接優化 |
| transcribe.cpp 完整 Vulkan | 上游有 Qwen encoder＋LM、1.7B 模型及 Vulkan 驗證證據，非只有通用 GGUF 宣稱 | RX 570 未驗證；完整模型產物與既有 split Q5_K 不保證互換；需 adapter、提示詞及字幕回歸 | **RX 570 最值得研究的替代引擎**，仍屬高整合成本，沒有本機穩定證據 |

來源：[llama b7798 build](https://github.com/ggml-org/llama.cpp/blob/b7798/docs/build.md)、[SM61 建置條件](https://github.com/ggml-org/llama.cpp/blob/b7798/ggml/src/ggml-cuda/CMakeLists.txt)、[Qwen 官方](https://github.com/QwenLM/Qwen3-ASR)、[vLLM GPU 要求](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)、[DirectML](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)。AMD 矩陣、ORT ROCm 退役、Mesa 與 transcribe.cpp 的固定來源及限制見 [RX 570 專項查核](rx570-8gb-qwen3-asr-runtime-20261101.md)。

### 需要直接反映在穩定性判斷的限制

- **P4 後續維護**：NVIDIA 官方架構表列 Pascal 最後 CUDA 為 12.x、最後 driver branch 為 R580，Turing 則仍為 ongoing。這是升級支援差異，不是當前固定版本一定不穩。[NVIDIA 官方矩陣](https://docs.nvidia.com/datacenter/tesla/drivers/latest/cuda-toolkit-driver-and-architecture-matrix.html)
- **DML 不是已完成路線**：官方要求 `enable_mem_pattern=False`、`ORT_SEQUENTIAL`，同一 session 不可同時多執行緒 `Run`。現有 encoder 沒有明確關閉 memory pattern，必須先核對／補齊與實測；不能只看到 DML selector 就宣稱即裝即用。新功能移向 WinML 也不等於 DirectML 已停止支援。[官方限制](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html#configuration-options)
- **保留字幕與既有功能**：Qwen 官方 streaming 當前不直接回傳 timestamps；transcribe.cpp 的 Qwen 家族尚未提供原生 streaming／timestamps。改引擎時可評估沿用外層／aligner，但必須驗收，不能當成免費保留。[Qwen streaming 說明](https://github.com/QwenLM/Qwen3-ASR#streaming-inference)、[transcribe 家族移植契約](https://github.com/handy-computer/transcribe.cpp/blob/main/docs/porting/families/qwen3_asr.md)
- **不混改量化**：某 runtime 提供相同 1.7B 或名為 Q5 的模型，不代表既有模型 hash、量化及 encoder 全部等價。更換產物或量化另列實驗，不能混成同模型／同量化的 runtime 結果。

## 後續實測才可回答的問題

沿用[端到端計畫](../superpowers/plans/2026-11-01-p4-qwen3-asr-dictation-latency.md)的同 client monotonic clock、2／5／10 秒、冷／暖／idle 後第一句分組。GPU 規格不能代替主指標 p50／p95。

穩定性驗收應另外記錄重啟與模型載入成功、重複／長時間聽寫的錯誤與 VRAM 漂移、長音檔與字幕、取消／deadline、OOM 復原、有效 provider／offload 是否意外回退 CPU；比較環境需固定 image digest、模型 hash、driver 及 Mesa／runtime 版本。若需要驅動替換、引擎編譯或手動 patch，將步驟與維護成本記入 issue。不能以 CPU fallback 後服務仍健康當成 GPU 路線穩定通過。

本次僅研究文件；沒有操作正式 Portainer、容器、GPU 時脈或模型 volume，沒有測得四卡速度或故障率。

## 來源留存

本次補充取回的證據：`/tmp/gpu-study-ort-cuda-20261101.md`、`/tmp/gpu-study-ort-directml-20261101.md`、`/tmp/gpu-study-llama-b7798-build-20261101.md`、`/tmp/gpu-study-llama-b7798-sm-20261101.md`、`/tmp/gpu-study-qwen-official-20261101.md`、`/tmp/gpu-study-nvidia-architecture-20261101.md`；AMD 證據清單位於專項報告。只採用實際取回並閱讀的來源，未把錯誤頁當成支援證據。

代表取回命令：

```sh
rtk proxy smart-search fetch https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html --format markdown --output /tmp/gpu-study-ort-directml-20261101.md
rtk proxy smart-search fetch https://docs.nvidia.com/datacenter/tesla/drivers/latest/cuda-toolkit-driver-and-architecture-matrix.html --format markdown --output /tmp/gpu-study-nvidia-architecture-20261101.md
```
