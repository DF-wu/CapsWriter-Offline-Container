# Tesla P4 選用優化

適用於 fork-v2.2.0 候選版的 Linux NVIDIA Docker Server。此版本尚在發版準備。

在 Portainer Stack 的環境變數或 Compose `.env` 設定：

```dotenv
CAPSWRITER_TESLA_P4_OPTIMIZATION=true
```

使用包含新 runtime 的 server image，取消唯一 `docker-compose.yml` 內 NVIDIA
區塊的註解以暴露 GPU（見[單檔指南](compose.md)），再啟動：

```sh
docker compose up -d
```

這是**執行時選用旗標**，不需要在 host 安裝 CUDA／編譯器。runtime 在 image 的免洗
build stage 內編譯，權重仍自動使用既有下載流程。此開關不調整時脈、功耗、風扇、
權限或模型權重，也不會啟用另一個獨立的 NVIDIA boost 功能。

## 啟用與預設行為

| 設定 | 行為 |
|---|---|
| 未設定或 `false`（預設） | 保留原 CPU／Vulkan 選擇，不註冊或使用 bundled SM61 runtime |
| `true` ＋ NVIDIA GPU 可見 ＋ bundle 齊備 | 選用 b7798 CUDA 11.8 SM61 decoder |
| `true`，但沒有 NVIDIA GPU 或 bundle | 明確拒絕啟動，不偷偷回落到 CPU |
| `CAPSWRITER_INFERENCE_HARDWARE=cpu` 或 Qwen `cpu_only` | CPU 優先，旗標不強迫使用 CUDA |
| Qwen `low_vram_gpu` | ONNX encoder 使用 GPU，decoder 保持 CPU |
| `CAPSWRITER_LLAMA_BACKEND=cpu`／`vulkan` | 明確選擇優先，不讀取或使用 P4 CUDA bundle |
| `CAPSWRITER_LLAMA_BACKEND=cuda` | 同樣必須設定 P4 旗標，不能繞過 opt-in |

接受 `true/false`、`1/0`、`yes/no`、`on/off`，忽略大小寫與前後空白。
更改旗標後需要重新啟動套用新環境變數的容器；僅修改 Stack UI 而沒有重新部署，
既有容器的注入環境不會改變。保留所有原有 Portainer 參數與 model/log volumes。

已選定 CUDA 時，啟動 probe 會要求實際註冊 `CUDA0`，並檢查可見 CUDA 裝置都是
Tesla P4；P40、RTX、混合 P4／其他 GPU 都不會被當成已驗證硬體。多 GPU 主機應先
透過 NVIDIA device reservation／可見性設定只暴露 P4。原本的 Vulkan／CPU 模式
仍保留既有 fallback 行為，選定 P4 CUDA 後的錯誤則會中止啟動。

## 使用的優化

- 原 FP16 ONNX encoder＋Q5_K_M GGUF decoder，沒有再量化或替換模型。
- 固定 llama.cpp b7798 commit `c301172f660a1fe0b42023da990bf7385d69adb4`，使用
  CUDA 11.8、`CMAKE_CUDA_ARCHITECTURES=61` 與可重定位 `$ORIGIN`。
- 驗證原始碼 archive、runtime archive 與每份 DSO 的 SHA-256，沿用既有 atomic
  staging/promote。Windows／CPU／其他 GPU 不會自動使用此 runtime。
- 保留 `appuser`，不需要 NVIDIA_BOOST 區塊的 root 或額外 capabilities。

此處的 P4 專用部分是 runtime／kernel，GGUF 權重格式仍是原 Q5_K_M。GPU decoder
會增加顯存占用；先前本機觀測約由 1.2 GiB 增至 2.9 GiB，與 Immich 競爭時須留足
顯存及散熱餘裕。

## Prefill 與音訊長度調校

P4 flag 只選用已驗證 decoder runtime，不改 batch、ubatch、context 或模型精度。
既有短音訊基準為 requested batch4096／effective2048、ubatch512、context2048。
較低 micro-batch 可能省暫存，但未證明更快；Server `CHUNK_SIZE=80` 超過即截斷，
不能靠減小該值取得同品質加速。完整參數取捨、音訊位置預算與試驗矩陣見
[Qwen prefill 調校指南](qwen-tuning.md)。

## 實測與限制

[原始實測報告](reports/p4-cuda-runtime-deployment-20261101.md)記錄 P4 的 HTTP
p50 約提升 4.03–5.25 倍，基準是當時 Vulkan 未找到裝置而使用 CPU decoder 的原服務。
先前修好 EGL 的 Vulkan 與 CUDA 引擎測試，10 秒音訊延遲分別為 0.8649／0.5831 秒。
這些是歷史 runtime 證據，不代表 2.2 正式 release image 已發佈或重新取得相同倍率。

五份完整音訊的 greedy 正規化輸出相同。既有 HTTP worker 實際仍使用預設
temperature=0.4，即使 API 接受 0；本版未改動此行為。完整 image、Windows 打包、
發版 SHA／digest 與 CI 狀態見 [2.2 發版準備](releases/fork-v2.2.0.md)。

## 關閉與升級

將旗標設為 `false` 或移除，並以原 Stack 配置重新部署新 image，即恢復原本的
CPU／Vulkan 選擇。CPU／Vulkan runtime 會透過既有下載驗證流程準備。保留既有
models／logs volumes，不需刪除或重抓模型權重。

上一輪同 ID 的本機 writable-layer hotfix 沒有這個旗標，普通 restart 仍保留該版
行為。要取得這個 opt-in 契約，必須部署本 PR 的新版 source image；此 PR／發版準備
不會修改現行容器或 Portainer 注入參數。
