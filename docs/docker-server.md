# CapsWriter Server Docker 部署

這份文件只覆蓋 **Server 端**。Client 仍維持原本 Windows 用法。

## 目標

- 以 Docker Image 方式封裝 CapsWriter-Offline Server
- 在 Linux / Docker headless 環境中可穩定啟動
- 模型檔與日誌走宿主機 volume，不把大模型烤進映像
- 讓 `qwen_asr` 的設定行為、啟動行為、與文件描述保持一致

## 內容物

- `docker/server/Dockerfile`：Server 專用 production image
- `docker-compose.yml`：本機啟動與模型準備
- `docker/server/download_models.py`：自動下載官方模型資產
- `docker/server/healthcheck.py`：容器健康檢查
- `docker/server/probe_backend.py`：後端探測與驗證
- `requirements-server-docker.txt`：Linux/headless 最小依賴集

## 預設策略

- 預設模型：`qwen_asr`
- 預設 Qwen profile：`default`
- `qwen_asr` 的官方 `default` profile = **ONNX GPU + llama CPU**
- 若 GPU runtime 不可用，Qwen 會自動回退到 CPU
- Docker 內強制關閉 tray；DirectML 預設關閉；Qwen backend 改由 preset + policy 決定
- 模型掛載到 `./models`，日誌持久化到 Docker named volume

> 目前公開 image 的優先目標是 **Tesla P4 / Pascal** 這類已驗證硬體。其他 GPU 世代的廣泛相容性仍列為後續 TODO。

## 啟動前準備

```bash
cp .env.example .env
cp hot-server.example.txt hot-server.txt
```

根目錄 `.env` 只作為本機啟動設定使用，已被 Docker build context 排除，不會被打進 image。

## 1. 準備公開 image

預設情況下，這份 compose 會直接使用公開 image：

```text
ghcr.io/df-wu/capswriter-offline-server:latest
```

你可以在 `.env` 裡覆蓋成別的 tag 或私有 image，但大多數情況不需要。

## 2. 啟動 Server

```bash
docker compose up -d capswriter-server
```

預設會在容器啟動時自動下載 `qwen_asr` 所需模型，並依 `CAPSWRITER_QWEN_PRESET` + `CAPSWRITER_INFERENCE_HARDWARE` 解析 Qwen runtime profile。

如果你要切到 `fun_asr_nano`，先在 `.env` 裡設定：

```env
CAPSWRITER_MODEL_TYPE=fun_asr_nano
```

可選值：

- `fun_asr_nano`
- `qwen_asr`

第一次冷啟動可能需要比較久，因為容器會先下載模型與 `llama.cpp` 共享庫，再啟動服務。

---

## 3. Qwen 官方 preset

`CAPSWRITER_QWEN_PRESET` 目前只保留兩個官方值：

- `default`
- `cpu_only`

### `default`

這是本 fork 目前對 **Tesla P4 / Pascal** 的正式推薦路線：

- ONNX encoder：GPU（CUDA，可用時）
- llama / GGUF：CPU

這條路徑對目前已驗證硬體的成本 / 延遲比最好，因此被定義成官方預設。

### `cpu_only`

這會強制 Qwen 整條路徑走 CPU：

- ONNX encoder：CPU
- llama / GGUF：CPU

適合：

- 不想請求 GPU
- GPU 需要讓給別的服務
- 要做純 CPU 排障或對照 benchmark

### 相容 alias

舊值仍會被接受，但會在解析時正規化：

- `low_vram_gpu` -> `default`
- `balanced` -> `default`
- `quality` -> `default`
- `defaults` -> `default`
- `cpu` -> `cpu_only`

**文件、compose 範例、與新設定檔都只應該寫官方值。**

---

## 4. Inference hardware 策略

`CAPSWRITER_INFERENCE_HARDWARE` 是 **硬體政策約束**，不是直接的 backend 選單。

可選值：

- `auto`
- `gpu`
- `cpu`

### `auto`

- 允許 preset 使用 GPU
- 若 runtime 不可用，自動回退 CPU
- 不讓服務因為 GPU 缺失直接失敗

### `gpu`

- 要求優先嘗試 GPU
- 若 runtime 不可用，仍安全回退 CPU
- 適合你希望明確以 GPU 優先啟動，但不要因 probe 失敗而整個炸掉

### `cpu`

- 無論 preset 或 advanced override 怎麼寫，最後都強制 CPU
- 這是最高優先級的硬體約束

---

## 5. Qwen 設定解析優先序

這是 `qwen_asr` 目前的**正式解析規則**。

### 解析順序

1. 內建 baseline
2. preset normalize
3. 套用 preset defaults
4. 套用顯式 env override
5. 套用 `CAPSWRITER_INFERENCE_HARDWARE` 約束
6. 若所需硬體不可用，套用 runtime fallback

### 什麼叫做「顯式 env override」

只有在某個 env key **有非空值** 時，才算覆寫。

例如：

```env
CAPSWRITER_QWEN_N_CTX=1536
```

這會覆蓋 preset 內建的 `n_ctx`。

但若你寫成：

```env
CAPSWRITER_QWEN_N_CTX=
```

就視為**未設定**，不會覆蓋 preset。

### 精確優先序表

| 階段 | 來源 | 例子 | 作用 |
| --- | --- | --- | --- |
| 1 | baseline | `n_ctx=2048` | 最底層預設 |
| 2 | preset normalize | `defaults -> default` | 正規化舊值 / alias |
| 3 | preset defaults | `default -> use_cuda=true` | 載入完整 profile |
| 4 | explicit env override | `CAPSWRITER_QWEN_N_CTX=1536` | 覆蓋 preset tuning |
| 5 | hardware policy | `CAPSWRITER_INFERENCE_HARDWARE=cpu` | 強制 CPU 約束 |
| 6 | runtime fallback | 容器看不到 NVIDIA runtime | 安全回退 |

---

## 6. Qwen 設定面分層

### 6.1 主要設定面

這些是正常使用時最應該看的欄位：

- `CAPSWRITER_MODEL_TYPE`
- `CAPSWRITER_QWEN_PRESET`
- `CAPSWRITER_INFERENCE_HARDWARE`
- `CAPSWRITER_GPU_DEVICE_COUNT`
- `CAPSWRITER_NUM_THREADS`
- `CAPSWRITER_SERVER_PORT`
- `CAPSWRITER_LOG_LEVEL`

### 6.2 Qwen tuning 覆寫

這些是可以安全調整的 Qwen tuning 欄位；若填了非空值，就會覆蓋 preset defaults：

- `CAPSWRITER_QWEN_CHUNK_SIZE`
- `CAPSWRITER_QWEN_N_CTX`
- `CAPSWRITER_QWEN_MEMORY_NUM`
- `CAPSWRITER_QWEN_PAD_TO`
- `CAPSWRITER_QWEN_N_PREDICT`
- `CAPSWRITER_QWEN_LLAMA_N_BATCH`
- `CAPSWRITER_QWEN_LLAMA_N_UBATCH`
- `CAPSWRITER_QWEN_LLAMA_FLASH_ATTN`
- `CAPSWRITER_QWEN_LLAMA_OFFLOAD_KQV`

### 6.3 Advanced / internal backend override

這些欄位仍被支援，但不建議一般使用者碰：

- `CAPSWRITER_QWEN_USE_CUDA`
- `CAPSWRITER_QWEN_VULKAN_ENABLE`
- `CAPSWRITER_QWEN_USE_DML`
- `CAPSWRITER_QWEN_VULKAN_FORCE_FP32`

若你顯式設定它們，會先覆蓋 preset backend defaults；但後面仍可能被：

- `CAPSWRITER_INFERENCE_HARDWARE=cpu`
- runtime fallback

再次收斂或改寫。

---

## 7. Compose 層 GPU request

Compose 額外提供 `CAPSWRITER_GPU_DEVICE_COUNT`：

- `all`：向容器請求 GPU（預設）
- `0`：不向容器請求 GPU，適合 CPU-only 啟動

如果你要用同一份 compose 直接走 CPU-only 啟動，可以這樣：

```bash
CAPSWRITER_GPU_DEVICE_COUNT=0 \
CAPSWRITER_INFERENCE_HARDWARE=cpu \
docker compose up -d --force-recreate capswriter-server
```

---

## 8. 直接使用 image

如果你要直接使用 image，也可以：

```bash
docker run -d --name capswriter-server \
  --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -e CAPSWRITER_QWEN_PRESET=default \
  -e CAPSWRITER_INFERENCE_HARDWARE=auto \
  -p 6016:6016 \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/hot-server.txt:/app/hot-server.txt" \
  ghcr.io/df-wu/capswriter-offline-server:latest
```

改成 Qwen 純 CPU：

```bash
docker run -d --name capswriter-server \
  -e CAPSWRITER_QWEN_PRESET=cpu_only \
  -e CAPSWRITER_INFERENCE_HARDWARE=cpu \
  -p 6016:6016 \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/hot-server.txt:/app/hot-server.txt" \
  ghcr.io/df-wu/capswriter-offline-server:latest
```

改成 `fun_asr_nano`：

```bash
docker run -d --name capswriter-server \
  -e CAPSWRITER_MODEL_TYPE=fun_asr_nano \
  -e CAPSWRITER_INFERENCE_HARDWARE=auto \
  --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -p 6016:6016 \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/hot-server.txt:/app/hot-server.txt" \
  ghcr.io/df-wu/capswriter-offline-server:latest
```

---

## 9. 啟動時你應該看到什麼

當模型是 `qwen_asr`，server 啟動 log 會輸出已解析的 Qwen 設定摘要，例如：

```text
[qwen-config] raw_preset=defaults normalized_preset=default profile=onnx_gpu_llama_cpu inference_hardware=auto
[qwen-config] onnx_backend=cuda llama_backend=cpu gpu_visible=True nvidia_visible=True
[qwen-config] n_ctx=1536 chunk_size=30.0 memory_num=1 pad_to=15 n_predict=512 n_threads=32
[qwen-config] llama_n_batch=4096 llama_n_ubatch=512 flash_attn=True offload_kqv=True
```

這些行可以直接用來判斷：

- preset alias 是否被正規化
- ONNX 最後是不是走 CUDA
- llama 最後是不是停在 CPU
- tuning 值到底是哪一組

---

## 10. 查看狀態

```bash
docker compose ps
docker compose logs -f capswriter-server
```

健康檢查通過後，Server 會在：

```text
ws://127.0.0.1:${CAPSWRITER_SERVER_PORT}
```

預設是：

```text
ws://127.0.0.1:6016
```

---

## 11. 停止

```bash
docker compose down
```

---

## 12. 驗證標準

完成部署後，至少要看到：

1. `docker compose ps` 顯示 `healthy`
2. log 出現模型載入完成與 WebSocket 監聽訊息
3. `qwen_asr` 時，log 出現 Qwen resolved config 摘要
4. 本機可以連到對應 port

---

## 13. 常用覆寫參數

主要欄位：

- `CAPSWRITER_SERVER_IMAGE`
- `CAPSWRITER_MODEL_TYPE`
- `CAPSWRITER_QWEN_PRESET`
- `CAPSWRITER_INFERENCE_HARDWARE`
- `CAPSWRITER_GPU_DEVICE_COUNT`
- `CAPSWRITER_SERVER_PORT`
- `CAPSWRITER_LOG_LEVEL`
- `CAPSWRITER_NUM_THREADS`
- `CAPSWRITER_REMOVE_MODEL_ARCHIVES`

Qwen tuning override：

- `CAPSWRITER_QWEN_CHUNK_SIZE`
- `CAPSWRITER_QWEN_N_CTX`
- `CAPSWRITER_QWEN_MEMORY_NUM`
- `CAPSWRITER_QWEN_PAD_TO`
- `CAPSWRITER_QWEN_N_PREDICT`
- `CAPSWRITER_QWEN_LLAMA_N_BATCH`
- `CAPSWRITER_QWEN_LLAMA_N_UBATCH`
- `CAPSWRITER_QWEN_LLAMA_FLASH_ATTN`
- `CAPSWRITER_QWEN_LLAMA_OFFLOAD_KQV`

Advanced / internal：

- `CAPSWRITER_QWEN_USE_CUDA`
- `CAPSWRITER_QWEN_VULKAN_ENABLE`
- `CAPSWRITER_QWEN_USE_DML`
- `CAPSWRITER_QWEN_VULKAN_FORCE_FP32`

---

## 14. TODO

- 驗證更新的 NVIDIA GPU 世代是否同樣適合這條 CUDA 11.8 / ORT 1.18 image 線
- 視驗證結果決定是否維持單一 public image，或改成多個硬體相容 tag
- 未來若要研究真正的 llama layer partial offload，再另外設計 `n_gpu_layers` 等級的實驗與設定面
