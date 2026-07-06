# Docker 部署細節

本文件只覆蓋 **Server 端**。No-GUI CLI 與 Web Console 請見 [cli-client.md](cli-client.md) 與 [web-console.md](web-console.md)。

---

## 1. 概念

| | |
|---|---|
| Base image | `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04` |
| Python | 3.10 (venv at `/opt/venv`) |
| 入口 | [`docker/server/entrypoint.sh`](../docker/server/entrypoint.sh) → `start_server_docker.py` |
| 模型策略 | 容器啟動時自動下載缺失模型；走 host bind-mount `./models:/app/models` |
| GPU 策略 | `CAPSWRITER_INFERENCE_HARDWARE=auto`：先試 GPU，失敗回退 CPU |
| Healthcheck | WebSocket port probe；若 `CAPSWRITER_HTTP_API_ENABLE=true`，再要求 `/ready` 回 `status=ok` |
| 公開 image | `ghcr.io/df-wu/capswriter-offline-server:latest`（[`publish-server-image.yml`](../.github/workflows/publish-server-image.yml) 自動發布） |

---

## 2. 最小啟動流程

```bash
cp .env.example .env
cp hot-server.example.txt hot-server.txt
docker compose up -d capswriter-server
docker compose logs -f capswriter-server
```

第一次啟動會：
1. 拉 `ghcr.io/df-wu/capswriter-offline-server:latest`（~5 GB）
2. 探測 GPU runtime（`/dev/nvidiactl` 或 `/dev/dri`）並決定 Vulkan / CPU
3. 下載模型到 `./models/Qwen3-ASR/`（~1.8 GB）
4. 下載 llama.cpp Linux .so 到 `core/server/engines/*/inference/bin/`
5. 啟動識別子進程載入模型（30-60s）
6. 開 WebSocket 服務於 `0.0.0.0:6016`

`docker compose ps` 顯示 `Up (healthy)` 即成功。

---

## 3. 環境變數完整表

完整列表見 [`.env.example`](../.env.example)。下表為**最常用**的：

### 3.1 部署選擇

| 變數 | 預設 | 說明 |
|---|---|---|
| `CAPSWRITER_SERVER_IMAGE` | `ghcr.io/df-wu/capswriter-offline-server:latest` | image tag |
| `CAPSWRITER_MODEL_TYPE` | `qwen_asr` | `qwen_asr` / `fun_asr_nano` |
| `CAPSWRITER_QWEN_PRESET` | `default` | `default`（CUDA ONNX + Vulkan llama）/ `low_vram_gpu`（CUDA ONNX + CPU llama）/ `cpu_only` |
| `CAPSWRITER_INFERENCE_HARDWARE` | `auto` | `auto` / `gpu` / `cpu` |
| `CAPSWRITER_GPU_DEVICE_COUNT` | `all` | NVIDIA GPU 數量；`0` = CPU-only |

### 3.2 網路與認證

| 變數 | 預設 | 說明 |
|---|---|---|
| `CAPSWRITER_SERVER_PORT` | `6016` | WebSocket port |
| `CAPSWRITER_HTTP_API_ENABLE` | `false` | 啟用 OpenAI HTTP API |
| `CAPSWRITER_HTTP_API_BIND` | `127.0.0.1` | 對外請改 `0.0.0.0` |
| `CAPSWRITER_HTTP_API_PORT` | `6017` | HTTP API port |
| `CAPSWRITER_HTTP_API_KEY` | _(空)_ | Bearer token；對外時必填 |
| `CAPSWRITER_HTTP_API_ALLOW_INSECURE_BIND` | `false` | 允許非 loopback bind 無 KEY 啟動；只適合受信任測試網路 |
| `CAPSWRITER_HTTP_API_MAX_UPLOAD_MB` | `100` | 單次 HTTP 音訊上傳上限 |
| `CAPSWRITER_HTTP_API_TASK_TIMEOUT` | `600` | 單次 HTTP 轉錄超時；ffmpeg 解碼與等待識別共用 |
| `CAPSWRITER_HTTP_API_MAX_CONCURRENT_REQUESTS` | `2` | HTTP 轉錄請求同時上傳/解碼/等待的上限 |
| `CAPSWRITER_HTTP_API_CORS_ORIGINS` | _(空)_ | 逗號分隔的瀏覽器 origin allowlist |
| `CAPSWRITER_HTTP_API_LOG_TRANSCRIPTS` | `false` | 是否把 prompt/context 與轉錄全文寫入 server log/console；production 建議維持 `false` |

HTTP API 相關 env 會在啟動時做範圍驗證。錯誤的 port、上傳大小、timeout 或 CORS origin 會讓 server 直接退出，避免 production 以意外預設值啟動。
當 HTTP API 啟用且 `CAPSWRITER_HTTP_API_BIND` 不是 loopback 時，也會要求 `CAPSWRITER_HTTP_API_KEY`，除非明確設定 `CAPSWRITER_HTTP_API_ALLOW_INSECURE_BIND=true`。
HTTP API 預設不把 prompt/context 或轉錄內容寫入 server log/console；只在受信任本機 debug 時才啟用 `CAPSWRITER_HTTP_API_LOG_TRANSCRIPTS=true`。

### 3.3 GPU/Vulkan 細部

| 變數 | 預設 | 說明 |
|---|---|---|
| `CAPSWRITER_QWEN_VULKAN_ENABLE` | `true` | Qwen llama 是否走 Vulkan |
| `CAPSWRITER_FUNASR_VULKAN_ENABLE` | `true` | Fun-ASR llama 是否走 Vulkan |
| `GGML_VK_DISABLE_COOPMAT` | _(空)_ | AMD iGPU 無法載入 GGUF 時設 `1` |
| `GGML_VK_DISABLE_F16` | _(空)_ | iGPU 解碼錯誤、熔斷時設 `1` |

### 3.4 日誌與資源

| 變數 | 預設 | 說明 |
|---|---|---|
| `CAPSWRITER_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `CAPSWRITER_NUM_THREADS` | `4` | CPU-bound 階段的 thread hint |
| `CAPSWRITER_REMOVE_MODEL_ARCHIVES` | `false` | `true` = 解壓後刪除壓縮包 |

---

## 4. 模型自動下載

[`docker/server/download_models.py`](../docker/server/download_models.py) 處理：

1. 根據 `CAPSWRITER_MODEL_TYPE` 找出對應 ASSETS
2. 檢查 `is_ready()`：required_files 全在 → 跳過下載
3. 否則下載 zip 到 `models/.downloads/`，校驗 sha256，解壓到 target_dir
4. 對 GGUF backend（qwen_asr / fun_asr_nano），下載對應 llama.cpp Linux `.so` 到三個 inference/bin 目錄：
   - `core/server/engines/qwen_asr_gguf/inference/bin/`
   - `core/server/engines/fun_asr_gguf/inference/bin/`
   - `core/server/engines/force_aligner_gguf/inference/bin/`
5. 對 `CAPSWRITER_LLAMA_BACKEND=vulkan`，會拉 `libggml-vulkan.so`；CPU 則只拉 cpu 版

容器自動下載與 production gate 目前支援的模型（對應 upstream v2.6 的 GGUF 路徑）：

| 模型 | Release zip | sha256 對齊 |
|---|---|---|
| qwen_asr | `Qwen3-ASR-1.7B-q5_k.zip` | ✅ |
| fun_asr_nano | `Fun-ASR-Nano-GGUF.zip` | ✅ |

上游裸機路徑仍保留 `sensevoice`、`paraformer` 等模型設定；本 fork 的 Docker 自動下載、GPU/CPU fallback 與驗證 gate 尚未覆蓋這些模型。若 `CAPSWRITER_MODEL_TYPE` 設為未支援值，container 會在啟動前退出並提示改用上游裸機部署或先補齊 Docker asset 支援。

---

## 5. GPU 偵測邏輯

[`docker/server/entrypoint.sh`](../docker/server/entrypoint.sh) `configure_backend()`：

```
inference_hardware ≔ CAPSWRITER_INFERENCE_HARDWARE (or CAPSWRITER_GPU_MODE)
nvidia_visible ≔ /dev/nvidiactl 存在?
gpu_visible    ≔ /dev/nvidiactl 存在? 或 /dev/dri 存在?

if inference_hardware = cpu:
    → 整條 pipeline CPU
elif gpu_visible:
    → llama_backend = vulkan
    → 若 nvidia_visible: QWEN_USE_CUDA=true, FUNASR_USE_CUDA=true
    → QWEN_VULKAN_ENABLE=true, FUNASR_VULKAN_ENABLE=true

    if qwen_preset = low_vram_gpu:
        → QWEN_VULKAN_ENABLE=false (llama CPU)
        → QWEN_USE_CUDA=true (ONNX 仍 GPU)
else:
    → CPU fallback
```

若 ONNX 啟動失敗（probe 失敗），自動重置 CUDA 為 false 並重跑 download_models。

---

## 6. 切換到 Fun-ASR-Nano

低延遲 / 互動場景：

```bash
docker compose -f docker-compose.yml -f docker-compose.fun-asr.yml up -d
```

[`docker-compose.fun-asr.yml`](../docker-compose.fun-asr.yml) override 同時調 `CAPSWRITER_MODEL_TYPE=fun_asr_nano` 及對應參數。

---

## 7. Volumes & 持久化

```yaml
volumes:
  - ./models:/app/models               # 模型資料夾（host 共享，自動下載落地於此）
  - ./hot-server.txt:/app/hot-server.txt   # 服務端熱詞
  - capswriter-server-logs:/app/logs   # 日誌 (named volume)
```

說明：
- 模型不打進 image，因為太大且想跨環境重用
- `./hot-server.txt` 從 `hot-server.example.txt` 複製一份
- 日誌用 named volume；想看：`docker compose logs -f capswriter-server`

---

## 8. 構建自己的 image（可選）

如果想 fork 開發，本機 build：

```bash
docker build -t capswriter-server:local -f docker/server/Dockerfile .
```

`.env` 設 `CAPSWRITER_SERVER_IMAGE=capswriter-server:local` 切換。

CI 自動 build 走 [`.github/workflows/publish-server-image.yml`](../.github/workflows/publish-server-image.yml)：push 到 master 即觸發。

---

## 9. 健康檢查與 ops

```bash
# 是否健康
docker compose ps

# HTTP API readiness (when enabled)
curl http://localhost:6017/ready

# 跑了多久
docker ps --filter "name=capswriter" --format '{{.Names}} {{.Status}}'

# 看 log
docker compose logs -f capswriter-server

# 進容器
docker compose exec capswriter-server bash

# 重新建容器（不重 pull image）
docker compose up -d --force-recreate capswriter-server

# 拉新 image 並 recreate
docker compose pull && docker compose up -d --force-recreate capswriter-server
```

Container healthcheck 行為：

1. 一律對 `CAPSWRITER_SERVER_PORT` 發送合法 HTTP/1.1 request，確認 WebSocket server 已接受連線且不製造 traceback noise。
2. 若 `CAPSWRITER_HTTP_API_ENABLE=true`，再呼叫容器內 `CAPSWRITER_HTTP_API_PORT` 的 `/ready`。
3. `/ready` 必須回 HTTP `200` 且 JSON `status="ok"`；`503 degraded` 會讓容器保持 unhealthy，避免 HTTP sidecar 未綁定 router 或找不到 `ffmpeg` 時被當作可接流量。

---

## 10. 故障排除

| Symptom | 解法 |
|---|---|
| 容器啟動 20 分鐘後 healthcheck 還沒過 | 模型尚未載入或 HTTP `/ready` 還是 degraded；`docker logs` 看進度，`curl :6017/ready` 看 checks。冷啟動可能 30+ 分鐘 |
| `ONNXRuntimeError: NO_SUCHFILE` | 模型檔不在預期路徑。檢查 `./models/Qwen3-ASR/Qwen3-ASR-1.7B/` 內檔名是否對齊 [`config_server.py:ModelPaths`](../config_server.py) |
| AMD iGPU 模型載入失敗 | `.env` 設 `GGML_VK_DISABLE_COOPMAT=1` |
| Intel iGPU 解碼結果亂 | `.env` 設 `GGML_VK_DISABLE_F16=1` |
| `Container is not running` 立刻退出 | `docker logs <container>` 看 traceback；通常是 missing dep 或 model file 缺失 |
| WebSocket port 6016 被占用 | `.env` 改 `CAPSWRITER_SERVER_PORT=6116` 並調 ports 映射 |

更多疑難排解見 [docs/HTTP_API.md §8](HTTP_API.md#8-故障排除)（HTTP API 相關）和 [docs/upstream-sync-guide.md](upstream-sync-guide.md)（merge 上游後問題）。
