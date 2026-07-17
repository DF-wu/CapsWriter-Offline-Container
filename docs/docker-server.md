# Docker 部署細節

本文件只覆蓋 **Server 端**。無 GUI CLI 與 Web Console 請見[繁體中文 CLI 指南](zh-TW/cli-client.md)與 [Web Console Client 指南](zh-TW/web-console.md)。

---

## 1. 概念

| | |
|---|---|
| Image architecture | `linux/amd64`；目前沒有 ARM64 dependency／native runtime／image gate |
| Base image | `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04@sha256:85fb7ac694079fff1061a0140fd5b5a641997880e12112d92589c3bbb1e8b7ca` |
| Python | 3.10 (venv at `/opt/venv`) |
| Python bootstrap tooling | `packaging==26.2`, `pip==26.1.2`, `setuptools==83.0.0`, `wheel==0.47.0` |
| Python runtime dependencies | [`requirements-server-docker.lock`](../requirements-server-docker.lock) pins the Docker image's transitive runtime versions and package hashes |
| 入口 | [`docker/server/entrypoint.sh`](../docker/server/entrypoint.sh) → `start_server_docker.py` |
| 模型策略 | 容器啟動時自動下載缺失模型；預設保存於 `capswriter-server-models` named volume，host bind mount 為明確 opt in |
| GPU 策略 | Base Compose 不暴露 device；加入 NVIDIA／Intel／AMD override 後，`auto` 優先 GPU；GPU bootstrap 或 probe 失敗時必須重新準備並 probe 完整 CPU fallback |
| Healthcheck | WebSocket port probe；若 `CAPSWRITER_HTTP_API_ENABLE=true`，再要求 `/ready` 回 `status=ok` |
| Runtime hardening | Image 內以 `appuser` 執行；Compose 預設 `no-new-privileges` 並 drop Linux capabilities |
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
2. 探測 container 內可見的 GPU runtime（base file 不暴露 device）並決定 Vulkan / CPU
3. 下載模型到 named volume 的 `/app/models/Qwen3-ASR/`（~1.8 GB）
4. 下載 llama.cpp Linux .so 到 `core/server/engines/*/inference/bin/`
5. 啟動識別子進程載入模型（30-60s）
6. 容器內開 WebSocket 服務於 `0.0.0.0:6016`，Docker Compose 預設只發布到 host `127.0.0.1:6016`

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
| `CAPSWRITER_GPU_DEVICE_COUNT` | `all` | 只供 `docker-compose.gpu.yml` 使用；`all` 或正整數。CPU-only 應省略 GPU override |
| `CAPSWRITER_DRI_RENDER_GID` | `109` | 只供 `docker-compose.igpu.yml` 使用；host render node 的 numeric GID |
| `CAPSWRITER_DRI_VIDEO_GID` | `44` | 只供 `docker-compose.igpu.yml` 使用；host card node 的 numeric GID |

### 3.2 網路與認證

| 變數 | 預設 | 說明 |
|---|---|---|
| `CAPSWRITER_SERVER_PUBLISH_HOST` | `127.0.0.1` | Docker Compose 發布 WebSocket port 的 host interface；LAN 共享才改 `0.0.0.0` |
| `CAPSWRITER_SERVER_PORT` | `6016` | WebSocket port |
| `CAPSWRITER_HTTP_API_ENABLE` | `false` | 啟用 OpenAI HTTP API |
| `CAPSWRITER_HTTP_API_BIND` | `0.0.0.0` | Docker 容器內監聽位址；host 是否對外由 `CAPSWRITER_HTTP_API_PUBLISH_HOST` 控制 |
| `CAPSWRITER_HTTP_API_PUBLISH_HOST` | `127.0.0.1` | Docker Compose 發布 HTTP API port 的 host interface；LAN 共享才改 `0.0.0.0` |
| `CAPSWRITER_HTTP_API_PORT` | `6017` | HTTP API port |
| `CAPSWRITER_HTTP_API_KEY` | _(空)_ | Bearer token；Docker 預設 `CAPSWRITER_HTTP_API_BIND=0.0.0.0`，啟用 HTTP API 時需設定 |
| `CAPSWRITER_HTTP_API_KEY_FILE` | _(空)_ | Bearer token 檔案；適合 Docker secrets / service manager，明確 `KEY` 優先 |
| `CAPSWRITER_HTTP_API_ALLOW_INSECURE_BIND` | `false` | 允許非 loopback bind 無 KEY 啟動；只適合受信任測試網路 |
| `CAPSWRITER_HTTP_API_MAX_UPLOAD_MB` | `100` | 單次 HTTP 音訊上傳上限 |
| `CAPSWRITER_HTTP_API_TASK_TIMEOUT` | `600` | 單次 HTTP 轉錄超時；ffmpeg 解碼與等待識別共用 |
| `CAPSWRITER_HTTP_API_MAX_CONCURRENT_REQUESTS` | `2` | HTTP 轉錄請求同時上傳/解碼/等待的上限 |
| `CAPSWRITER_HTTP_API_CORS_ORIGINS` | _(空)_ | 逗號分隔的瀏覽器 origin allowlist |
| `CAPSWRITER_HTTP_API_LOG_TRANSCRIPTS` | `false` | 是否把 prompt/context 與轉錄全文寫入 server log/console；production 建議維持 `false` |

Server、模型調校與 HTTP API env 會在啟動時做格式/範圍驗證。錯誤的 model type、port、log level、boolean、Qwen preset、數值調校、上傳大小、timeout 或 CORS origin 會讓 server 直接退出，避免 production 以意外預設值啟動。
當 HTTP API 啟用且 `CAPSWRITER_HTTP_API_BIND` 不是 loopback 時，也會要求 `CAPSWRITER_HTTP_API_KEY` 或 `CAPSWRITER_HTTP_API_KEY_FILE`，除非明確設定 `CAPSWRITER_HTTP_API_ALLOW_INSECURE_BIND=true`。
HTTP API 預設不把 prompt/context 或轉錄內容寫入 server log/console；只在受信任本機 debug 時才啟用 `CAPSWRITER_HTTP_API_LOG_TRANSCRIPTS=true`。

### 3.3 GPU/Vulkan 細部

| 變數 | 預設 | 說明 |
|---|---|---|
| `CAPSWRITER_INFERENCE_HARDWARE` | `auto` | Docker entrypoint 的硬體策略；用這個控制 GPU/CPU 偏好 |
| `CAPSWRITER_QWEN_PRESET` | `default` | Qwen backend 組合；`low_vram_gpu` 可讓 ONNX 留在 GPU、llama 改 CPU |
| `CAPSWRITER_DRI_RENDER_GID` | `109` | `docker-compose.igpu.yml` 加入的 render group；必須依 host `/dev/dri/renderD*` 修正 |
| `CAPSWRITER_DRI_VIDEO_GID` | `44` | `docker-compose.igpu.yml` 加入的 video/card group；必須依 host `/dev/dri/card*` 修正 |
| `GGML_VK_DISABLE_COOPMAT` | _(空)_ | AMD iGPU 無法載入 GGUF 時設 `1` |
| `GGML_VK_DISABLE_F16` | _(空)_ | iGPU 解碼錯誤、熔斷時設 `1` |

`CAPSWRITER_QWEN_USE_CUDA`、`CAPSWRITER_FUNASR_USE_CUDA`、`CAPSWRITER_QWEN_VULKAN_ENABLE` 與 `CAPSWRITER_FUNASR_VULKAN_ENABLE` 由 `docker/server/entrypoint.sh` 依硬體偵測結果設定；不要把它們當成 `.env` 的主要操作介面。

Linux Intel／AMD iGPU 必須明確暴露 `/dev/dri`，並把實際 host group GID 加入
container。不要假設不同 distribution 都使用 `109`／`44`：

```bash
stat -c '%n %g' /dev/dri/renderD* /dev/dri/card*
# 把對應 numeric GID 寫入 .env 的 CAPSWRITER_DRI_RENDER_GID / VIDEO_GID
docker compose -f docker-compose.yml -f docker-compose.igpu.yml up -d --force-recreate capswriter-server
```

Image 內包含 Mesa Vulkan ICD (`mesa-vulkan-drivers`)，但 host kernel driver、可用
render node 與正確 group access 仍是必要條件。NVIDIA device 則使用
`docker-compose.gpu.yml`；base file 保持 CPU-safe。

### 3.4 模型調校

| 變數 | 預設 | 說明 |
|---|---|---|
| `CAPSWRITER_QWEN_CHUNK_SIZE` | `80` | Qwen 分段長度（秒） |
| `CAPSWRITER_QWEN_N_CTX` | `2048` | Qwen llama context size |
| `CAPSWRITER_QWEN_MEMORY_NUM` | `1` | Qwen 記憶段數 |
| `CAPSWRITER_QWEN_PAD_TO` | `30` | Qwen ONNX padding 長度 |
| `CAPSWRITER_QWEN_LLAMA_N_BATCH` | _(空)_ | expert-only llama batch override |
| `CAPSWRITER_QWEN_LLAMA_N_UBATCH` | _(空)_ | expert-only llama ubatch override |
| `CAPSWRITER_QWEN_LLAMA_FLASH_ATTN` | _(空)_ | expert-only llama flash-attention override |
| `CAPSWRITER_QWEN_LLAMA_OFFLOAD_KQV` | _(空)_ | expert-only llama K/Q/V offload override |
| `CAPSWRITER_FUNASR_ENABLE_CTC` | `true` | Fun-ASR CTC hotword retrieval |
| `CAPSWRITER_FUNASR_N_PREDICT` | `512` | Fun-ASR decoder token limit |
| `CAPSWRITER_FUNASR_PAD_TO` | `30` | Fun-ASR ONNX padding 長度 |
| `CAPSWRITER_FUNASR_MAX_HOTWORDS` | `20` | 傳入 Fun-ASR decoder 的熱詞上限 |
| `CAPSWRITER_FUNASR_SIMILAR_THRESHOLD` | `0.6` | Fun-ASR 熱詞相似度門檻 |

### 3.5 日誌與資源

| 變數 | 預設 | 說明 |
|---|---|---|
| `CAPSWRITER_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `CAPSWRITER_NUM_THREADS` | `4` | CPU-bound 階段的 thread hint |
| `CAPSWRITER_SERVER_MAX_WEBSOCKET_CONNECTIONS` | `8` | 同時 admitted 的 WebSocket client 上限；必須為 `1..1024`，超額以 `1013` 拒絕 |
| `CAPSWRITER_SERVER_MAX_WEBSOCKET_TASK_SECONDS` | `3600` | 單一 WebSocket task 的累計音訊上限；必須為 `1..86400` 秒，超限只清除該 composite stream |
| `CAPSWRITER_MODEL_DOWNLOAD_TIMEOUT` | `60` | 模型與 llama.cpp archive 下載的 socket idle timeout（秒）；必須 > 0 |
| `CAPSWRITER_MODEL_BOOTSTRAP_LOCK_TIMEOUT` | `1800` | 等待共享 model volume bootstrap advisory lock 的總 deadline（秒）；必須 > 0 且 <= 86400 |
| `CAPSWRITER_BACKEND_PROBE_TIMEOUT` | `300` | 單次 configured engine construction probe 的 wall-clock deadline（秒）；必須 > 0 且 <= 1800，timeout 會 fail closed |
| `CAPSWRITER_ENGINE_FFMPEG_TIMEOUT` | `120` | direct engine file-decode `ffmpeg` timeout（秒）；HTTP API 上傳解碼仍使用 `CAPSWRITER_HTTP_API_TASK_TIMEOUT` |
| `CAPSWRITER_GPU_BOOST_TIMEOUT` | `5` | upstream GPU boost/unboost shell command timeout（秒）；只在啟用該 upstream 功能時生效 |
| `CAPSWRITER_SERVER_MODEL_LOAD_TIMEOUT` | `600` | recognizer child 載入模型的 startup deadline；超時會 bounded terminate/kill/reap 並讓 container 啟動失敗，由 supervisor 重啟 |
| `CAPSWRITER_SERVER_WORKER_STOP_TIMEOUT` | `2` | server 停止時等待 recognizer worker 優雅退出的秒數；之後會 bounded terminate/kill cleanup |
| `CAPSWRITER_SERVER_WORKER_STALL_TIMEOUT` | `900` | 單次同步 recognizer call 上限；超時會回收 child 並讓 server fail-stop，由 supervisor 重啟 |
| `CAPSWRITER_REMOVE_MODEL_ARCHIVES` | `false` | `true` = 解壓後刪除壓縮包 |

---

## 4. 模型自動下載

[`docker/server/download_models.py`](../docker/server/download_models.py) 處理：

1. 根據 `CAPSWRITER_MODEL_TYPE` 找出對應 ASSETS
2. 重新 hash required model artifacts，並要求 schema 2 `.capswriter-model-ready.json` 的 archive identity／SHA-256 與 size／SHA-256 manifest 完全相符
3. Readiness 不成立時，以 bounded streaming download 寫入 `models/.downloads/*.part`，完成後才 atomic replace 到 archive path，校驗 SHA-256，安全解壓到 staging，再 atomic promote 到 target directory（拒絕 traversal / link / special-file archive members）
4. 對 GGUF backend（qwen_asr / fun_asr_nano），下載對應 llama.cpp Linux `.so` 到三個 inference/bin 目錄，並要求 schema 2 `.capswriter-llama-ready.json` 綁定 selected backend、archive SHA-256 與每個 `.so` 的 size／SHA-256；runtime-linked `libggml.so.0` / `libggml-base.so.0` 等 SONAME 也必須存在：
   - `core/server/engines/qwen_asr_gguf/inference/bin/`
   - `core/server/engines/fun_asr_gguf/inference/bin/`
   - `core/server/engines/force_aligner_gguf/inference/bin/`
5. 對 `CAPSWRITER_LLAMA_BACKEND=vulkan`，會拉 `libggml-vulkan.so`；CPU 則只拉 cpu 版

Schema 2 marker 是可重建的 readiness metadata，不會取代實際檔案驗證。每次 warm
start 都會重新計算 manifest 並做 exact comparison；即使檔案大小不變，只要內容
被修改、marker 對錯 backend/archive、或 runtime marker 缺失，就不會走 read-only
fast path，而會在 bootstrap lock 內 repair。舊 model directory 只有在 source 內已
pin 的 required size 與 SHA-256 全部吻合時才可相容；llama runtime 不接受無 marker
的 legacy directory。

下載使用 `CAPSWRITER_MODEL_DOWNLOAD_TIMEOUT` 作為每次 socket connect/read 的 idle timeout；這不是整體下載時間上限，慢速但持續有資料的模型下載不會因為總耗時超過 60 秒而被中斷。失敗或 timeout 時只會留下已存在且完整的舊 archive，不會把半截檔案放在正式 cache path。解壓會先檢查 archive member path 與類型，避免惡意或損壞 archive 寫出目標目錄。

多個 container 共用同一 model volume 時，所有 recovery、download、promotion 與
archive-cache cleanup 都由 `/app/models/.capswriter-bootstrap.lock` 的 non-blocking
`flock(2)` 序列化，並共用一個 `CAPSWRITER_MODEL_BOOTSTRAP_LOCK_TIMEOUT` absolute
deadline。取得 lock 後會重新檢查 readiness，避免第二個 container 重複下載。Lock
file 必須是目前 non-root user 擁有、不可由 group／other 寫入且只有一個 hard
link 的 regular file；symlink 或不安全的既有 lock 會讓 bootstrap fail closed。

這是 POSIX advisory lock：共享 storage／volume driver 必須在所有 client 間提供
一致的 `flock(2)` semantics，而且所有會修改該目錄的 writer 都必須遵守同一個
lock。沒有證明 lock coherence 前，不要把共同 bootstrap 目錄放在 NFS／SMB 等
network filesystem。模型、ready marker 與 llama runtime 都已完整，且未要求刪除
archive 時，warm fast path 會在建立 lock 前返回；只有這個完全 ready 的情況可以
讓 model root 維持 read-only。任何缺少 asset、recovery 或 cache cleanup 都需要
可寫 storage。

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
    → 關閉 llama Vulkan、Qwen/FunASR CUDA 與 Vulkan；整條 pipeline CPU
elif gpu_visible:
    → llama_backend = vulkan
    → 若 nvidia_visible: QWEN_USE_CUDA=true, FUNASR_USE_CUDA=true
    → QWEN_VULKAN_ENABLE=true, FUNASR_VULKAN_ENABLE=true

    if qwen_preset = low_vram_gpu:
        → QWEN_VULKAN_ENABLE=false (llama CPU)
        → QWEN_USE_CUDA=true (ONNX 仍 GPU)
else:
    → 關閉所有 CUDA/Vulkan flag；整條 pipeline CPU
```

只要上述設定留下任何 CUDA／Vulkan path，entrypoint 會先 bootstrap 對應 runtime，
並用 `EngineFactory` 建構一次完整 configured engine。GPU runtime bootstrap 或 probe
任一失敗（包括 archive/runtime、library、model load、driver、GPU OOM 或 probe
timeout），都必須把 llama backend 與 Qwen／FunASR 的 CUDA、Vulkan flag 全部切到
CPU，重新執行 `download_models.py` 準備 CPU runtime，再建構一次 CPU engine。第二次
probe 也失敗就拒絕啟動；因此 log 中的「falling back」不是未驗證的設定變更。

每次 engine construction 都在受監督的 child 內執行，wall-clock deadline 由
`CAPSWRITER_BACKEND_PROBE_TIMEOUT` 控制（預設 `300` 秒，合法值 > 0 且 <= `1800`）。
這套 fallback 只能處理 container 已成功建立後的 bootstrap／startup probe failure；
Docker 在 entrypoint 前就無法完成 NVIDIA device reservation 時，仍需移除 GPU
override。

---

## 6. 切換到 Fun-ASR-Nano

低延遲 / 互動場景：

```bash
docker compose -f docker-compose.yml -f docker-compose.fun-asr.yml up -d
```

[`docker-compose.fun-asr.yml`](../docker-compose.fun-asr.yml) override 同時調 `CAPSWRITER_MODEL_TYPE=fun_asr_nano` 及對應參數。

---

## 7. Volumes & 持久化

Base Compose 預設：

```yaml
volumes:
  - capswriter-server-models:/app/models  # 模型 named volume
  - type: bind                            # 服務端熱詞（唯讀）
    source: ./hot-server.txt
    target: /app/hot-server.txt
    read_only: true
    bind:
      create_host_path: false
  - capswriter-server-logs:/app/logs       # 日誌 named volume
```

Named model volume 由 Docker 建立並保留，刪除／recreate container 不會刪除它；
只有明確執行 `docker compose down -v` 或 `docker volume rm` 才會移除。這個預設也
避免 host directory 的 UID／GID 與寫入權限問題。

若 operator 確實需要直接管理 host files，加入明確 bind override：

```bash
image="$(docker compose config --images | sed -n '1p')"
uid="$(docker run --rm --entrypoint id "$image" -u appuser)"
gid="$(docker run --rm --entrypoint id "$image" -g appuser)"
mkdir -p models
sudo chown -R "$uid:$gid" models
docker compose -f docker-compose.yml -f docker-compose.models-bind.yml up -d capswriter-server
```

`docker-compose.models-bind.yml` 會以 `./models:/app/models` 取代同一 target 的
named volume。Bootstrap 期間整個 directory、lock file、`.downloads` 與 staging
paths 都必須能由 image 內的 `appuser` 寫入；不要只讓最終 model file 可讀。
共享 bind mount 也必須符合上一節的 advisory-lock 條件。

`./hot-server.txt` 應從 `hot-server.example.txt` 複製。日誌用 named volume；一般
查看方式是 `docker compose logs -f capswriter-server`。

---

## 8. 構建自己的 image（可選）

如果想 fork 開發，本機 build：

```bash
docker build -t capswriter-server:local -f docker/server/Dockerfile .
```

`.env` 設 `CAPSWRITER_SERVER_IMAGE=capswriter-server:local` 切換。

CI 自動 build 走 [`.github/workflows/publish-server-image.yml`](../.github/workflows/publish-server-image.yml)：push 到 master 即觸發。該 workflow 會先跑 `python scripts/verify_all.py --skip-web`，通過後才 build/push GHCR server image，並附帶 provenance 與 SBOM attestations。Dockerfile 的 CUDA base image、Python bootstrap tooling 與 Docker runtime dependency version/hash lock 都固定，避免 build 時拉到浮動最新版本或未預期 package artifact。

Server image build context 由 [`.dockerignore`](../.dockerignore) 排除本機 `models/`、Web/CLI 產物、`.env*`、secret-like key/cert files 與下載中的 archive，避免把本機模型、token 或驗證輸出打進 image。

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
| 模型下載卡住或反覆 timeout | 檢查容器到 GitHub release 的網路；慢速網路可調高 `CAPSWRITER_MODEL_DOWNLOAD_TIMEOUT`。中斷下載只會留下 `.part` 暫存檔，重啟會重新下載 |
| 等待 bootstrap lock timeout／拒絕 lock file | 找出仍在寫同一 model volume 的 instance；確認 shared filesystem 支援 coherent `flock(2)`，以及 lock 是 `appuser` 擁有、mode `0600`、單一 hard link 的 regular file。不要刪除 live process 正持有的 lock |
| `configured backend probe timed out` | Probe 的預設 wall-clock deadline 是 300 秒；先查 model/runtime/driver log。只有確認硬體初始化確實較慢時才提高 `CAPSWRITER_BACKEND_PROBE_TIMEOUT`，且不可超過 1800；CPU fallback probe 仍失敗時 container 會拒絕啟動 |
| Warm start 重新下載／repair model 或 llama runtime | Schema 2 ready marker 與重新計算的 size／SHA-256 manifest 不符；檢查 volume 是否被手動改動、截斷或混入另一 backend 的 runtime，不要手動偽造 marker |
| `ONNXRuntimeError: NO_SUCHFILE` | 模型檔不在預期路徑。以 `docker compose exec capswriter-server ls -l /app/models/Qwen3-ASR/Qwen3-ASR-1.7B/` 檢查檔名是否對齊 [`config_server.py:ModelPaths`](../config_server.py) |
| `libggml-base.so.0: cannot open shared object file` | llama.cpp runtime libraries 不完整；重新跑 `python docker/server/download_models.py` 或重建/重啟容器讓 entrypoint 補齊 versioned `.so` |
| iGPU 沒有被使用 | 加入 `docker-compose.igpu.yml`，再依 host `/dev/dri/*` numeric GID 修正兩個 `CAPSWRITER_DRI_*_GID` 值 |
| AMD iGPU 模型載入失敗 | 先驗證 device/group access；再於 `.env` 設 `GGML_VK_DISABLE_COOPMAT=1` |
| Intel iGPU 解碼結果亂 | `.env` 設 `GGML_VK_DISABLE_F16=1` |
| `Container is not running` 立刻退出 | `docker logs <container>` 看 traceback；通常是 missing dep 或 model file 缺失 |
| WebSocket port 6016 被占用 | `.env` 改 `CAPSWRITER_SERVER_PORT=6116` 並調 ports 映射 |

更多疑難排解見 [docs/HTTP_API.md §8](HTTP_API.md#8-故障排除)（HTTP API 相關）和 [docs/upstream-sync-guide.md](upstream-sync-guide.md)（merge 上游後問題）。
