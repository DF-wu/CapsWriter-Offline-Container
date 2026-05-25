# Fork 現況（State of Fork）

> **更新時間**：2026-05-25（reset 後）
> **基底**：origin/master @ `0362630` (upstream v2.5)
> **工作分支**：`feat/reset-to-upstream`

---

## 1. 一句話定位

本 fork 是 [HaujetZhao/CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline) 的 **Linux container 化貼皮**。fork 的全部價值集中在：(a) Linux/Docker 容器部署、(b) env 驅動配置、(c) OpenAI Whisper 相容 HTTP API。**ASR 識別引擎完全來自 upstream**。

---

## 2. 設計現況

| | |
|---|---|
| 上游檔案修改數 | **1**（僅 `.gitignore` 加 `.fork-archive-*/` 一行） |
| Fork 新增檔案 | `fork_server/`、`docker/`、`docker-compose*.yml`、`.env.example`、`requirements-server-docker.txt`、`start_server_docker.py`、`.github/workflows/publish-server-image.yml` |
| Hook 策略 | 子類化 + 1 處 monkey-patch（`server_manager.ws_send`） |
| 唯一漂移點 | [`fork_server/http_api/ws_send_with_http.py`](../fork_server/http_api/ws_send_with_http.py) 內嵌了上游 ws_send 邏輯複本 |
| Safety tag | `fork-pre-reset-20260525-1411`（pre-reset 快照，已 push fork remote） |

詳細設計：[architecture.md](architecture.md)。對齊上游：[upstream-sync-guide.md](upstream-sync-guide.md)。

---

## 3. 驗證紀錄（2026-05-25）

於 `/tmp/cw-reset-test/` 用獨立 compose project（name=`cwreset`、port 16016/16017）跑本地 build image + bind-mount 模型，**全程不影響** production container `capswriter-offline-capswriter-server-1`（port 6016，Up 3 weeks healthy 整個過程未中斷）。

| 模式 | 結果 |
|---|---|
| qwen_asr container build | ✅ image ~5GB |
| qwen_asr container start | ✅ healthcheck pass 47s |
| qwen_asr WebSocket port listen | ✅ 6016/16016 各自 listen |
| qwen_asr HTTP `/health` | ✅ 200 |
| qwen_asr HTTP `/v1/models` | ✅ 200 |
| qwen_asr HTTP `/v1/audio/transcriptions` × 5 format | ✅ json/text/srt/vtt/verbose_json 全 200 |
| 順序重複請求 | ✅ 3 次連續 200，輸出一致，延遲穩定 ~4s |
| 並發請求 | ✅ task_router 正確處理 |
| 模型載入時 GPU 偵測 | ✅ 自動 fallback CPU（host 無 GPU 暴露給 docker） |
| Production unaffected | ✅ port 6016 容器 throughout healthy |

**未跑的測試**：
- Fun-ASR-Nano live 端到端：host 模型是舊 int4 格式，新 v2.5 expect fp16。架構與 qwen 共享，HTTP 路徑等效驗證
- 多 GPU 環境驗證：host 此次測試 GPU 已被 production 容器占用，CPU fallback 路徑已通

---

## 4. 容器內目前可用功能

### 4.1 啟動

```bash
# 預設：qwen_asr + WebSocket
docker compose up -d capswriter-server

# 切 fun_asr_nano（低延遲）
docker compose -f docker-compose.yml -f docker-compose.fun-asr.yml up -d
```

### 4.2 OpenAI HTTP API

預設關閉。設 `CAPSWRITER_HTTP_API_ENABLE=true` + 打開 port 映射即可。詳見 [HTTP_API.md](HTTP_API.md)。

支援端點：
| 端點 | 用途 |
|---|---|
| `POST /v1/audio/transcriptions` | OpenAI Whisper API；5 format |
| `GET  /v1/models` | OpenAI SDK introspection |
| `GET  /health` | Liveness |
| `POST /v1/audio/translations` | 明確 501（本地模型不翻譯） |

### 4.3 env 驅動的關鍵變數

| 變數 | 預設 | |
|---|---|---|
| `CAPSWRITER_MODEL_TYPE` | `qwen_asr` | `fun_asr_nano` / `sensevoice` / `paraformer` |
| `CAPSWRITER_QWEN_PRESET` | `default` | `low_vram_gpu` / `cpu_only` |
| `CAPSWRITER_INFERENCE_HARDWARE` | `auto` | `gpu` / `cpu` |
| `CAPSWRITER_HTTP_API_ENABLE` | `false` | `true` 開 HTTP |
| 其他 | 見 [`.env.example`](../.env.example) | 含 CUDA / Vulkan / iGPU 補丁 |

---

## 5. 已知未決事項

### 5.1 Fun-ASR-Nano live 測試延後
Host 上模型仍是舊 int4 ONNX，需新版 fp16 release zip 重下載。**架構驗證已透過 qwen_asr 共用通路完成**，Fun-ASR 只是換引擎參數，理論上同 image 直接可跑。

### 5.2 GHCR `:latest` 尚未含本次 reset 的程式碼
GHCR 上的 `:latest` 是 reset 前的 fork。push 後 [`publish-server-image.yml`](../.github/workflows/publish-server-image.yml) 會自動 build；CI 綠燈後使用者可 `docker compose pull && up -d --force-recreate` 升級。

### 5.3 Production port 6016 容器
仍跑舊 image。**reset 不影響它**。要 migrate 自行：

```bash
docker compose pull capswriter-server
docker compose up -d --force-recreate capswriter-server
```

### 5.4 Force aligner 模型未自動下載
上游 v2.5 引入 `force_aligner_gguf` 引擎用於字級時間戳。本次 fork 的 [`docker/server/download_models.py`](../docker/server/download_models.py) 只下載 qwen / fun-asr 主模型，未含 aligner（`models/Qwen3-ForcedAligner/Qwen3-ForcedAligner-0.6B/`）。

影響：HTTP API 的 `verbose_json` 仍能提供字級 timestamp（recognizer 內建估算），只是精度可能略低於專屬 aligner。`aligner_idle_timeout=10` 在 [`config_server.py`](../config_server.py) 已預設啟用 idle 卸載，缺 aligner 時 `EngineFactory.create_align_engine()` 自動 fallback 到 no-op，無錯誤。

---

## 6. 檔案地圖

### Fork 加值（上游無此目錄/檔）

- [`fork_server/`](../fork_server/) — Sidecar 套件
  - `bootstrap.py` — `ForkedCapsWriterServer.start()`
  - `env_config.py` — env → ServerConfig
  - `http_api/` — OpenAI Whisper API
- [`docker/`](../docker/) — 容器構建
- [`docker-compose.yml`](../docker-compose.yml)、[`docker-compose.fun-asr.yml`](../docker-compose.fun-asr.yml)
- [`.env.example`](../.env.example)、[`requirements-server-docker.txt`](../requirements-server-docker.txt)
- [`start_server_docker.py`](../start_server_docker.py) — Fork 入口
- [`.github/workflows/publish-server-image.yml`](../.github/workflows/publish-server-image.yml)

### 重寫但與上游同名（接受偶發 merge 衝突）

- [`readme.md`](../readme.md) — fork 視角

### 修改的上游檔（**只有 1 個**）

- [`.gitignore`](../.gitignore) — 加 `.fork-archive-*/` 排除

### 文件

- [`readme.md`](../readme.md) — fork 首頁
- [`docs/architecture.md`](architecture.md) — Sidecar 設計
- [`docs/upstream-sync-guide.md`](upstream-sync-guide.md) — Merge SOP
- [`docs/HTTP_API.md`](HTTP_API.md) — OpenAI API 規格
- [`docs/docker-server.md`](docker-server.md) — Container 部署細節
- [`docs/state-of-fork.md`](state-of-fork.md) — 本檔

---

## 7. 對話歷史的廢棄部分

reset 之前的 session 紀錄出現過、但 reset 後**不再代表方向**：

- ~~`util/server/*` 模組路徑~~ — reset 後全部移到 `fork_server/http_api/`
- ~~Cosmic 全域單例~~ — 改用 `state` 注入（上游 v2.5 設計）
- ~~`core_server.py` cotask 啟動~~ — 改在 `ForkedCapsWriterServer.start()` 用 `asyncio.gather`
- ~~`server_ws_send.py` 修改~~ — 改 monkey-patch `server_manager.ws_send`，上游檔不動
- ~~`balanced` / `quality` preset~~ — 已正式淘汰，剩 `default` / `low_vram_gpu` / `cpu_only`

---

## 8. 回滾保險

```bash
# 列出 safety tags
git tag | grep fork-pre-

# 緊急回滾到 reset 前狀態
git checkout fork-pre-reset-20260525-1411
# 或
git reset --hard fork-pre-reset-20260525-1411
```

Pre-reset 狀態詳細快照保存在工作樹的 `.fork-archive-2026-05-25/`（27 個關鍵檔），未追蹤但保留在 disk 上作為次要保險。
