# CapsWriter-Offline Linux Container Fork

> **離線中文/英文語音識別，跑在 Linux 容器，講 OpenAI Whisper 的 API。**
>
> *English: see [README.en.md](README.en.md).*

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker&logoColor=white)](docker-compose.yml)
[![OpenAI Compatible](https://img.shields.io/badge/OpenAI%20Whisper-compatible-10A37F?logo=openai&logoColor=white)](docs/HTTP_API.md)
[![GPU](https://img.shields.io/badge/GPU-CUDA%20%2B%20Vulkan%20%2B%20CPU--fallback-76B900?logo=nvidia&logoColor=white)](docs/docker-server.md)
[![Upstream](https://img.shields.io/badge/upstream-HaujetZhao%2FCapsWriter--Offline-181717?logo=github)](https://github.com/HaujetZhao/CapsWriter-Offline)

本專案是 [HaujetZhao/CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline) 的 **Linux Container 化貼皮**。上游的識別引擎完全保留，本 fork 只專注一件事：**讓 server 端能用一行 `docker compose up` 跑起來，並提供 OpenAI Whisper 相容的 HTTP 端點**。

> 如果你要在 Windows 桌面用語音輸入，用上游。
> 如果你要把 ASR 當服務跑在 Linux/容器/GPU 機器上、給其他應用用 OpenAI SDK 對接，用本 fork。

---

## 為什麼選這個 Fork

| | |
|---|---|
| 🐳 **一行部署** | `docker compose up -d`，含模型自動下載 |
| 🤖 **OpenAI SDK 直接相容** | `POST /v1/audio/transcriptions`，5 種 response_format |
| ⚡ **GPU 加速** | NVIDIA CUDA + Vulkan + iGPU 補丁；自動偵測 GPU 並 graceful fallback 到 CPU |
| 🎯 **支援兩種模型** | `qwen_asr` (高精度，長段) + `fun_asr_nano` (低延遲，互動) |
| 🔄 **易跟進上游** | 主要加值都在新增目錄；目前只有 5 個 upstream-tracked 檔案有刻意差異，並已在同步 SOP 中列明。詳見 [docs/architecture.md](docs/architecture.md) |
| 🔒 **完全離線** | 模型本地推論，不向任何雲端傳音訊 |

---

## Quick Start

### 0. 先決條件

- Linux + Docker Engine + Docker Compose plugin
- (可選) NVIDIA driver + NVIDIA Container Toolkit (GPU 加速)
- 約 6 GB 磁碟空間（模型 + image）

### 1. 取得設定

```bash
git clone https://github.com/DF-wu/CapsWriter-Offline-Container.git
cd CapsWriter-Offline-Container
cp .env.example .env
cp hot-server.example.txt hot-server.txt
```

### 2. 啟動

```bash
docker compose up -d capswriter-server
docker compose logs -f capswriter-server
```

第一次啟動會自動下載 Qwen3-ASR 模型 + llama.cpp Vulkan 二進位（總計 ~5 GB），需要幾分鐘。

### 3. 驗證

```bash
docker compose ps                          # capswriter-server 應為 Up (healthy)
curl http://localhost:6016                 # WebSocket 在這 port (上游客戶端用)
```

### 4. (可選) 開 OpenAI HTTP API

編輯 `.env`：

```bash
CAPSWRITER_HTTP_API_ENABLE=true
CAPSWRITER_HTTP_API_BIND=0.0.0.0
CAPSWRITER_HTTP_API_PORT=6017
CAPSWRITER_HTTP_API_KEY=sk-your-token       # 對外時務必設定
CAPSWRITER_HTTP_API_CORS_ORIGINS=http://127.0.0.1:5173  # 使用 Web Console 時設定
```

打開 [`docker-compose.yml`](docker-compose.yml) 的 port mapping（取消 `# - "6017:6017"` 那行的註解），然後：

```bash
docker compose up -d --force-recreate capswriter-server
curl http://localhost:6017/health
curl http://localhost:6017/ready
```

任何 OpenAI SDK 設 `base_url=http://localhost:6017/v1` 即可使用：

```python
from openai import OpenAI
client = OpenAI(api_key="sk-your-token", base_url="http://localhost:6017/v1")
with open("audio.mp3", "rb") as f:
    t = client.audio.transcriptions.create(model="whisper-1", file=f)
print(t.text)
```

詳細 API 規格見 [docs/HTTP_API.md](docs/HTTP_API.md)。
瀏覽器工作台見 [docs/web-console.md](docs/web-console.md)。
Web Console 真實瀏覽器 smoke test：在 `client/web` 執行 `npm run browser-smoke`。
無 GUI CLI 支援 `health` / `ready` / `models` / `transcribe` / `speak`，可封裝為單檔 zipapp：`python client/cli/scripts/build_zipapp.py`，詳見 [docs/cli-client.md](docs/cli-client.md)。

Web Console 也可用靜態容器部署：

```bash
docker compose -f docker-compose.web.yml up -d --build capswriter-web
```

發布 image：`ghcr.io/df-wu/capswriter-offline-web:latest`。

---

## 模型矩陣

| 模型 | 精度 | 延遲 | 場景 | env |
|---|---|---|---|---|
| **qwen_asr** | 🟢 高 | 🟡 中 | 對話精準、長段轉錄、字幕 | `CAPSWRITER_MODEL_TYPE=qwen_asr` (預設) |
| **fun_asr_nano** | 🟡 中 | 🟢 低 | 即時互動、語音助手、whisper SDK | `CAPSWRITER_MODEL_TYPE=fun_asr_nano` |

切換 `fun_asr_nano` 用 override（低延遲調優設定都在裡面）:

```bash
docker compose -f docker-compose.yml -f docker-compose.fun-asr.yml up -d
```

---

## GPU 加速

[`docker-compose.yml`](docker-compose.yml) 預設請求 `count: all` 的 NVIDIA GPU。[`docker/server/entrypoint.sh`](docker/server/entrypoint.sh) 會自動偵測：

| 環境 | 偵測結果 | 行為 |
|---|---|---|
| NVIDIA GPU + 驅動正常 | `nvidiactl` 存在 | ONNX 走 CUDA、llama 走 Vulkan |
| iGPU (AMD/Intel) | `/dev/dri` 存在 | llama 走 Vulkan、ONNX 走 CPU |
| 無 GPU | 都不在 | 整條 pipeline 走 CPU（功能正常但慢） |

集顯（iGPU）若 GGUF 載入或解碼出錯，取消 [`.env.example`](.env.example) 內的 GPU 相容性 patch：

```bash
GGML_VK_DISABLE_COOPMAT=1   # AMD iGPU 無法載入 GGUF 時
GGML_VK_DISABLE_F16=1       # iGPU 解碼錯誤時
```

詳見 [docs/docker-server.md](docs/docker-server.md)。

---

## 架構速覽

本 fork 的 server / Web / CLI 加值都住在獨立目錄；目前刻意 diverge 的 upstream-tracked 檔案只有 `.gitignore`、`readme.md`、`requirements-server.txt`、`LLM/default.py`、`assets/BUILD_GUIDE.md`：

```
fork_server/              ← Sidecar 套件 (上游無此目錄)
├── env_config.py         ← env → ServerConfig 屬性
├── bootstrap.py          ← ForkedCapsWriterServer (子類化 upstream)
└── http_api/             ← OpenAI Whisper API
docker/                   ← Container 構建 (上游無此目錄)
docker-compose*.yml       ← Compose 部署
.env.example              ← env 變數一覽
start_server_docker.py    ← Fork entrypoint (與上游 start_server.py 並存)
```

詳細設計請看 [docs/architecture.md](docs/architecture.md)。

未來要拉上游更新只需 `git fetch origin && git merge origin/master`，
參見 [docs/upstream-sync-guide.md](docs/upstream-sync-guide.md)。

---

## 文件地圖

| 文件 | 內容 |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Sidecar 設計、hook 策略、已知 upstream divergence |
| [docs/upstream-sync-guide.md](docs/upstream-sync-guide.md) | 未來如何 `git merge origin/master`、衝突處理 SOP |
| [docs/HTTP_API.md](docs/HTTP_API.md) | OpenAI Whisper API 規格、5 種 response_format、認證、SDK 範例 |
| [docs/web-console.md](docs/web-console.md) | Web Console STT/TTS 工作台、CORS、隔離開發、清理與驗證流程 |
| [docs/cli-client.md](docs/cli-client.md) | 無 GUI CLI client、批次轉錄、本機 TTS、跨平台驗證與清理 |
| [docs/verification.md](docs/verification.md) | repo-level 驗證、CI、live HTTP check、自動清理策略 |
| [docs/docker-server.md](docs/docker-server.md) | 容器部署細節、GPU 設定、env 變數完整表 |
| [docs/state-of-fork.md](docs/state-of-fork.md) | Fork 當前狀態快照（divergence、已驗證項目） |

---

## 與上游的關係

| 面向 | 上游 | 本 Fork |
|---|---|---|
| 主要場景 | Windows 桌面語音輸入 | Linux 服務端 ASR |
| 部署 | `start_server.py` 直跑 | Docker container |
| 介面 | WebSocket | WebSocket + OpenAI HTTP API |
| 識別引擎 | 完全相同 | 完全相同（直接用上游 `core/server/engines/`） |
| Client | 上游有完整 Windows client + GUI | **未改動**，沿用上游 |

本 fork **不重新發明識別、熱詞、LLM 角色**，那些都是上游的功能，直接受惠於上游更新。

---

## 授權與貢獻

- License: 與上游同 MIT（[LICENSE](LICENSE)）
- 貢獻：bug report、PR 都歡迎。改 server 端 / Docker / HTTP API 相關的請開 issue；改識別引擎本身請去上游
- 上游：[HaujetZhao/CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline)
- 致謝：上游作者 [@HaujetZhao](https://github.com/HaujetZhao) 與 sherpa-onnx / FunASR / Qwen3-ASR / llama.cpp 社群

---

## FAQ

**Q: 跟上游有什麼不同？我可以直接用上游嗎？**
A: 上游聚焦 Windows 桌面語音輸入。如果你要的就是「Windows 上按 CapsLock 講話 → 自動打字」，用上游。本 fork 的價值在於「把 server 跑成 Linux 服務、給其他應用用 OpenAI SDK 接」。

**Q: 為什麼 image 這麼大（5 GB）？**
A: CUDA runtime base + Python deps + sherpa-onnx + ONNX runtime GPU。模型本身是另外掛 volume 不在 image 內。

**Q: 容器內 CUDA 11.8 是否會跟我的 NVIDIA driver 衝突？**
A: 不會。CUDA Toolkit 11.8 runtime 對驅動的要求是 ≥520。較新驅動相容（forward compat）。

**Q: 我有舊版 fork 跑著（util/server/ 結構），怎麼升級？**
A: `docker compose pull && docker compose up -d --force-recreate`。env 變數命名向後相容。
