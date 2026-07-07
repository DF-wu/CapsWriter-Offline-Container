# Fork 架構：Sidecar 設計

> 本文件解釋 fork 如何把 Linux container 部署、OpenAI HTTP API、Web Console 與 no-GUI CLI 放在低漂移架構中，同時清楚列出少數刻意 diverge 的 upstream-tracked 檔案。

## 1. 設計原則

**最高原則**：fork 的每一行新代碼都住在「不會與上游路徑衝突」的位置。

| 目標 | 怎麼做到 |
|---|---|
| 未來 `git merge origin/master` 低衝突 | 主要 fork 加值在獨立目錄 `fork_server/` / `docker/` / `client/cli/` / `client/web/`；少數 upstream-tracked divergence 由同步 SOP 管理 |
| 上游更新識別引擎時自動受惠 | 不修改 `core/server/engines/`，直接用 `EngineFactory` |
| 易理解、易維護 | 每個 hook 點都是「子類化」或「module attribute 替換」，零侵入 |
| 容器啟動可預測 | env 驅動的設定 + 自動模型下載 + GPU 偵測 + CPU fallback |

## 2. 上游結構回顧（origin/master @ 7d7fac3, upstream v2.6）

```
core/
├── server/
│   ├── app.py                          ← CapsWriterServer (facade)
│   ├── state.py                        ← ServerState (queue_in/out, sockets_id)
│   ├── schema.py                       ← Task / Result dataclasses
│   ├── connection/
│   │   ├── server_manager.py           ← SocketManager (websockets.serve)
│   │   ├── ws_recv.py                  ← 收音訊 → state.queue_in
│   │   └── ws_send.py                  ← state.queue_out → websocket.send
│   ├── worker/
│   │   ├── process_manager.py          ← 拉識別子進程
│   │   ├── worker.py                   ← RecognizerWorker
│   │   ├── task_handler.py             ← 收 queue_in / 派 queue_out
│   │   ├── pipeline.py
│   │   └── model_loader.py
│   └── engines/
│       ├── base.py                     ← BaseASREngine ABC
│       ├── factory.py                  ← EngineFactory.create_asr_engine()
│       ├── qwen_asr_gguf/              ← Qwen3-ASR 引擎
│       ├── fun_asr_gguf/               ← Fun-ASR-Nano 引擎
│       ├── ct_transformer/             ← 標點引擎
│       └── force_aligner_gguf/         ← 字級時間戳對齊
config_server.py                        ← Class-based config (純 attributes)
start_server.py                         ← 上游入口
```

上游無 `docker/`、無 `docker-compose*.yml`、無 HTTP API。

## 3. Fork 加值結構

```
fork_server/                            ← Sidecar 套件 (上游無此目錄)
├── __init__.py
├── env_config.py                       ← env → ServerConfig / Args 屬性
├── bootstrap.py                        ← ForkedCapsWriterServer 子類
└── http_api/
    ├── __init__.py
    ├── api.py                          ← FastAPI app + 4 endpoints
    ├── task_router.py                  ← HTTP task ↔ asyncio.Future routing
    ├── audio_decoder.py                ← ffmpeg → PCM
    ├── openai_formatter.py             ← Result → 5 種 response_format
    ├── ws_send_with_http.py            ← 上游 ws_send + HTTP try_resolve
    └── serve.py                        ← uvicorn cotask

docker/                                 ← Container 構建 (上游無此目錄)
├── server/
│   ├── Dockerfile                      ← CUDA 11.8 + Python 3.10 + tini
│   ├── entrypoint.sh                   ← GPU 偵測 + env defaults
│   ├── download_models.py              ← 模型 + llama.cpp .so 下載
│   ├── probe_backend.py                ← 啟動前 GPU smoke test
│   └── healthcheck.py                  ← WebSocket probe + optional HTTP /ready
docker-compose.yml                      ← 主 compose
docker-compose.fun-asr.yml              ← Fun-ASR override
docker-compose.example.yml              ← 範例
.env.example                            ← env 變數一覽
.dockerignore
requirements-server-docker.txt          ← Linux GPU 版依賴
.github/workflows/publish-server-image.yml  ← GHCR 自動發 image

start_server_docker.py                  ← Fork 入口 (與上游 start_server.py 並存)
```

**刻意 diverge 的 upstream-tracked 檔案：13 個**。

| 檔案 | 原因 |
|---|---|
| `.gitignore` | 排除 Web/verification cache、model download cache、versioned shared libraries 與本地工具狀態 |
| `readme.md` | 中文首頁改成 fork 視角，避免把 Linux server fork 說成 upstream Windows desktop product |
| `requirements-server.txt` | 補上裸機 HTTP API runtime dependencies，讓 `start_server_docker.py` 與 diagnostic imports 可重現 |
| `LLM/default.py` | 移除 upstream template 內的 API-key-like placeholder，避免 repository secret scanning / 使用者誤啟用 |
| `assets/BUILD_GUIDE.md` | 讓打包文件列出目前 server dependency set |
| `zip_release.py` | legacy PyInstaller ZIP packaging 的 7-Zip subprocess timeout 與失敗後 temp file cleanup 需要 release-grade guard |
| `core/client/hotword/hotword_standalone.py` | local Ollama chat helper 需 bounded request timeout，避免未回應的本機 LLM endpoint 卡住 demo/client 流程 |
| `core/server/engines/qwen_asr_gguf/inference/audio.py` | direct engine file transcription 的 `ffmpeg` decode 需 bounded timeout/error preview |
| `core/server/engines/force_aligner_gguf/inference/audio.py` | direct aligner file decode 的 `ffmpeg` subprocess 需 bounded timeout/error preview |
| `core/server/engines/sensevoice_onnx/inference/audio.py` | SenseVoice direct file decode 的 `ffmpeg` subprocess 需 bounded timeout/error preview |
| `core/server/engines/fun_asr_gguf/inference/audio.py` | Fun-ASR direct file decode 改用 bounded `ffmpeg` subprocess，避免 pydub/ffmpeg path 卡住 |
| `core/server/worker/gpu_boost.py` | server GPU boost/unboost shell command 需 bounded timeout，避免自訂管理命令卡住 worker loop |
| `core/tools/window_detector.py` | macOS/Linux foreground-window helper 需 bounded `osascript`/`wmctrl` subprocess，避免桌面 client output path 被卡住 |

## 4. Hook 策略

下表是 fork 介入上游的 4 個點。每個都選了**最小侵入**的方法。

| 介入點 | 方法 | 為什麼這樣選 |
|---|---|---|
| **env → config** | `env_config.apply()` 用 setattr 改 ServerConfig / Args classes 屬性，必須在 `import core.server.*` 之前 | 上游 config 是 class-based，setattr 自然不需動到上游檔。`import` 之前是因為 `core/server/__init__.py` 在 import 時 snapshot `Config.log_level` |
| **絕對路徑化 ModelPaths** | `env_config._absolutize_model_paths()` | 上游某些 inference 模組會 `os.chdir(lib_dir)` 載入 llama.cpp，cwd 可能不一致。絕對路徑一勞永逸 |
| **加 HTTP API cotask** | `ForkedCapsWriterServer.start()` 在 `Config.http_api_enable=True` 時, `asyncio.gather(socket_manager.start(), run_http_server())` | 上游 `start()` 用單一 `run_until_complete(socket_manager.start())`。子類化覆寫 `start()` 比子類化 SocketManager 更小 |
| **HTTP 結果攔截** | Monkey-patch `core.server.connection.server_manager.ws_send` 為 `ws_send_with_http` | `server_manager.py` 在 module-load 時 `from .ws_send import ws_send`，已 import 後改 module attribute 即生效。比子類化整個 SocketManager 簡潔 |
| **HTTP 任務注入** | HTTP handler 直接 `state.queue_in.put(Task(..., socket_id=f"http:{task_id}"))` | upstream `TaskHandler` 用 `task.socket_id not in sockets_id` 跳過孤兒任務，fork 把合成 id 註冊到 sockets_id ListProxy 即解決 |

## 5. 識別結果路由（HTTP 任務 vs WebSocket）

```
                    ┌─────────────────────────────────────────┐
                    │  Recognizer Subprocess (worker)         │
                    │  consumes Task from queue_in,           │
                    │  produces Result to queue_out           │
                    │  routes by Task.socket_id               │
                    └────────────────┬────────────────────────┘
                                     │
                                     ▼
                            state.queue_out (mp.Queue)
                                     │
                                     │ ws_send_with_http
                                     ▼
                ┌────────────────────┴────────────────────┐
                │  task_router.try_resolve(result)?       │
                │                                          │
                │  task_id ∈ pending HTTP futures          │
                │  or canceled HTTP tombstones?            │
                └─────────┬───────────────────┬───────────┘
                   YES    │                   │  NO
                          ▼                   ▼
              ┌──────────────────┐   ┌──────────────────┐
              │ HTTP Future      │   │ ws_send 廣播     │
              │ .set_result() or │   │ (與上游同)       │
              │ absorb late HTTP │   │                  │
              └──────────────────┘   └──────────────────┘
                       │                       │
                       ▼                       ▼
              uvicorn 回應 HTTP        WebSocket client
```

關鍵：兩條路徑共用同一個 `queue_out` 與同一個識別子進程。HTTP 任務的 `socket_id = "http:<task_id>"`，永遠不會與真實 WebSocket 的 socket id 碰撞。timeout 或 client cancel 會移除 pending future 與合成 socket id；若 recognizer 之後才送回該 HTTP task 的結果，`TaskRouter` 會用 bounded tombstone 吸收，避免落入 WebSocket 派發。

## 6. 啟動流程

```
start_server_docker.py
  ↓
fork_server.bootstrap.apply_env_config()   ← env → ServerConfig / Args
                                              絕對路徑化 ModelPaths
  ↓                                         (必須在 import core.server 之前!)
fork_server.bootstrap.create_server()
  ↓
ForkedCapsWriterServer().start()
  ↓
  ├─ register_signal(self.stop)
  ├─ tray_manager.start()                  ← Linux 環境會 no-op
  ├─ process_manager.start()               ← 拉識別子進程, 載入模型
  │                                          (state.sockets_id 在這初始化)
  └─ if Config.http_api_enable:
       ├─ _install_ws_send_hook()           ← 替換 server_manager.ws_send
       └─ loop.run_until_complete(
            asyncio.gather(
              socket_manager.start(),       ← WebSocket + ws_send_with_http
              run_http_server(self),        ← uvicorn FastAPI
            )
          )
       else:
         loop.run_until_complete(
           socket_manager.start()           ← 完全等同上游
         )
```

## 7. 唯一的漂移點

`fork_server/http_api/ws_send_with_http.py` **內嵌了上游 `ws_send` 函式邏輯的複本**，僅在開頭加了一行 `if task_router.try_resolve(result): continue`。

若上游 `core/server/connection/ws_send.py` 將來簽名或邏輯變更（例如新增 Result 欄位、改廣播協議），需要手動 re-port 到此檔。

**漂移偵測**：HTTP unit test 會用 AST 比對 upstream `ws_send` 與 fork 版本（只允許 HTTP `task_router.try_resolve` hook 與 log 文字差異）。merge 上游後若 `python -m unittest fork_server.http_api.tests.test_ws_send_with_http -v` 失敗，先比對 `core/server/connection/ws_send.py` 並手動 re-port，再跑完整 gate。詳見 [upstream-sync-guide.md](upstream-sync-guide.md)。

## 8. 為什麼不用 monkey-patch 全部？

考慮過用 100% monkey-patching（runtime 改 ws_send / SocketManager 的 method）。優點是字面 0 個檔修改。缺點：
- 偵錯困難（看程式碼看不到行為）
- 多人協作時 onboarding 成本高
- 不安全：upstream 重命名屬性時 patch 變成 silent no-op

最終取捨：
- 「子類化」用在 `ForkedCapsWriterServer` 與 HTTP handler — 顯式、可讀
- 「Monkey-patch」只用在 `ws_send` 一個地方 — 因為要避開複製 SocketManager.start 整段 async with 結構

## 9. 想動上游的話怎麼辦？

如果遇到必須修改上游檔案的情況（例如上游 bug 阻擋 fork 功能）：

1. **第一選擇**：上游 PR 修 bug，等合入
2. **第二選擇**：fork 內 monkey-patch（runtime 替換）
3. **第三選擇**：直接修改上游檔。這時必須在本文件與 `upstream-sync-guide.md` 的 known divergent files 清單加一筆，說明原因與 merge 時的處理方式。

目前 (2026-07-07) 為止：第三類只包含上方 13 個已知檔案；不要新增未記錄的 upstream divergence。
