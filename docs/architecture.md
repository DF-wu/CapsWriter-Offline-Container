# Fork 架構：Sidecar 設計

> 本文件解釋 fork 如何在「修改 0 個上游檔案」的前提下，加上 Linux container 部署 + OpenAI HTTP API。

## 1. 設計原則

**最高原則**：fork 的每一行新代碼都住在「不會與上游路徑衝突」的位置。

| 目標 | 怎麼做到 |
|---|---|
| 未來 `git merge origin/master` 零衝突 | 所有 fork 加值在獨立目錄 `fork_server/` / `docker/` / 獨立檔名 |
| 上游更新識別引擎時自動受惠 | 不修改 `core/server/engines/`，直接用 `EngineFactory` |
| 易理解、易維護 | 每個 hook 點都是「子類化」或「module attribute 替換」，零侵入 |
| 容器啟動可預測 | env 驅動的設定 + 自動模型下載 + GPU 偵測 + CPU fallback |

## 2. 上游結構回顧（origin/master @ 0362630, v2.5）

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

**修改的上游檔案：1 個**（`.gitignore`，加入 `.fork-archive-*/` 一行）。

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
                │  task_id ∈ pending HTTP futures?         │
                └─────────┬───────────────────┬───────────┘
                   YES    │                   │  NO
                          ▼                   ▼
              ┌──────────────────┐   ┌──────────────────┐
              │ HTTP Future      │   │ ws_send 廣播     │
              │ .set_result()    │   │ (與上游同)       │
              └──────────────────┘   └──────────────────┘
                       │                       │
                       ▼                       ▼
              uvicorn 回應 HTTP        WebSocket client
```

關鍵：兩條路徑共用同一個 `queue_out` 與同一個識別子進程。HTTP 任務的 `socket_id = "http:<task_id>"`，永遠不會與真實 WebSocket 的 socket id 碰撞。

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

**漂移偵測**：merge 上游後跑 `git diff origin/master:core/server/connection/ws_send.py fork_server/http_api/ws_send_with_http.py` 看核心 loop 是否還對齊。詳見 [upstream-sync-guide.md](upstream-sync-guide.md)。

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
3. **第三選擇**：直接修改上游檔。**這時要在該檔頂部加註解標記 fork-modified，並在 `upstream-sync-guide.md` 的「known divergent files」加一筆**

目前 (2026-05-25) 為止：**第三類為 0**。
