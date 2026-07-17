# Fork 架構：Sidecar + Audited Touchpoints

> 本文件解釋 fork 如何把 Linux container 部署、OpenAI HTTP API、Web Console 與 no-GUI CLI 放在可審查的低漂移架構中，同時清楚列出刻意 diverge 的 upstream-tracked 檔案。

## 1. 設計原則

**最高原則**：主要產品功能優先住在 fork-owned 路徑；無法 sidecar 化的相容性與
安全修正必須限制在精確 allowlist，並為 upstream merge 寫明理由與 regression。

| 目標 | 怎麼做到 |
|---|---|
| 未來 `git merge origin/master` 可審查 | 主要 fork 加值在獨立目錄 `fork_server/` / `docker/` / `client/{cli,web,tui}/`；59 個 upstream-tracked divergence 由 guard 與同步 SOP 管理 |
| 上游更新識別引擎時可持續受惠 | 模型推論算法仍以 upstream 為基線；`core/server/engines/` 只保留已記錄的 bounded I/O 與 privacy logging 接觸點 |
| 易理解、易維護 | Sidecar hook 使用子類化／module replacement；直接接觸 upstream 的部分則按責任分組並由 regression 固定契約 |
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
    ├── api.py                          ← FastAPI app + 5 endpoints
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
requirements-server-docker.txt          ← Linux GPU 版 top-level 依賴
requirements-server-docker.lock         ← Docker image runtime transitive dependency version/hash lock
.github/workflows/publish-server-image.yml  ← GHCR 自動發 image

start_server_docker.py                  ← Fork 入口 (與上游 start_server.py 並存)
```

**刻意 diverge 的 upstream-tracked 檔案：59 個**。以下用 merge/review
責任分組；括號數量合計即 guard 的完整 allowlist。

| 群組（數量） | Upstream-tracked 路徑 | 保留原因 |
|---|---|---|
| Repository、build、release（7） | `.gitignore`、`CLAUDE.md`、`assets/BUILD_GUIDE.md`、`build.spec`、`readme.md`、`requirements-server.txt`、`zip_release.py` | Fork 視角與 contributor metadata、universal Windows server packaging、可重現 dependency、cache/secret hygiene，以及 bounded release subprocess |
| LLM role 安全預設（3） | `LLM/default.py`、`LLM/大助理.py`、`docs/角色功能如何使用.md` | 移除 API-key-like placeholder、停用開箱即聯網角色，並讓使用說明與安全預設一致 |
| Desktop portability、bounded audio 與 lifecycle（24） | `core/client/audio/{file_manager.py,recorder.py,stream.py}`、`core/client/clipboard/clipboard.py`、`core/client/connection/websocket_manager.py`、`core/client/global_hotkey/{__init__.py,global_hotkey.py}`、`core/client/hotword/{hotword_standalone.py,hotword_standalone.ipynb}`、`core/client/llm/llm_output_typing.py`、`core/client/manager/{file_runner.py,tray_manager.py}`、`core/client/output/{result_processor.py,text_output.py}`、`core/client/shortcut/{emulator.py,key_mapper.py,shortcut_manager.py}`、`core/client/state.py`、`core/client/transcribe/{file_transcriber.py,media_tool.py,srt_adjuster.py}`、`core/tools/window_detector.py`、`core/ui/tray.py`、`start_client.py` | 保留 Windows 行為與 artifact self-check，同時讓 headless／pure Wayland 啟動不需載入 `pynput`，並提供 X11/unsupported-session detection、Windows `keyboard.write`／Linux non-root `pynput` text injection、bounded callback/queue/WebSocket、ordered audio、UUID4 task identity、concurrent file receive/deadline，以及所有外部 desktop/media process 的 timeout、detached stdio 與 cleanup guard |
| Protocol 與安全錯誤傳遞（3） | `core/protocol.py`、`core/server/connection/ws_send.py`、`core/server/schema.py` | 在不破壞成功訊息相容性的前提下，傳遞安全的 worker error，並在 `Task` 尾端加入 HTTP privacy/deadline metadata |
| WebSocket ingress 與 transport controls（2） | `core/server/connection/{server_manager.py,ws_recv.py}` | 依 configured bind 解析 IPv4/IPv6 family；限制 frame/prefetch，嚴格驗證 metadata/PCM/單一 active stream，並在 executor thread 對 bounded worker queue 施加 backpressure |
| Worker/service resource controls（8） | `core/server/app.py`、`core/server/state.py`、`core/server/worker/{__init__.py,gpu_boost.py,pipeline.py,process_manager.py,task_handler.py,worker.py}` | Task-local transcript privacy、bounded result dispatch/fair scheduling/shutdown、cross-process deadline、safe error result、GPU/stop timeout，以及 parent inference watchdog/fail-stop |
| Engine export I/O（3） | `core/server/engines/{force_aligner_gguf,fun_asr_gguf,qwen_asr_gguf}/export/gguf/utility.py` | Remote safetensor `GET`/`HEAD` 使用 bounded timeout |
| Engine audio decode I/O（4） | `core/server/engines/{force_aligner_gguf,fun_asr_gguf,qwen_asr_gguf,sensevoice_onnx}/inference/audio.py` | Direct file decode 的 `ffmpeg` 使用 bounded timeout、kill cleanup 與 stderr preview |
| Engine privacy logging（3） | `core/server/engines/{force_aligner_gguf,qwen_asr_gguf}/inference/aligner.py`、`core/server/engines/fun_asr_gguf/inference/prompt_builder.py` | Privacy-off HTTP task 不把 token、prompt、context 或 audio-derived detected hotword 寫入 log；模型數學與輸出語意不變 |
| Upstream 文件正確性／a11y（2） | `docs/text_merge_algorithm.md`、`docs/显卡加速的若干问题.md` | 對齊目前 text-merger 實作與補上有意義的圖像替代文字 |

`scripts/check_upstream_divergence.py` 以 `origin/master` 直接對工作樹比較，
因此已提交、staged 與 unstaged 的 tracked 修改都會進入這 59 檔檢查；fork-only
新增路徑與 untracked work-in-progress 不會被誤算成 upstream divergence。

## 4. Hook 策略

下表是 fork 的主要 sidecar hook 點；每個都選了**最小侵入**的方法。

| 介入點 | 方法 | 為什麼這樣選 |
|---|---|---|
| **env → config** | `env_config.apply()` 用 setattr 改 ServerConfig / Args classes 屬性，必須在 `import core.server.*` 之前 | 上游 config 是 class-based，setattr 自然不需動到上游檔。`import` 之前是因為 `core/server/__init__.py` 在 import 時 snapshot `Config.log_level` |
| **絕對路徑化 ModelPaths** | `env_config._absolutize_model_paths()` | 上游某些 inference 模組會 `os.chdir(lib_dir)` 載入 llama.cpp，cwd 可能不一致。絕對路徑一勞永逸 |
| **加 HTTP API cotask** | `ForkedCapsWriterServer.start()` 在 `Config.http_api_enable=True` 時，以 `_run_fork_services()` 監督 WebSocket、HTTP 與 shutdown event；任一先完成即進入 bounded cleanup | 上游 `start()` 用單一 `run_until_complete(socket_manager.start())`。子類化覆寫 `start()` 比子類化 SocketManager 更小，且不讓單一 listener 結束後留下另一個服務 |
| **HTTP 結果攔截** | Monkey-patch `core.server.connection.server_manager.ws_send` 為 `ws_send_with_http` | `server_manager.py` 在 module-load 時 `from .ws_send import ws_send`，已 import 後改 module attribute 即生效。比子類化整個 SocketManager 簡潔 |
| **HTTP 任務注入** | HTTP handler 在 thread 內以短 timeout 重試 bounded `state.queue_in.put(Task(..., socket_id=f"http:{task_id}"))` | 避免阻塞 asyncio loop；每次重試都檢查 deadline、cancel 與 synthetic socket。upstream `TaskHandler` 用 `task.socket_id not in sockets_id` 跳過孤兒任務 |

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
              │ HTTP Future      │   │ ws_send 定向派發 │
              │ .set_result() or │   │ (與上游同)       │
              │ absorb late HTTP │   │                  │
              └──────────────────┘   └──────────────────┘
                       │                       │
                       ▼                       ▼
              uvicorn 回應 HTTP        WebSocket client
```

關鍵：兩條路徑共用同一個 `queue_out` 與同一個識別子進程。HTTP 任務的 `socket_id = "http:<task_id>"`，永遠不會與真實 WebSocket 的 socket id 碰撞。timeout 或 client cancel 會移除 pending future 與合成 socket id；若 recognizer 之後才送回該 HTTP task 的結果，`TaskRouter` 會用 bounded tombstone 吸收，避免落入 WebSocket 派發。

輸入端不是無界 queue：跨 process `queue_in` 固定最多 8 個 Task，worker 內公平
round-robin buffer 另最多 8 個。每個 audio Task 最大 4,096,000 bytes；HTTP 與
WebSocket producer 都在容量不足時施加 backpressure，WS 以 non-blocking put 在
event loop 上 cooperative retry。WebSocket transport 另限制每個 admitted connection
為 6 MiB frame 與一個 queued message；預設 admission 上限是 8 個 connection，超額
以 `1013` 拒絕，close handshake 一秒後仍未完成便 abort。

每個 WebSocket task 的累計 audio 預設上限為 3,600 秒（可設定範圍
`1..86400`）。超限 frame 不會加入 audio queue；ingress cache 會 reset，接著以內部
control task 清除且只清除該 `(socket_id, task_id)` 的 worker buffer/session，並回傳
final `websocket_task_audio_limit_exceeded` error。Connection、同 socket 的其他 task，
以及其他 socket 上碰撞的 task ID 都不受影響。

輸出端的跨 process `queue_out` 固定最多 8 筆。每個 WebSocket peer 只有一個 active
send 與最多 8 個依 task 排序的 pending snapshot；同 task intermediate 可 coalesce，
跨 task final 仍保留。Send 超過五秒或 pending overflow 只會隔離該 peer，不會卡住
HTTP result resolution 或其他 WebSocket client。

Worker 的公平 buffer 與 recognition session 都以 `(socket_id, task_id)` 作為內部
identity，而 wire result 仍保留原本的 `task_id`。因此兩個 WebSocket client 即使送出
相同 task id，也不會合併 transcript 或把結果送錯連線；pipeline failure 會清掉該
composite session 尚未處理的 segments，但不影響另一個 socket 的同名 task。
WebSocket ingress 另強制 source／boolean final flag、有限且 sample-aligned 的分段幾何、
有界且無 control character 的 task/context/language，以及每條連線同時只能有一個
task/source stream。違規 message 會以 policy code `1008` 關閉，不會 log-and-continue。

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
            _run_fork_services(...)
              ├─ WebSocket listener + ws_send_with_http
              ├─ uvicorn HTTP server
              ├─ shutdown event
              └─ FIRST_COMPLETED → bounded cleanup/reap all cotasks
          )
       else:
         loop.run_until_complete(
           socket_manager.start()           ← 沿用 upstream WebSocket contract
         )
```

## 7. 需 source-sync 的高漂移點

`fork_server/http_api/ws_send_with_http.py` **內嵌了上游 `ws_send` 函式邏輯的複本**，僅在開頭加了一行 `if task_router.try_resolve(result): continue`。

若上游 `core/server/connection/ws_send.py` 將來簽名或邏輯變更（例如新增 Result 欄位、改定向派發協議），需要手動 re-port 到此檔。

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

目前 (2026-07-17) 為止：第三類只包含上方 59 個已知檔案；不要新增未記錄的 upstream divergence。
