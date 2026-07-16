# 對齊上游指南

> 未來要拉上游更新時的標準操作流程。主要功能維持 sidecar，但 59 個已記錄的 upstream-tracked 接觸點仍須依群組 review；不要把 merge 成功等同於同步完成。

## 1. 心智模型

```
upstream (HaujetZhao/CapsWriter-Offline)
        │
        │  fork modifies 59 upstream-tracked files:
        │    repo/build/release + role/docs safety       (12)
        │    desktop portability and bounded lifecycle  (24)
        │    protocol, WebSocket and worker controls    (13)
        │    engine bounded I/O and privacy logging     (10)
        │  fork adds:    fork_server/ docker/ client/{cli,web,tui}/
        │                docs/ scripts/ docker-compose*.yml .env.example
        │                .github/workflows/ requirements-server-docker.txt
        │
        ▼
fork (DF-wu/CapsWriter-Offline-Container) master/feat/*
```

關鍵：fork 修改 upstream-tracked 檔案數 = **59**。主要產品功能仍優先放在
fork-owned 新路徑；這 59 檔是需要逐組 rebase/merge review 的明確邊界。

| 群組（數量） | Upstream-tracked 路徑 | Merge 時必須保留／重驗 |
|---|---|---|
| Repository、build、release（7） | `.gitignore`、`CLAUDE.md`、`assets/BUILD_GUIDE.md`、`build.spec`、`readme.md`、`requirements-server.txt`、`zip_release.py` | 合併 upstream metadata/dependencies；保留 universal server packaging、fork README、安全 ignore 與 bounded release cleanup |
| LLM role 安全預設（3） | `LLM/default.py`、`LLM/大助理.py`、`docs/角色功能如何使用.md` | 保留空 key、disabled network role 與相符文件；同步其他 upstream prompt/template 欄位 |
| Desktop portability、bounded audio 與 lifecycle（24） | `core/client/audio/{file_manager.py,recorder.py,stream.py}`、`core/client/clipboard/clipboard.py`、`core/client/connection/websocket_manager.py`、`core/client/global_hotkey/{__init__.py,global_hotkey.py}`、`core/client/hotword/{hotword_standalone.py,hotword_standalone.ipynb}`、`core/client/llm/llm_output_typing.py`、`core/client/manager/{file_runner.py,tray_manager.py}`、`core/client/output/{result_processor.py,text_output.py}`、`core/client/shortcut/{emulator.py,key_mapper.py,shortcut_manager.py}`、`core/client/state.py`、`core/client/transcribe/{file_transcriber.py,media_tool.py,srt_adjuster.py}`、`core/tools/window_detector.py`、`core/ui/tray.py`、`start_client.py` | 先同步 upstream Windows 行為，再重套 artifact self-check、headless／pure Wayland lazy input imports、X11/unsupported-session、Windows `keyboard.write`／Linux non-root `pynput` text injection、bounded callback/queue/WebSocket、ordered stream、UUID4 與 concurrent file deadline contract；跑 desktop portability/backpressure regressions |
| Protocol 與安全錯誤傳遞（3） | `core/protocol.py`、`core/server/connection/ws_send.py`、`core/server/schema.py` | 保留 optional error fields 與尾端 HTTP task metadata；重跑 protocol 和 `ws_send_with_http` drift tests |
| WebSocket ingress 與 transport controls（2） | `core/server/connection/{server_manager.py,ws_recv.py}` | 保留 IPv4/IPv6 bind preflight、bounded frame/prefetch、off-loop queue backpressure、嚴格 metadata/PCM/active-stream policy close；跑 server queue/protocol regressions |
| Worker/service resource controls（8） | `core/server/app.py`、`core/server/state.py`、`core/server/worker/{__init__.py,gpu_boost.py,pipeline.py,process_manager.py,task_handler.py,worker.py}` | 保留 privacy、deadline、bounded result dispatch／shutdown、公平排程、安全錯誤、GPU command 及 inference watchdog；同步 upstream service/worker lifecycle 後跑 regressions |
| Engine export I/O（3） | `core/server/engines/{force_aligner_gguf,fun_asr_gguf,qwen_asr_gguf}/export/gguf/utility.py` | 保留 `CAPSWRITER_GGUF_EXPORT_HTTP_TIMEOUT`；同步 upstream export logic |
| Engine audio decode I/O（4） | `core/server/engines/{force_aligner_gguf,fun_asr_gguf,qwen_asr_gguf,sensevoice_onnx}/inference/audio.py` | 保留 `CAPSWRITER_ENGINE_FFMPEG_TIMEOUT`、kill cleanup、bounded stderr；同步 upstream decode logic |
| Engine privacy logging（3） | `core/server/engines/{force_aligner_gguf,qwen_asr_gguf}/inference/aligner.py`、`core/server/engines/fun_asr_gguf/inference/prompt_builder.py` | 保留 token／prompt／context／audio-derived detected-hotword 的 task-local redaction，但逐段同步 upstream prompt/alignment 語意；跑 worker privacy regressions |
| Upstream 文件正確性／a11y（2） | `docs/text_merge_algorithm.md`、`docs/显卡加速的若干问题.md` | 保留與現行 merger 一致的說明及有意義的 image alt text；同步 upstream 其他內容 |

## 2. 標準同步流程

### 2.1 預設情境：upstream 沒改動 fork 在意的檔案

```bash
git fetch origin
git checkout master
git merge --ff-only origin/master   # 試 fast-forward
```

如果 `--ff-only` 失敗，代表 fork 有自己的 commits 領先：

```bash
git merge origin/master
```

預期結果：衝突應只落在上方 59 個已知 divergent files，並依群組處理；
任何第 60 個 upstream-tracked 路徑都先視為未記錄 drift。

跑驗證：

```bash
# 1. Repo gate
python scripts/verify_all.py

# 2. Container 構建
docker build -t capswriter-server:upstream-merge-test -f docker/server/Dockerfile .

# 3. Web image smoke (optional but recommended after Web changes)
python scripts/verify_all.py --skip-web --docker-build-web
```

### 2.2 高風險情境：upstream 改了 fork 接觸點

下表是 fork 對上游的「間接依賴」。**任何一項變動都應觸發完整 re-test**。

| 上游檔/符號 | 變動類型 | Fork 對應檔 | Fork 動作 |
|---|---|---|---|
| `core/server/connection/ws_send.py` | 函式邏輯、Result 欄位、訊息協議 | `fork_server/http_api/ws_send_with_http.py` | HTTP unit test 會偵測 drift；失敗時 **手動 re-port** 上游修改 |
| `core/server/connection/server_manager.py` | 重命名 `ws_send` import 或 `SocketManager.start` 結構 | `fork_server/bootstrap.py::_install_ws_send_hook` | 確認 module attribute 名稱仍正確 |
| `core/server/app.py` | `CapsWriterServer.start()` 流程 (signal/tray/process/socket 順序) | `fork_server/bootstrap.py::ForkedCapsWriterServer.start` | 比對覆寫版是否需同步調整 |
| `core/server/schema.py` | `Task` 或 `Result` 新欄位 | `fork_server/http_api/{api.py,task_router.py,ws_send_with_http.py}` | 保留尾端的 HTTP privacy/deadline 欄位與既有 positional ABI；執行 protocol regression |
| `core/server/worker/task_handler.py` | queue drain、dispatch、session cleanup | HTTP cross-process deadline + fair bounded scheduling + safe worker errors | 手動合併 upstream loop 變更，再跑 `scripts/tests/test_task_handler.py` |
| `core/server/worker/pipeline.py` | transcript/context log 或 console site | `Task.log_transcript` / `core/server/privacy.py` | 所有新 text-bearing site 必須套用 task-local privacy policy；跑 worker privacy regression |
| `core/server/engines/fun_asr_gguf/inference/prompt_builder.py` | prompt/context 組裝或 logging | HTTP prompt privacy | 保留 prompt 語意，但在 privacy-off task 中只記錄 redacted metadata |
| `core/server/engines/{force_aligner_gguf,qwen_asr_gguf}/inference/aligner.py` | degraded-match logging | HTTP transcript privacy | Token warning 必須遵守 task-local privacy policy |
| `config_server.py` | `Qwen3ASRGGUFArgs` / `FunASRNanoGGUFArgs` / `ModelPaths` 新增屬性或重命名 | `fork_server/env_config.py` | 加新 env binding、移除過時的 |
| `core/server/engines/factory.py` | `EngineFactory.create_asr_engine()` 簽名 | `docker/server/probe_backend.py` | 適配 probe |
| `core/server/worker/model_loader.py` | 模型載入順序、新模型類型 | `docker/server/download_models.py` | 新增 ASSETS 條目 |

### 2.3 衝突處理 SOP

如果 merge 出現衝突：

#### 衝突在 `.gitignore`
99% auto-merge 成功。手動處理也只是兩邊加的 ignore 規則合在一起。

#### 衝突在 `readme.md`
本 fork 的 readme 完全是 fork 視角。**保留 fork 版本**：

```bash
git checkout --ours readme.md
git add readme.md
```

如果上游 readme 有重要新內容（例如新模型支援），手動把那段引用到 fork readme 對應段落。

#### 衝突在 `requirements-server.txt`
保留 fork 明確配對的 HTTP API runtime pins（`fastapi`、`starlette`、
`uvicorn[standard]`、`python-multipart`），再手動加入 upstream 新增的 server
dependency；parser stack 升級前必須重跑 partial-spool cancellation contract。

#### 衝突在 `LLM/default.py`
保留 `api_key = ''`。如果 upstream 更新 prompt、模型名稱或欄位，手動搬回 fork 版本，但不要恢復 API-key-like placeholder。

#### 衝突在 `assets/BUILD_GUIDE.md`
合併 dependency 說明；確保文件仍列出 fork HTTP API dependencies。

#### 衝突在 `core/server/connection/ws_send.py`
這是已記錄的 protocol 接觸點：fork 會把 worker 的 optional
`error_code`／`error_message` 放進 WebSocket `RecognitionMessage`。先同步 upstream
的 send loop，再保留這兩個 optional 欄位；接著跑 protocol regression 與
`fork_server.http_api.tests.test_ws_send_with_http`，確保 fork 內嵌 loop 同步更新。

#### 衝突在刻意維護的 `core/server/` 接觸點

`schema.py`、`state.py`、`connection/{server_manager,ws_recv,ws_send}.py`、
`worker/{__init__,gpu_boost,pipeline,process_manager,task_handler,worker}.py`、
FunASR prompt builder 與兩份 aligner，都是明確記錄的 fork
接觸點。逐段保留 upstream 行為，再重新套用 protocol error、privacy、deadline、
fairness、WebSocket ingress/backpressure、IPv6 bind、watchdog 與 bounded
lifecycle 最小修改並跑上表 regression。其他未列出的
`core/server/` 衝突仍視為意外 drift，應先追查來源，不可直接擴大 allowlist。

## 3. 每次 merge 後的 health check

把這當作 PR check list：

```bash
# (A) Syntax / Import smoke
python3 -m py_compile $(find fork_server start_server_docker.py -name "*.py")
python3 -c "from fork_server.bootstrap import apply_env_config, create_server"

# (B) 上游檔修改數應保持在已記錄清單內
python scripts/check_upstream_divergence.py --require-base

# (C) Container build
docker build -t capswriter-server:merge-test -f docker/server/Dockerfile .

# (D) ws_send_with_http 仍對齊上游；失敗時再人工 diff/re-port
python -m unittest fork_server.http_api.tests.test_ws_send_with_http -v

# (E) 隔離 smoke test (見下節)
```

步驟 (B) 會拿 `origin/master` 直接對目前工作樹比較，因此 commit、staged 與
unstaged 的 tracked edit 都不能繞過檢查。預期首行為：

```text
Upstream divergence guard passed: 59 upstream-tracked file(s) changed
```

後續 59 條路徑必須與本文件第 1 節的十個群組完全一致；若出現第 60 條，先查
來源與能否移入 fork-owned path，不可只為了讓 CI 綠燈就擴大 allowlist。

## 4. 隔離 smoke test (建議流程)

每次 merge 後跑這個矩陣，不影響 production：

```bash
mkdir -p /tmp/cw-merge-test
export CAPSWRITER_HTTP_API_KEY="$(< /secure/path/capswriter-merge-test.key)"
export CAPSWRITER_HOT_SERVER_PATH=/path/to/CapsWriter-Offline/hot-server.txt
cat > /tmp/cw-merge-test/docker-compose.yml <<'EOF'
services:
  cwmerge-server:
    image: capswriter-server:merge-test
    container_name: cwmerge-server
    restart: "no"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      CAPSWRITER_MODEL_TYPE: qwen_asr
      CAPSWRITER_HTTP_API_ENABLE: "true"
      CAPSWRITER_HTTP_API_BIND: 0.0.0.0
      CAPSWRITER_HTTP_API_KEY: ${CAPSWRITER_HTTP_API_KEY:?set_on_host}
      CAPSWRITER_HTTP_API_PORT: 16017
      CAPSWRITER_SERVER_PORT: 16016
      CAPSWRITER_LOG_LEVEL: INFO
    ports:
      - "127.0.0.1:16016:16016"
      - "127.0.0.1:16017:16017"
    volumes:
      - cwmerge-models:/app/models
      - type: bind
        source: ${CAPSWRITER_HOT_SERVER_PATH:?set_on_host}
        target: /app/hot-server.txt
        read_only: true
        bind:
          create_host_path: false
volumes:
  cwmerge-models:
EOF

docker compose -p cwmerge -f /tmp/cw-merge-test/docker-compose.yml up -d
sleep 90   # 模型載入

curl http://localhost:16017/health
curl -H "Authorization: Bearer ${CAPSWRITER_HTTP_API_KEY}" http://localhost:16017/v1/models

# 用任意音檔測 5 種 format
for fmt in json text srt vtt verbose_json; do
  curl -X POST http://localhost:16017/v1/audio/transcriptions \
    -H "Authorization: Bearer ${CAPSWRITER_HTTP_API_KEY}" \
    -F file=@some_audio.wav -F model=whisper-1 \
    -F response_format=$fmt -w "[$fmt %{http_code}]\n"
done

# Production 不能受影響
docker ps --filter "name=capswriter-offline" --format '{{.Names}} {{.Status}}'

# 清理
docker compose -p cwmerge -f /tmp/cw-merge-test/docker-compose.yml down -v
docker rmi capswriter-server:merge-test
rm -rf /tmp/cw-merge-test
```

## 5. 沒事先做這些 = 痛苦

1. **不要直接在 production container 上 `docker compose pull && up`**。先在隔離專案測過。
2. **不要 `git merge origin/master` 進 master**。在 feature branch 上 merge 與測試，OK 後再 fast-forward master。
3. **不要忽略 ws_send_with_http re-port**。上游若改 ws_send 邏輯而 fork 沒同步，HTTP 任務的 result 可能會誤走 WebSocket 定向派發（合成 socket 找不到而被丟棄）或漏掉；HTTP unit test 的 source guard 失敗時先 re-port，再跑完整 gate。

## 6. Image 重 build & 發布

合入 master 後：

- [`.github/workflows/publish-server-image.yml`](../.github/workflows/publish-server-image.yml) 發布 `capswriter-offline-server`
- [`.github/workflows/publish-web-image.yml`](../.github/workflows/publish-web-image.yml) 發布 `capswriter-offline-web`

等 CI / publish workflows 綠燈後，再通知使用者拉新 image。

## 7. 萬一被卡死的回滾

每次 reset 都有 safety tag：

```bash
git tag | grep fork-pre-
# 例: fork-pre-reset-20260525-1411

# 緊急回滾
git reset --hard fork-pre-reset-20260525-1411
git push --force fork master   # 只在自己控制的 fork 上才能 force-push
```

當前 safety tag 與雙軌 ref 請見[繁體中文版本政策](zh-TW/versioning.md)或
[English versioning policy](en/versioning.md)。
