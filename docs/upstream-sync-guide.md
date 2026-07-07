# 對齊上游指南

> 未來要拉上游更新時的標準操作流程。本 fork 刻意設計成 merge 衝突極小，理想情況 `git merge origin/master` 一行搞定。

## 1. 心智模型

```
upstream (HaujetZhao/CapsWriter-Offline)
        │
        │  fork modifies: .gitignore, readme.md, requirements-server.txt,
        │                 LLM/default.py, assets/BUILD_GUIDE.md, zip_release.py,
        │                 core/client/hotword/hotword_standalone.py,
        │                 core/server/engines/*/inference/audio.py (4 files),
        │                 core/server/worker/gpu_boost.py,
        │                 core/tools/window_detector.py
        │  fork adds:    fork_server/ docker/ client/cli/ client/web/
        │                docs/ scripts/ docker-compose*.yml .env.example
        │                .github/workflows/ requirements-server-docker.txt
        │
        ▼
fork (DF-wu/CapsWriter-Offline-Container) master/feat/*
```

關鍵：fork 修改 upstream-tracked 檔案數 = **17**。其他主要功能都在新增路徑，正常情況不會與上游衝突。

| Divergent file | Fork 保留原因 | Merge 處理 |
|---|---|---|
| `.gitignore` | 排除 Web/verification cache、model download cache、versioned `.so` 與本地工具狀態 | 合併雙方新增規則 |
| `readme.md` | 中文首頁是 fork 視角 | 通常保留 fork 版本，再手動引用 upstream 重要新內容 |
| `requirements-server.txt` | 裸機 server/HTTP API dependency set | 保留 fork HTTP dependencies，並手動納入 upstream 新 dependency |
| `LLM/default.py` | 移除 API-key-like placeholder，降低 secret-scanning 與誤啟用風險 | 保留空 `api_key`；同步 upstream 其他 template 欄位 |
| `assets/BUILD_GUIDE.md` | 打包文件需反映 fork dependency set | 合併 dependency 說明 |
| `zip_release.py` | legacy PyInstaller ZIP packaging 需要 bounded 7-Zip subprocess 與失敗 cleanup | 保留 timeout/cleanup guard；同步 upstream 其他 packaging 規則 |
| `core/client/audio/file_manager.py` | GUI recorder MP3 `ffmpeg` finalize 需要 bounded wait 與 kill cleanup | 保留 `CAPSWRITER_CLIENT_AUDIO_FINISH_TIMEOUT` guard；同步 upstream audio file lifecycle logic |
| `core/client/hotword/hotword_standalone.py` | standalone hotword Ollama chat helper 需要 bounded local HTTP request | 保留 `CAPSWRITER_OLLAMA_CHAT_TIMEOUT` guard；同步 upstream hotword/demo logic |
| `core/client/transcribe/media_tool.py` | GUI file transcription 的 `ffprobe` duration probe 需要 bounded timeout 與 kill cleanup | 保留 `CAPSWRITER_CLIENT_MEDIA_TIMEOUT` guard；同步 upstream media environment/probe logic |
| `core/client/transcribe/file_transcriber.py` | GUI file transcription 的 `ffmpeg` streaming subprocess 需要 bounded stdout read、final wait 與 kill cleanup | 保留 `CAPSWRITER_CLIENT_MEDIA_TIMEOUT` guard；同步 upstream file transcription flow |
| `core/server/engines/{qwen_asr_gguf,force_aligner_gguf,sensevoice_onnx,fun_asr_gguf}/inference/audio.py` | direct engine file-decode `ffmpeg` subprocess 需要 bounded timeout、kill cleanup 與 bounded stderr preview | 保留 `CAPSWRITER_ENGINE_FFMPEG_TIMEOUT` guard；同步 upstream audio loading logic |
| `core/server/worker/gpu_boost.py` | GPU boost/unboost shell commands 需要 bounded timeout，避免自訂管理命令卡住 worker loop | 保留 `CAPSWRITER_GPU_BOOST_TIMEOUT` guard；同步 upstream GPU boost state logic |
| `core/server/worker/process_manager.py` | recognizer worker shutdown 需要 bounded graceful join、terminate wait 與 kill fallback | 保留 `CAPSWRITER_SERVER_WORKER_STOP_TIMEOUT` guard；同步 upstream worker lifecycle logic |
| `core/tools/window_detector.py` | macOS/Linux window detection subprocess 要有 timeout，避免桌面 client output path hang | 保留 timeout guard；同步 upstream window/app detection 規則 |

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

預期結果：低衝突。若衝突，通常只會落在上方 17 個已知 divergent files。

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
| `core/server/schema.py` | `Task` 或 `Result` 新欄位 | `fork_server/http_api/{api.py,task_router.py,ws_send_with_http.py}` | 如新欄位影響 HTTP 路由 → 適配 |
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
保留 fork 需要的 HTTP API runtime dependencies（`fastapi`、`uvicorn[standard]`、`python-multipart`），再手動加入 upstream 新增的 server dependency。

#### 衝突在 `LLM/default.py`
保留 `api_key = ''`。如果 upstream 更新 prompt、模型名稱或欄位，手動搬回 fork 版本，但不要恢復 API-key-like placeholder。

#### 衝突在 `assets/BUILD_GUIDE.md`
合併 dependency 說明；確保文件仍列出 fork HTTP API dependencies。

#### 衝突在 `core/server/connection/ws_send.py`
**這不應該衝突** — fork 沒改這檔。如果衝突，代表 fork 之前曾經被「污染」修改過。檢查：

```bash
git log --oneline origin/master..HEAD -- core/server/connection/ws_send.py
```

如果有結果，那就是污染源。決定是 revert 還是 keep。

#### 衝突在 `core/server/` 任何其他檔
**這代表設計失敗了** — fork 不應該動上游檔案。看 commit history 找污染源，回滾。

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

步驟 (B) 預期只會看到：

```text
  .gitignore
  LLM/default.py
  assets/BUILD_GUIDE.md
  core/client/hotword/hotword_standalone.py
  core/server/engines/force_aligner_gguf/inference/audio.py
  core/server/engines/fun_asr_gguf/inference/audio.py
  core/server/engines/qwen_asr_gguf/inference/audio.py
  core/server/engines/sensevoice_onnx/inference/audio.py
  core/server/worker/gpu_boost.py
  core/tools/window_detector.py
  readme.md
  requirements-server.txt
  zip_release.py
```

## 4. 隔離 smoke test (建議流程)

每次 merge 後跑這個矩陣，不影響 production：

```bash
mkdir -p /tmp/cw-merge-test
cat > /tmp/cw-merge-test/docker-compose.yml <<EOF
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
      CAPSWRITER_HTTP_API_PORT: 16017
      CAPSWRITER_SERVER_PORT: 16016
      CAPSWRITER_LOG_LEVEL: INFO
    ports:
      - "16016:16016"
      - "16017:16017"
    volumes:
      - /home/df/workspace/CapsWriter-Offline/models:/app/models
      - /home/df/workspace/CapsWriter-Offline/hot-server.txt:/app/hot-server.txt:ro
EOF

docker compose -p cwmerge -f /tmp/cw-merge-test/docker-compose.yml up -d
sleep 90   # 模型載入

curl http://localhost:16017/health
curl http://localhost:16017/v1/models

# 用任意音檔測 5 種 format
for fmt in json text srt vtt verbose_json; do
  curl -X POST http://localhost:16017/v1/audio/transcriptions \
    -F file=@some_audio.wav -F response_format=$fmt -w "[$fmt %{http_code}]\n"
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
3. **不要忽略 ws_send_with_http re-port**。上游若改 ws_send 邏輯而 fork 沒同步，HTTP 任務的 result 可能會被 ws 廣播（送錯地方）或漏掉；HTTP unit test 的 source guard 失敗時先 re-port，再跑完整 gate。

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

詳見 [state-of-fork.md](state-of-fork.md) 內當前 safety tag 名稱。
