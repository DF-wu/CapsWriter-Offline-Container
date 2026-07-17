# 疑難排解

> [文件首頁](README.md) · [English](../en/troubleshooting.md) · [部署](deployment.md)

CapsWriter 應從 server 向外診斷。Client error 往往只是 readiness、authentication、
decode 或 model failure 的最後可見症狀。

![有界 OpenAI 相容 request lifecycle：從 pre-body 檢查、admission、decode、共享 ASR inference 到 format 與 cleanup](../assets/openai-api-lifecycle.svg)

文字等價說明：讀取 body 前先驗證 authentication 與 declared size；bounded
admission 後才進入 multipart spooling 與 capped ffmpeg decode，接著送到 shared
ASR worker、response formatter 與 cleanup。Queue overflow、invalid input、timeout、
cancellation 都走明確 error／cleanup path，不會靜默繼續。

## 診斷順序

請依序檢查，並停在第一個失敗點：

1. **Process/container：**預期 process 是否執行？是否使用目前 source／image／config？
2. **Transport：**client 是否能到達 configured host／port？
3. **Health：**HTTP app 是否回應 `/health`？
4. **Readiness：**`/ready` 是否回報 router、recognizer child、dependency、ffmpeg
   ready？
5. **Models/auth：**同一 Bearer token 呼叫 `/v1/models` 是否成功？
6. **Known audio：**小型已知 file 能否以 `text` 轉錄？
7. **Requested feature：**最後才加入 verbose JSON、字幕、prompt、browser
   microphone、TUI recording 或 batch processing。

無 GUI CLI 可在沒有 browser state 的情況完成前五個 HTTP check：

```bash
export CAPSWRITER_API_BASE=http://127.0.0.1:6017
export CAPSWRITER_HTTP_API_KEY_FILE=/path/to/capswriter-http.key
python client/cli/capswriter_cli.py health
python client/cli/capswriter_cli.py ready
python client/cli/capswriter_cli.py models
```

## Desktop 問題

| 症狀 | 可能邊界 | 處理 |
|---|---|---|
| Windows source 正常，packaged executable 失敗 | Hidden import／native library／package layout | 從 clean environment rebuild，保留 build log，驗證兩個 packaged executable，並比對 exact dependency／PyInstaller version |
| Windows tray／shortcut 無反應 | Desktop permission／session／device runtime | 確認一般 interactive session、configured shortcut、audio permission／device，以及其他 app 未占用該 key |
| Linux shortcut unavailable | 不是 X11 session 或 listener init 失敗 | 檢查 `XDG_SESSION_TYPE=x11`、`DISPLAY`、X11 dependencies 與 startup diagnostics |
| Linux `suppress=True` 無效果 | 預期 X11 safety policy | 不可啟用 whole-keyboard grab；改用不需要 suppression 的 shortcut |
| Wayland／headless 回報 unsupported hotkey | 刻意 support boundary | Desktop hotkey 改用 X11，或使用不需 global capture 的 CLI／TUI／Web／file workflow |
| File transcription 正常、microphone 失敗 | Device／permission／native dependency | 獨立驗證 OS input device 與 client audio stack；server readiness 不能證明麥克風 |

Exact claim 請見[桌面可攜性](desktop-portability.md)。

## Container 與 model 問題

### Container 長時間維持 `starting`

第一次 bootstrap 可能在較長 health-check start period 下載多個 asset。請先查看，
不要反覆 recreate：

```bash
docker compose ps
docker compose logs --tail=300 capswriter-server
```

尋找 bounded download error、archive validation error、missing hotword mount、
backend probe failure、model load failure 或 recognizer child exit。

若多個 container 共用 model storage，其中一個可能正持有
`/app/models/.capswriter-bootstrap.lock`；其他 instance 最多等待
`CAPSWRITER_MODEL_BOOTSTRAP_LOCK_TIMEOUT`（預設 1800 秒），取得後會重查
readiness。Timeout、symlink／hardlink／ownership／mode rejection 通常代表第一個
bootstrap 卡住或 storage 不符合安全 lock contract。先停止多餘 writer、確認
filesystem 在 container 間支援 coherent POSIX `flock(2)`，並修正 bind ownership；
不要直接刪除一個仍由 live process 使用的 lock file。完全 warm 的正常啟動不會
建立或寫入 lock。

### GPU 未使用

```bash
docker compose config
docker compose logs --tail=300 capswriter-server
```

確認 host driver／toolkit、requested GPU count、visible device、selected model
backend。`CAPSWRITER_INFERENCE_HARDWARE=auto` 在 configured GPU probe 失敗後會
關閉所有 CUDA／Vulkan flag、準備 CPU runtime，再要求 CPU probe 通過；看到這個
完整 fallback 可能是正確行為。若第二次 CPU probe 也失敗，container 會拒絕啟動。
調整 GPU flag 前先證明 CPU transcription 可用。

`probe_backend.py` 會建構完整 ASR engine。不可在 active server 內用
`docker compose exec` 執行，否則可能同時載入第二份 model，耗盡 RAM／VRAM。
若 log 不足且確實需要 deep probe，請安排 maintenance window，讓 service 在
one-shot container 執行期間保持停止（probe 失敗也仍要執行最後的 `start`）：

```bash
docker compose stop capswriter-server
docker compose run --rm --no-deps --entrypoint python capswriter-server \
  docker/server/probe_backend.py
docker compose start capswriter-server
```

Intel／AMD iGPU 另請確認 override 與實際 host GID：

```bash
stat -c '%n %g' /dev/dri/renderD* /dev/dri/card*
docker compose -f docker-compose.yml -f docker-compose.igpu.yml config
docker compose -f docker-compose.yml -f docker-compose.igpu.yml exec capswriter-server id
```

把 node 的 numeric GID 寫入 `.env` 的 `CAPSWRITER_DRI_RENDER_GID`／
`CAPSWRITER_DRI_VIDEO_GID` 後 recreate。Image 有 Mesa Vulkan ICD，但沒有
`/dev/dri`、host driver 不可用或 supplementary group 錯誤仍會讓 GPU probe 失敗。

CPU-only host：

```dotenv
CAPSWRITER_INFERENCE_HARDWARE=cpu
```

只使用 `docker-compose.yml`，不要加入 `docker-compose.gpu.yml` 或
`docker-compose.igpu.yml`。Device exposure／reservation 是 Docker admission
階段的決定；host runtime 或 device node 不可用時，container 內的 fallback
尚未開始，無法修復這類失敗。

### Readiness 顯示 ffmpeg 或 model unavailable

- Source deployment 確認已安裝 ffmpeg；image 則確認 binary 存在。
- 預設 `capswriter-server-models` named volume 在 bootstrap 期間必須可寫；若加入
  `docker-compose.models-bind.yml`，則 `./models` 的整個 directory（不只是既有
  model files）必須可由 image `appuser` 寫入。
- 多個 container 共用 model storage 時，確認 filesystem 提供 coherent POSIX
  advisory lock，重試前先查 `CAPSWRITER_MODEL_BOOTSTRAP_LOCK_TIMEOUT` 錯誤。
- 確認 `CAPSWRITER_MODEL_TYPE` 與 tuning value 合法；startup 會拒絕 malformed
  value，不會靜默改用其他 model。
- 檢查 recognizer-child log，不要只重試 `/health`。

## HTTP status 與 contract 問題

| Status／症狀 | 意義 | 下一步 |
|---|---|---|
| Connection refused | Client host／port 沒有可到達 service | Publish／bind address、container state、firewall、browser-visible hostname |
| `400` | Invalid multipart field、model／format／value、unsupported option，或 audio 無法 decode／太短 | 從一個小 file、`model=whisper-1`、`response_format=text`、無 optional field 開始 |
| `401` | Bearer token 缺失或錯誤 | 使用相同 trimmed token／key file；不可把 credential 放在 URL |
| `413` | Declared／raw／file upload 或 decoded-duration limit 超出 | 讀 `/ready` limits；縮短／切割 audio，或有意識地提高 bounded server limit |
| `429` | Active work 加 bounded pending queue 已滿 | 遵守 `Retry-After`、降低 concurrency，或依量測調整 capacity |
| `500` 並提到 decoder／internal recognition | ffmpeg unavailable／failure 或 recognizer error | Server log、ffmpeg path、model child、bounded error preview |
| `501` translations | 刻意 unsupported endpoint | 使用 transcription；未實作本機 translation |
| `504` | End-to-end task deadline 到期 | Queue depth、decode／inference time、hardware，再看 bounded timeout config |
| JSON endpoint 回 invalid JSON | 錯誤 upstream／proxy／path 或 incompatible old server | 看 status／content type 並確認 API root；不可把 proxy HTML 當 transcript parse |
| Redirect response | Client 刻意不帶 credential 跟隨 redirect | 明確設定 final trusted API URL |

HTTP API 會拒絕 unsupported capability，不會靜默忽略。提出 server bug 前，先確認
[API compatibility surface](openai-api.md)。

## Web Console 問題

| 症狀 | 處理 |
|---|---|
| CORS error | 將 exact browser origin 加入 `CAPSWRITER_HTTP_API_CORS_ORIGINS`；`localhost` 與 `127.0.0.1` 是不同 origin |
| Web health 正常但 API diagnostics 失敗 | `/config.js` API root 由 browser resolve；設定 browser 可到達 URL，檢查 API publish／firewall |
| Browser microphone denied | 使用 loopback 或 HTTPS，檢查 browser／OS permission，停止其他 exclusive audio user |
| Upload 前就拒絕 transcription | `/ready.config.max_upload_mb` preflight 發現 file 必定超過 server limit |
| TTS 無 voice／聲音 | Browser Web Speech 依賴本機 browser／OS voice 與 user-gesture／audio policy |
| Default API key 未出現 | 除非同時設定 key 與 `CAPSWRITER_WEB_ALLOW_PUBLIC_API_KEY=true`，否則是預期行為；UI 內輸入較安全 |
| Cancel／換檔後出現舊結果 | 記錄 version 與 reproducer；目前 client 有 stale-result guard，這可能是 regression |

只有 unit／build 通過且已安裝 `agent-browser` 時才跑 `npm run browser-smoke`。
它使用 mock API；通過不證明真實 model 或 microphone。

## CLI 問題

- `--base-url` 必須是 absolute HTTP(S) root，可帶尾端 `/v1`；credential、query、
  fragment、其他 scheme 都會被拒絕。
- 長期使用優先 `--key-file` 或 `CAPSWRITER_HTTP_API_KEY_FILE`。
- `--timeout`、`--max-response-mb` 有上限；只有量測證明 legitimate long
  request 時才提高。
- Batch output 會在 request 前拒絕 portable-name collision。請 rename source
  或使用不同 output directory，不可繞過 guard。
- Local `speak` 在 Windows 需要 PowerShell System.Speech，在 Linux 需要支援的
  local speech command；它不呼叫 CapsWriter server。

## TUI 問題

| 症狀 | 處理 |
|---|---|
| Verifier 回 dependency mismatch | 重建 venv，以 `--require-hashes --only-binary=:all:` 安裝 `requirements-tui.lock` |
| **僅檔案模式** | Core operation 正常；選用 `sounddevice`／PortAudio／device stack unavailable |
| F5 health 正常、model degraded | 逐行讀 diagnostic；readiness／model state 未理解前不要轉錄 |
| Audio path rejected | 使用 TUI process 可見的既有 file；container path 與 host path 不同 |
| Save rejected | Destination 必須不同於 source audio，且在 private recording directory 外 |
| Terminal layout 擁擠 | 盡量使用至少 100×30 cells；更窄時會切成 stacked layout |

Key、bounds、privacy 與 screenshot provenance 請見 [TUI 指南](tui.md)。

## 安全蒐集 evidence

請記錄：

- source commit 或 immutable image digest；
- OS／session（`Windows`、`X11`、`Wayland`、headless）、Python version、client；
- 與 failure 有關、已 sanitize 的 `.env`／Compose effective values；
- 移除 secret 後的 `/health`、`/ready` response；
- exact command、status code、response format、bounded log excerpt；
- model type、CPU／GPU selection，以及小型 known file 是否成功。

**不可**公開 API key、key-file content、private transcript／prompt、recording、完整
browser localStorage、signed URL 或 unrestricted environment dump。若 maintainer
需要 minimal redacted reproducer，請私下保留 original。

疑似 vulnerability 請依[支援與安全](support-security.md)的 private-reporting
guidance，不要公開 exploit report。
