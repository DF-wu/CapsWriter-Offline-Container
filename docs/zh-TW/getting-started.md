# 開始使用

> [文件首頁](README.md) · [English](../en/getting-started.md) · [疑難排解](troubleshooting.md)

CapsWriter fork v2 有多個有效入口。請依實際 machine 與 interaction model 選擇
正確路徑。

![真實 CapsWriter Textual 工作台；畫面上方是 server 診斷，下方可見檔案輸入與轉錄面板](../assets/tui-workbench.svg)

這張 screenshot 由真實 Textual application 產生，不是 mockup。它示範其中一個
portable client surface；Windows desktop app、Web Console 與無 GUI CLI 仍是
不同選項。

## 選擇路徑

| 需求 | 建議路徑 | 不可推論 |
|---|---|---|
| 完整 Windows tray／hotkey desktop | Windows desktop package／source path | CI compile 不等於已簽署、經 hardware 驗證的 release binary |
| Linux desktop dictation | Linux X11 source desktop path | X11 支援不代表 Wayland global hotkey 支援 |
| Shared／headless ASR | Linux container server | HTTP process healthy 不等於 model ready |
| Browser 互動 | Server + Web Console | Browser TTS 是本機 browser／OS 能力，不是 server TTS |
| Script、SSH、batch job | 無 GUI CLI | CLI 不會 global text injection，也沒有 tray |
| 鍵盤優先 terminal workflow | Textual TUI | Core lock 保證 file mode；native microphone 為選用 |
| 既有 OpenAI client code | 選用 HTTP API | 只實作文件列出的 transcription subset |

## 共通先決條件

- Git，以及足以容納 repository、所選 model 與 generated artifact 的 storage。
- Fork portable client／verification surface 使用 Python 3.10–3.12。
- 只有所選 path 使用 audio input/output 時才需要對應裝置。
- Source/server host 若需解碼相關格式，必須有 ffmpeg。
- Container path 需要 Docker Engine + Compose plugin。
- Linux desktop global shortcut 需要已登入的 X11 session。

請 clone 本 fork，而不是在這份文件中使用 upstream remote：

```bash
git clone https://github.com/DF-wu/CapsWriter-Offline-Container.git
cd CapsWriter-Offline-Container
```

不要 commit `.env`、API key、本機錄音、model archive 或 generated release
directory。

## 路徑 A：Windows desktop

Windows desktop path 保留上游 tray、shortcut、recorder 與 file transcription
surface。Fork `build.spec` 會 package universal server entrypoint，因此可加入選用
HTTP API，又不會取代一般 desktop default。

1. 先讀[桌面可攜性](desktop-portability.md#windows-打包與-http-api)。
2. 建立可丟棄的 Windows virtual environment，安裝 client/server requirements
   與選定的 PyInstaller version。
3. 從 repository root build `build.spec`。
4. 在真實 Windows desktop 驗證兩個 packaged executable：launch／exit、tray、
   configured shortcut、microphone record／stop、file transcription、model load，
   以及選用 HTTP health／readiness。
5. 隨 release artifact 記錄 exact Python、dependency、PyInstaller、Windows 與
   hardware version。

Automated portability matrix 會在 `windows-2022` compile 並測試 portable source
contract，但不能取代步驟 4。

## 路徑 B：Linux X11 desktop

Linux desktop shortcut 需要真實 X11 session 與 client dependency stack。啟動前
先確認 environment：

```bash
test "${XDG_SESSION_TYPE:-}" = x11
test -n "${DISPLAY:-}"
CAPSWRITER_SERVER_ADDR=127.0.0.1 python start_server_universal.py
```

在相同已登入 graphical session 另行啟動 desktop client：

```bash
python start_client.py
```

X11 可處理 keyboard 與常見 side-button callback，但 CapsWriter 會強制關閉
shortcut suppression，因為 `pynput` 無法安全地只阻擋單一 configured key。
Wayland 與 headless session 會回報 unsupported backend，不會建構無法可靠工作的
listener。

修改 shortcut behavior 前，請讀 [Linux X11 快捷鍵](desktop-portability.md#linux-x11-快捷鍵)。

## 路徑 C： Linux container server

準備本機 configuration 與 hotword mount：

```bash
cp .env.example .env
cp hot-server.example.txt hot-server.txt
docker compose up -d capswriter-server
docker compose ps
docker compose logs -f capswriter-server
```

第一次 model bootstrap 可能需要時間，因此 Compose health check 有較長 start
period。Model 預設保留在 `capswriter-server-models` named volume；server log 使用
`capswriter-server-logs` volume。若確實需要在 host 直接管理 `./models`，才加入
`docker-compose.models-bind.yml`，並先確保 image 內 `appuser` 對該 directory 有
完整寫入權限：

```bash
docker compose -f docker-compose.yml -f docker-compose.models-bind.yml up -d capswriter-server
```

CPU-only 啟動：

```bash
CAPSWRITER_INFERENCE_HARDWARE=cpu \
docker compose up -d --force-recreate capswriter-server
```

Base Compose 刻意不保留 GPU，因此沒有 NVIDIA container runtime 的 host 也能
安全啟動。若要暴露 NVIDIA device，請明確加入 GPU override：

```bash
CAPSWRITER_GPU_DEVICE_COUNT=all \
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  up -d --force-recreate capswriter-server
```

Linux Intel／AMD iGPU 使用不同的明確 override。先以
`stat -c '%n %g' /dev/dri/renderD* /dev/dri/card*` 查出 host numeric GID，寫入
`.env` 的 `CAPSWRITER_DRI_RENDER_GID`／`CAPSWRITER_DRI_VIDEO_GID`，再啟動：

```bash
docker compose -f docker-compose.yml -f docker-compose.igpu.yml up -d --force-recreate capswriter-server
```

若任何 configured CUDA／Vulkan startup probe 失敗，entrypoint 會關閉所有 GPU
backend、準備 CPU runtime 並要求第二次 CPU probe 通過，才啟動 server。

WebSocket port 預設只發布到 host loopback。選用 HTTP mapping、認證、Web
Console、persistence 與 upgrade 請接著讀
[部署指南](deployment.md#linux-container-profile)。

## 連接 client

Web Console、CLI、TUI 與 OpenAI SDK 都需要 HTTP API。請明確啟用、設定 key、
取消 Compose `ports:` mapping 的註解、重建 server，並要求兩個 endpoint 都成功：

```bash
curl http://127.0.0.1:6017/health
curl http://127.0.0.1:6017/ready
```

接著選擇 client：

- [OpenAI 相容 API 與 SDK](openai-api.md)
- [Web Console](../web-console.md)
- [無 GUI CLI](cli-client.md)
- [Textual TUI](tui.md)

送出 audio 前應使用 `/ready`，不能只看 `/health`。Readiness 會回報 model
worker、router、ffmpeg 與 active limits，但不暴露 API key。

## 第一次成功轉錄

請使用內容與語言已知、體積小的 audio file。CLI 範例：

```bash
export CAPSWRITER_API_BASE=http://127.0.0.1:6017
export CAPSWRITER_HTTP_API_KEY=replace-with-a-long-random-token
python client/cli/capswriter_cli.py ready
python client/cli/capswriter_cli.py transcribe /path/to/known.wav --format text
```

長期 service 應優先使用 mode `0600` key file，不要把 token 放在 command line
或長期 shell history。

## 驗證 checkout

Dependency-light repository gate：

```bash
python scripts/verify_all.py
```

文件與 cleanup：

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/check_docs.py
python scripts/clean.py
python scripts/clean.py --check
```

Windows package build／relocation／import gate 會在 portability CI 獨立執行。
Hardware、model、display、browser-device 與真實 desktop behavior 仍是不同 release
evidence。詳見[驗證](../verification.md)與[支援／安全](support-security.md)。

## 下一步

- Production 或 LAN operation：[部署](deployment.md)
- Platform truth：[桌面可攜性](desktop-portability.md)
- Failure diagnosis：[疑難排解](troubleshooting.md)
- Security／support boundary：[支援與安全](support-security.md)
- Upgrade／目前變更：[Release notes](release-notes.md)
