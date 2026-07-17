# 桌面跨平臺與原生快捷鍵

> [文件首頁](README.md) · 繁體中文 · [English](../en/desktop-portability.md)

CapsWriter v2 在保留上游 Windows 桌面行為的同時，加入有明確邊界的 Linux
桌面路徑。這些是不同的支援聲明：Linux server/container 與可攜 CLI 不需要
圖形工作階段，但原生桌面快捷鍵一定要有受支援的 display server。

## 支援路徑

| 路徑 | 支援程度 | 重要邊界 |
|---|---|---|
| Windows 桌面 executable | 支援；保留上游 tray 與快捷鍵行為 | Windows CI 會 build、搬移、解壓並 import-smoke 可攜 artifact；硬體行為仍需真實 release test |
| Windows 或 Linux CLI | Python 3.10、3.12 支援 | 沒有全域快捷鍵、tray、麥克風擷取或自動文字注入 |
| Linux X11 桌面快捷鍵 | 有限制地支援 | 可以監聽，但不能選擇性阻擋單一按鍵 |
| Linux Wayland 桌面快捷鍵 | 不支援 | XWayland 無法保證 system-wide 擷取 |
| Linux headless server | 由 container/source server 路徑支援 | 沒有桌面快捷鍵或 tray |

Portability CI matrix 會在 `ubuntu-24.04`、`windows-2022` 與 Python 3.10、
3.12 上執行 dependency-light compile、打包 source guard、桌面 backend
policy test，以及完整的 standard-library CLI verifier。另有獨立的
`windows-package` job，在 `windows-2022`／Python 3.12 只從完整 hash lock 安裝，
build 兩個 executable，把 distribution 搬離 checkout 後經 ZIP 壓縮／解壓，拒絕
reparse point，透過兩個 EXE 執行 bounded import self-check，最後上傳實際測過的
ZIP。這些 job 都不代表已測過音訊硬體、真實 display server、Windows tray、
model load、known-audio transcription 或 hardware accelerator。

## Windows 打包與 HTTP API

[`build.spec`](../../build.spec) 仍會產生 `start_server.exe` 與
`start_client.exe`。Server executable 改由
[`start_server_universal.py`](../../start_server_universal.py) 分析，而不是
Docker entrypoint。Universal entrypoint 會：

- 保留 `config_server.py` 的 `enable_tray`、model、WebSocket address 與
  accelerator 等設定；
- 讓 OpenAI-compatible HTTP API 預設維持關閉；
- 驗證並套用 `CAPSWRITER_HTTP_API_*`，以及明確選用的
  `CAPSWRITER_SERVER_ADDR` WebSocket bind override；
- 把 fork server 與動態載入的 FastAPI／Starlette／Uvicorn／Pydantic module
  收入 server analysis，不把它們加入桌面 client analysis；以及
- 保留上游的 `freeze_support()` 與 executable 名稱。

請使用可丟棄的 Windows Python 3.12 x86-64 環境。正式 build 只能安裝已 commit
的 hash lock；`srt` source distribution 是 wheel-only policy 的唯一例外：

```powershell
py -3.12 -m venv "$env:TEMP\capswriter-build"
& "$env:TEMP\capswriter-build\Scripts\python.exe" -m pip install `
  --require-hashes `
  --only-binary=:all: `
  --no-deps `
  --requirement requirements-windows-build-bootstrap.lock
& "$env:TEMP\capswriter-build\Scripts\python.exe" -m pip install `
  --require-hashes `
  --only-binary=:all: `
  --no-binary=srt `
  --no-build-isolation `
  --requirement requirements-windows-build.lock
& "$env:TEMP\capswriter-build\Scripts\python.exe" -m PyInstaller --clean --noconfirm build.spec
```

小型 bootstrap lock 會先固定 `pip` 與 `setuptools`，再以停用 isolation 的方式
build `srt` source distribution；runner 預裝的 toolchain 不屬於 release contract。

唯一有效的 output root 是 `dist/CapsWriter-Offline/`。其中包含 `core/`、`LLM/`、
`assets/`、`docs/` 的真實副本，以及真實且初始為空的 `models/`、`logs/`；不含
指回 source tree 的 junction。本機 model、log、cache、secret、archive 與非
Windows shared library 會被排除。必要檔案或必要 collection dependency 缺少時，
build 會直接失敗。

壓縮後應在 checkout 外的不同目錄解壓，再執行：

```powershell
.\start_server.exe --artifact-self-check
.\start_client.exe --artifact-self-check
```

兩者都必須以 zero exit code 輸出 `CAPSWRITER_ARTIFACT_SELF_CHECK=`，且 report
status 為 `ok`。Self-check 只驗證 layout 與 server/client import，不會 bind socket、
開啟 device、建立 hook／tray、啟動 FFmpeg 或載入 model。完整 artifact contract、
CI sequence 與 lock regeneration command 請見雙語
[正式打包指南](../../assets/BUILD_GUIDE.md)。

沒有 HTTP 環境變數時，package 會沿用相同桌面預設值：

```powershell
& .\dist\CapsWriter-Offline\start_server.exe
```

若要啟用 loopback HTTP endpoint 且不改變 tray 設定：

```powershell
$env:CAPSWRITER_HTTP_API_ENABLE = "true"
$env:CAPSWRITER_SERVER_ADDR = "127.0.0.1"
$env:CAPSWRITER_HTTP_API_BIND = "127.0.0.1"
& .\dist\CapsWriter-Offline\start_server.exe
```

若沒有 `CAPSWRITER_SERVER_ADDR`，WebSocket listener 會刻意保留上游
`0.0.0.0` 預設值。只供單機使用時應明確設定此值。

Container 與 env-driven server 部署請使用
[`start_server_docker.py`](../../start_server_docker.py)。該入口刻意套用 headless
預設值，不是 Windows 桌面打包入口。

## 桌面 WebSocket 安全上限

Source client 與 packaged EXE 會把麥克風 ingress 限制為 128 個音訊 chunk，另保留
8 個 control-message slot。若 callback 產生資料的速度超過網路傳送速度，
CapsWriter 會取消該次錄音並關閉 WebSocket，不會無上限保留私密音訊，也不會讓
final marker 超前 audio data。Client message 上限為 16 MiB，receive queue 最多
4 個 message，所有 audio send 都維持原始順序。

可用兩個 process environment variable 調整失敗期限：

- `CAPSWRITER_CLIENT_WEBSOCKET_SEND_TIMEOUT`：每次 WebSocket send，預設 30 秒；
- `CAPSWRITER_CLIENT_FILE_RESULT_TIMEOUT`：檔案上傳完畢後等待 final result，預設
  600 秒。

兩者都必須是大於零的有限數字。超時會關閉 connection，作為 protocol-level
cancellation。只有刻意使用極慢 model 時才應提高檔案期限，不應停用期限。

## Linux X11 快捷鍵

桌面 shortcut manager 在 X11 使用 `pynput` callback。必須同時符合：

- CapsWriter process 位於已登入的 X11 session，且有 `DISPLAY`；
- 已安裝 Python client dependencies，包括 `pynput` 與其 X11 dependency；
- process 能連到同一位使用者的 X server；不需要、也不建議用 root 跑 client；
- X server 提供 `pynput` 使用的 RECORD 與 XTest extension。

`pynput` backend 只會在真正需要模擬輸入時 lazy import。因此 pure Wayland
與 headless session 可以先走完文件所述的偵測／拒絕路徑，不會在 client import
階段就崩潰。

鍵盤 press/release 與常見的側鍵 8/9（`x1`／`x2`）可用。側鍵編號取決於硬體與
driver，必須在目標 workstation 實測。

Paste mode 關閉時，Linux 的一般辨識輸出與 LLM streaming output 都使用同一個
`pynput` X11／XTest controller，不會走 `keyboard` 套件需要 root 與
`/dev/input` 的 backend；Windows 則保留原有 `keyboard.write` 行為。文字注入仍
取決於目標 application、keyboard layout 與 IME，必須用實際 editor 與語言驗證。

X11 有一項重要安全限制：`pynput` 可以 grab 整個鍵盤或 pointer，卻不能安全地
只阻擋 CapsWriter 設定的單一按鍵。因此 CapsWriter 絕不啟用 X11-wide grab。
X11 shortcut 即使設定 `suppress=True`，內部仍會以 `suppress=False` 處理，並寫入
warning；其他 application 仍會收到該鍵。建議使用影響較小的 `f12` 並明確設定
`suppress=False`。滑鼠上一頁／下一頁側鍵在開始錄音時，仍可能讓 focused app
執行 navigation。

## Wayland 與 headless 限制

Wayland 刻意阻止一般 client 觀察任意 system-wide input。目前沒有穩定且跨
compositor 的 `pynput` API 能等同 Windows global hook。即使存在 XWayland
`DISPLAY`，它也可能只看得到 X11 application event，不能當成全域覆蓋。
CapsWriter 會偵測 Wayland session 並拒絕啟動原生 global hotkey listener，避免
靜默提供不完整的擷取。

同一套 compositor policy 也可能限制模擬 paste/type 與 clipboard automation。
只證明 native hotkey detection 可用，不代表特定 Wayland compositor 上完整桌面
client output path 可用。

Wayland 或 headless host 請改用[可攜 CLI](cli-client.md)、browser console 或
HTTP API，不要依賴原生 global shortcut。若一定需要 system-wide 桌面快捷鍵，
請登入 X11 session，並在該 workstation 實測選定按鍵。

## 發布證據

Windows package job 會提供 clean hash install、PyInstaller、archive relocation、
no-reparse-point 與 executable import-smoke 證據。對外宣稱其 ZIP 可發布前，仍須在
目標 Windows hardware 補足：

1. 驗證上傳 ZIP digest，並保存 workflow run 與 dependency lock；
2. 不設定 HTTP 變數啟動 `start_server.exe`，確認 tray 與 WebSocket workflow
   保留上游行為；
3. 啟用 loopback HTTP API，驗證 readiness 與一次 transcription；
4. 啟動 `start_client.exe`，驗證設定的 keyboard/mouse hook；
5. 驗證 microphone／file workflow、FFmpeg、選定 model/runtime asset、known audio，
   以及每一個對外宣稱的 CPU／DirectML／GPU profile；
6. 確認 shutdown 會移除兩個 tray icon 與 child process。

Linux X11 證據應記錄 distribution、desktop environment、X server、keyboard
shortcut、使用時的 mouse mapping，以及是否也驗證文字注入。Headless CI 通過
並不能證明任何一項桌面屬性。
