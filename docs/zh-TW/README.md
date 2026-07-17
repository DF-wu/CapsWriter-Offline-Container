# CapsWriter fork v2 文件首頁

> 繁體中文 · [English](../en/README.md) · [專案 README](../../readme.md)

CapsWriter 有兩種角色：**ASR Server** 在本機執行模型推論；一個或多個
**Client** 收音／選檔並呈現逐字稿。若這個差異還不清楚，請先讀
[Server 與 Client 分工](server-and-clients.md)。

## 新使用者：依三個步驟開始

1. **認識元件：**[Server 與 Client 分工](server-and-clients.md)。
2. **選擇 Server 執行位置：**[開始使用](getting-started.md)。
3. **選擇 Client：**從下表選 desktop、Web、CLI、TUI 或 SDK。

## Server 文件

| 我需要…… | 請讀 |
|---|---|
| 啟動 Docker、Windows native 或 source Server | [部署](deployment.md) |
| 為 Web／CLI／TUI／SDK 啟用 HTTP `6017` | [OpenAI 相容 API](openai-api.md) |
| 理解 WebSocket `6016`、HTTP `6017` 與 readiness | [Server 與 Client 分工](server-and-clients.md#server-interface) |
| 設定 exposure、authentication、privacy 或 hardware claim | [支援與安全](support-security.md) |
| 排查 model、container、readiness 或 API 問題 | [疑難排解](troubleshooting.md) |

Server 負責 model、Server 端 FFmpeg decode、inference、hotword、queue 與
readiness；不提供 desktop hotkey、browser UI、terminal UI 或 Client 本機 TTS。
Desktop Client 可能自行使用 FFmpeg 做本機 media preparation，但不會執行 ASR
推論。

## Client 文件

| Client | Server 連線 | 指南 | Client 本機功能 |
|---|---|---|---|
| Windows／Linux X11 desktop | WebSocket `6016` | [桌面可攜性](desktop-portability.md) | Tray、hotkey、麥克風、檔案、文字注入 |
| Web Console | HTTP `6017` | [Web Console 指南](web-console.md) | Browser 錄音、history、download、browser／OS TTS |
| 無 GUI CLI | HTTP `6017` | [CLI 指南](cli-client.md) | File／batch automation、atomic output、選用 OS TTS |
| Textual TUI | HTTP `6017` | [TUI 指南](tui.md) | 鍵盤操作、檔案、選用 native 麥克風、儲存 |
| OpenAI SDK／curl | HTTP `6017/v1` | [API 指南](openai-api.md) | Integration 自己的 upload 與結果處理 |

上列 Client 都不含 ASR model。Web／CLI／TUI／SDK 需要選用 HTTP API；desktop
Client 使用 WebSocket，一般不需要 HTTP。

## 依部署情境選擇

| 目標 | Server | Client | 從這裡開始 |
|---|---|---|---|
| 個人 Windows 語音輸入 | Windows packaged／native Server | Windows desktop Client | [Windows desktop 路徑](getting-started.md#路徑-awindows-desktop) |
| Linux desktop 語音輸入 | X11 中的 native source Server | Linux X11 desktop Client | [Linux X11 路徑](getting-started.md#路徑-blinux-x11-desktop) |
| NAS／headless／共享 ASR | Linux `amd64` container | Web、CLI、TUI 或 SDK | [Container 路徑](getting-started.md#路徑-c-linux-container-server) |
| 既有 OpenAI integration | 任何已啟用 HTTP 的 Server | SDK／curl | [API 相容範圍](openai-api.md#相容範圍速覽) |

## 維運與 Release 文件

| 文件 | 適用對象與用途 |
|---|---|
| [Release notes](release-notes.md) | 檢視變更、migration、限制與 release evidence |
| [疑難排解](troubleshooting.md) | 依 Server-first 診斷順序排查問題 |
| [驗證與清理](../verification.md) | Contributor 與 release operator |
| [v1／v2 維護政策](versioning.md) | Maintainer 與規劃升級的使用者 |
| [架構](../architecture.md) | 修改 Server、sidecar 或 routing integration 的 contributor |
| [目前 fork 狀態](../state-of-fork.md) | 查核 implementation 與 evidence inventory 的 reviewer |
| [上游同步](../upstream-sync-guide.md) | 合併 upstream 正式變更的 maintainer |
| [上游產品 changelog](../CHANGELOG.md) | Recognition engine 與 upstream desktop 歷史 |

## 依讀者建議順序

- Desktop user：**Server／Client 分工 → 開始使用 → 桌面可攜性 → 疑難排解**。
- Server operator：**Server／Client 分工 → 部署 → 支援／安全 → 疑難排解**。
- API integrator：**Server／Client 分工 → API 指南 → 選定的 Client 指南**。
- Release reviewer：**Release notes → 支援／安全 → 驗證 → 目前 fork 狀態**。
- Maintainer：**架構 → 版本政策 → 上游同步 → 驗證**。

Automated evidence 有明確邊界。Compile、unit test 或 package import self-check
不能證明目標機器上的真實麥克風、display server、model、GPU、tray 或 global
shortcut。
