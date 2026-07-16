# CapsWriter fork v2 文件首頁

> 繁體中文 · [English](../en/README.md) · [專案 README](../../readme.md)

這是 active Windows + Linux fork 的文件首頁。請依工作選擇路徑，不要把單一
deployment path 誤當成整個產品。

## 依工作開始

| 我想要…… | 請讀 |
|---|---|
| 在 Windows desktop、Linux desktop、container、Web、CLI、TUI 之間選擇 | [開始使用](getting-started.md) |
| 維運 server 或 browser deployment | [部署](deployment.md) |
| 排查 desktop、container、API 或 client 問題 | [疑難排解](troubleshooting.md) |
| 理解「支援／有限支援／不支援」聲明 | [支援與安全](support-security.md) |
| 檢視變更並移轉到目前 v2 snapshot | [Release notes](release-notes.md) |

## 產品介面

| 介面 | 指南 | 重要邊界 |
|---|---|---|
| Windows desktop 與 package | [桌面可攜性](desktop-portability.md#windows-打包與-http-api) | CI 驗證 source/package contract；真實 Windows release 仍需 build、audio、tray、shortcut evidence |
| Linux desktop | [桌面可攜性](desktop-portability.md#linux-x11-快捷鍵) | X11 支援但沒有 selective suppression；Wayland global hotkey 不支援 |
| Linux `amd64` server 與 Docker | [部署](deployment.md#linux-container-profile)與 [Docker server reference](../docker-server.md) | Container 是主要 headless deployment path；ARM64 沒有 release gate |
| OpenAI 相容 HTTP API | [API 指南](openai-api.md) | 選用檔案轉錄子集，不是完整 OpenAI Audio API |
| Web Console | [部署](deployment.md#web-console-profile)與 [Web reference](../web-console.md) | Browser 麥克風需要 secure context；browser TTS 依賴本機 browser／OS voice |
| 無 GUI CLI | [CLI 指南](cli-client.md) | Standard-library client；沒有 desktop global hotkey 或 tray |
| Textual TUI | [TUI 指南](tui.md) | Core file mode 有 hash lock；麥克風是選用 native stack |

## 維運與 release 文件

| 文件 | 適用對象 |
|---|---|
| [驗證與清理](../verification.md) | Contributor 與 release operator |
| [v1／v2 維護政策](versioning.md) | Maintainer 與規劃升級的使用者 |
| [架構](../architecture.md) | 修改 fork integration point 的 contributor |
| [目前 fork 狀態](../state-of-fork.md) | 需要 implementation／evidence inventory 的 reviewer |
| [上游同步](../upstream-sync-guide.md) | 合併已發布 upstream 變更的 maintainer |
| [上游產品 changelog](../CHANGELOG.md) | 檢視辨識 engine 與 upstream desktop 歷史的使用者 |

## 建議閱讀路徑

- 新 desktop user：**開始使用 → 桌面可攜性 → 疑難排解**。
- 新 server operator：**開始使用 → 部署 → 支援／安全**。
- API／client integrator：**開始使用 → API 指南 → CLI、Web 或 TUI 指南**。
- Release reviewer：**Release notes → 支援／安全 → 驗證 → 目前 fork 狀態**。
- Maintainer：**架構 → 版本政策 → 上游同步 → 驗證**。

文件提到 automated evidence 時，請把它視為該證據的邊界。Compile 或 mock
contract test 並不能證明真實麥克風、display server、GPU、model download 或已
打包 desktop artifact。
