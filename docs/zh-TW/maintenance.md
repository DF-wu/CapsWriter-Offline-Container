# v1 上游整合、版本與支援政策

[English version](../en/maintenance.md)

## 狀態與分支模型

v1 是 Windows 桌面 Client 與 Linux／Docker Server 的獨立發行路線。
**接受完整上游更新，包括模型變更與必要重構，以日常可用性及穩定性為優先。**
此政策取代先前只接受重大修正的限制。

- 維護路線：`maintenance/v1`
- 長期比較／發行 PR 基底：`archive/v1-legacy`
- 原始維護快照：`b46ca74`
- 歷史安全標籤：`fork-pre-reset-20260525-1411`
- 本次上游整合：`84912d5`

Python 模組的應用程式版本 `2.6` 描述上游相容版本，不代表 fork 的發行路線；
內部 `2.x` 字串不會讓此分支變成 fork v2。不可將整條 v1 合併到 `master`。

## 範圍與相容性

上游錄音、快捷鍵、轉錄、模型、相依套件及架構改善，都可在安全整合後進入 v1。
保留 Windows source 入口、文件化的 WebSocket 行為、fork HTTP API 與
Linux／容器運作。變更預設值、設定或模型目錄時須提供遷移說明，並為行為變更
補上迴歸驗證。

v2 Web Console、無 GUI CLI、Textual TUI 與 universal Windows package
維持為獨立產品介面。其他桌面平台保留擴充空間，但不是目前 Client 驗收目標。

保留原生 Python 3.10–3.12 相容性，環境安裝以 Python 3.12 為基準；固定 Docker
runtime 仍為 Python 3.10。上游 Python 3.14／uv 遷移會配合平台依賴調整，
不代表此版本已完成 Python 3.14 發行驗證。原生 llama binding 與 `b7798` 成對
固定；採用上游 `b10621` 必須同步遷移 binding／binary 並驗證推論。
詳見 [遷移指南](../v1-upstream-refresh.md)。

## 支援與發行證據

| 路徑 | 目標 | 驗證邊界 |
| --- | --- | --- |
| Windows 桌面 Client | 第一支援平台；`start_client.py` | 可攜式語法／協定檢查無法驗證全域快捷鍵、系統匣、麥克風、剪貼簿、文字注入或 PyInstaller 執行檔。 |
| Linux Docker Server | 主要 Server 部署方式 | 單元測試通過不代表冷啟動模型下載、原生模型載入、CPU／GPU 推論或容器 bootstrap 已成功。 |
| 原生 Server | fork runtime 使用 `start_server_universal.py` | 須在目標主機檢查 FFmpeg、模型、原生函式庫與服務監控。 |
| HTTP API | 選用轉錄子集 | 須驗證驗證機制、限制、取消及真實模型請求；不等於完整 OpenAI API。 |
| 其他桌面平台 | 保留擴充空間 | 不宣稱 macOS／Linux 桌面已完成驗證。 |

目前 v1 發行仍為 source-only。Compose 從 checkout build
`capswriter-offline-v1-local:source`。公開
`ghcr.io/df-wu/capswriter-offline-server:latest` 屬於 v2，不可當作 v1 發行。
只有另外附上經驗證的 Windows artifact，才可宣稱提供 Windows 執行檔。

不承諾終止維護日期或回應 SLA，支援採 best effort。

## Runtime 邊界

更新上游內部架構時，保留 fork 的有界協定驗證、依連線隔離辨識狀態、取消清理，
以及 HTTP 驗證／上傳控制。精確限制以 [HTTP API](../HTTP_API.md) 與目前
runtime 設定為準；壓縮檔大小限制不能取代解碼音訊限制。未驗證的 WebSocket
服務應限定在可信任網路。原生模型函式庫、平台 hook 與加速 provider，須與
可攜式 Python 測試分開驗證。

## 驗證方式

在隔離環境執行完整 Linux CI gate：

```bash
python -m pip install -r requirements-maintenance.txt
python scripts/verify_v1.py
python -m compileall -q config_client.py config_server.py start_client.py start_server.py start_server_universal.py start_server_docker.py core fork_server docker/server
bash -n docker/server/entrypoint.sh
docker compose --env-file .env.example config --quiet
```

Windows CI 分別執行 `tests`、`scripts/tests`、`fork_server/http_api/tests`。
`docker/server/tests` 僅在 Linux 執行：容器 bootstrap 刻意依賴 POSIX
`O_NOFOLLOW`、目錄 file descriptor 與 advisory lock。Ubuntu 仍須通過四套測試。

目前 OS／Python 矩陣以 workflow 為準。測試結果必須對應確切 source revision；
歷史 CI 證據不能代替本次更新驗證。發行前應包含一次性容器 build／bootstrap、
中英文 known-audio 轉錄、CPU／GPU backend 資訊及真實 Windows 桌面 smoke test。
紀錄須包含模型、native runtime／driver、音訊來源與實際結果。

axolotl 測試須使用隔離容器、port、network 與一次性儲存空間；不升級現有服務，
不掛載可寫入的正式資料。僅清除本次測試建立的資源，保留驗證紀錄。

## 整合與發行流程

1. 從 v1 維護路線建立隔離作業分支。
2. 記錄上游 revision，審查刻意保留的整合差異。
3. 啟動前遷移設定值並確認模型目錄；不可用舊設定模組整份覆蓋新版。
4. 執行可攜式與相關 runtime 驗證，註明未測範圍；未取得證據不得宣稱 Windows
   或真實模型驗證完成。
5. 在 fork repository 向適當的 v1 基底開 PR，不向上游或 v2 `master` 開 PR。
6. Tag 使用 `fork-v1.<minor>.<patch>`，pre-release 可加 `-rc.<n>`，並列出
   確切 source commit。Server source、image、Windows source 與 binary 分別說明；
   PR 不代表已發行或部署。

弱點請盡可能私下回報；公開紀錄不可包含 key、逐字稿、音訊、模型產物或私人
正式環境 log。
