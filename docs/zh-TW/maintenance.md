# 舊版 v1 維護、版本與支援政策

[English version](../en/maintenance.md)

## 狀態與分支模型

本儲存庫為尚未能遷移至目前 v2 路線的部署，保留一條獨立的相容性維護線。

- 開發分支：`maintenance/v1`
- Pull request 基底：`archive/v1-legacy`
- 開始維護時的快照：`b46ca74`
- 安全標籤：`fork-pre-reset-20260525-1411`

「v1」是此 fork 的舊版發行路線名稱。該快照實際包含 upstream 2.5-alpha 時期的程式碼，以及 fork 的 Linux server、容器與 HTTP API 擴充，因此內部 `__version__` 仍為 `2.5-alpha`。v1 維護 PR 不可改以 `master` 為基底，也不可把內部版本字串誤解為分支名稱。

## 維護範圍

可接受的變更刻意維持狹窄：

- 安全性與隱私修正；
- 當機、關閉、資源洩漏與協定正確性修正；
- 為維持既有受支援路徑運作所必需的相依套件相容性修正；
- 迴歸測試、CI 維護與文件勘誤。

新的產品介面、模型家族遷移、大型重構，以及為 v2 開發的功能，應留在 v2。每一項 backport 都應能獨立審查，並保留既有 WebSocket wire format 與 Windows 桌面行為。

目前不承諾停止維護日期或回應時間 SLA；只要此路線仍可合理維護，修正將以 best-effort 方式提供。

## 支援矩陣

| 路徑 | 維護狀態 | 自動化涵蓋範圍 | 重要限制 |
| --- | --- | --- | --- |
| Windows 桌面 client | 保留相容性 | Windows 上以 Python 3.10、3.12 執行語法檢查與低相依協定測試 | 全域快捷鍵、系統匣、麥克風、剪貼簿與 PyInstaller 產物，發行前仍須在真實 Windows 主機驗證。 |
| Linux Docker server | 舊版主要 server 路徑 | Ubuntu 語法、協定／server 單元測試、Compose 驗證及 entrypoint shell 驗證；image runtime 為 Ubuntu 22.04／Python 3.10 | 輕量 PR gate 不會下載模型，也不會測 GPU provider 或真實模型推論。 |
| Linux 裸機 server | Best effort | 同一組 Python 3.10／3.12 低相依 server 測試 | FFmpeg、模型檔、原生函式庫與服務監控由部署者負責。 |
| macOS client/server | 未列入發行驗證 | 語法可能可編譯，但沒有 macOS CI job | 本路線不維護其權限、輸入與音訊整合。 |
| OpenAI-compatible HTTP API | 選用的舊版 server 功能 | 上傳限制與回應格式 helper 測試 | 僅支援轉錄；不實作翻譯，模型／token 時間戳亦只是近似值，並非完整 OpenAI 服務。 |

CI 通過只表示可攜式協定與 server 控制路徑通過；不代表音訊硬體、桌面整合、GPU backend、模型準確度或可發行執行檔已獲驗證。

「Windows desktop Client」只指此 tree 保留的 upstream-era `start_client.py`
source workflow；此維護線不含 v2 Web Console、無 GUI CLI、Textual TUI 或
universal Windows package。Release notes 必須分開列出 Server／API／container
deliverable 與 Windows Client source／binary 狀態。沒有 attached artifact 與真實
Windows qualification 時，不可宣稱有 Windows download。

目前 v1 release 為 source-only。Compose 從 checkout build 本機 v1 image；公開
`ghcr.io/df-wu/capswriter-offline-server:latest` 是 v2，不可宣稱為 v1 image。

## 輸入與安全基線

維護線對合法 client 保留既有協定，並套用以下防禦性限制：

- WebSocket JSON 訊息預設上限為 8 MiB（`CAPSWRITER_WS_MAX_MESSAGE_MB`）。
- 單一解碼後音訊 chunk 上限為 4 MiB；官方 client 的 60 秒 float32／16 kHz／mono chunk 為 3,840,000 bytes，仍可正常使用。
- `seg_duration` 必須大於零且不超過 300 秒；`seg_overlap` 必須介於 0 至 30 秒，換算後的 byte geometry 必須落在非零的 float32 sample 邊界。
- task ID 最長 128 字元，context 最長 8,192 字元。
- 同一連線必須先 finalize 目前 task，才可更換 task ID 或 source。
- 辨識狀態會依連線分區，因此不同 WebSocket session 使用相同 client task ID 時，不會互相合併逐字稿。
- HTTP 上傳採有界分塊讀取，一旦超過 `CAPSWRITER_HTTP_API_MAX_UPLOAD_MB` 即停止。
- Bearer token 採 constant-time 比對。除非已設定驗證及可信任的 reverse proxy／TLS 邊界，HTTP API 應只綁定 loopback。
- Transcription middleware 會先檢查 Bearer authentication 並拒絕非 multipart body，之後才讓 Starlette 啟動 form parser。

這些是應用程式層防護，不能取代對外服務的安全邊界。請勿把未驗證的 WebSocket 服務直接暴露於不可信任網路。

## 驗證方式

PR gate 會在 Ubuntu 與 Windows 的 Python 3.10 與 Python 3.12 執行以下可攜式檢查：

```bash
python -m pip install -r requirements-maintenance.txt
python -m unittest discover -s tests -p "test_*.py" -v
python -m compileall -q config_client.py config_server.py core_client.py core_server.py start_client.py start_server.py util docker/server
```

Linux／Python 3.10 lane 另外驗證 `docker/server/entrypoint.sh` 與 Compose 設定。發行驗證還應包含一次性容器 build、具真實模型的中英文音訊 smoke test，以及真實 Windows 桌面 smoke test；發行紀錄須註明模型、backend、driver／runtime、測試音訊來源與結果。

## Backport 與發行流程

1. 從 `maintenance/v1` 製作變更，且只與 `archive/v1-legacy` 比較。
2. 每項 backport 保持聚焦，並記錄所有行為變更或新增限制。
3. 在隔離環境執行可攜式 gate；不可使用 production 模型、憑證或可寫入的 production volume。
4. 影響 runtime 的變更，須再執行前述發行驗證。
5. PR base 必須是 `archive/v1-legacy`；不可把整條舊版線合併回 v2。
6. 舊版發行 tag 使用 `fork-v1.<minor>.<patch>`（pre-release 加 `-rc.<n>`），附上
   exact source commit，並分別說明是否交付 Server source、container image、
   Windows Client source 或經驗證的 Windows binary。只有在有意識的發行決策中，
   才變更應用程式內部版本。

## 已知殘餘風險

- Application middleware 會在 multipart parsing 前驗證，但不會限制 chunked transfer encoding 或 multipart overhead 的 raw body。HTTP API 若不只對 loopback 開放，reverse proxy 仍應執行 request-size 與驗證政策。
- 小型壓縮檔可在 FFmpeg 解碼後膨脹成大得多的 PCM；上傳大小限制不等於音訊時長或解碼後記憶體限制。
- 辨識採單一 worker 與 multiprocessing queue，沒有 durable job store，也沒有嚴格的全域 pending-job 配額。
- 原生模型函式庫、GPU provider、桌面 hook 與封裝的各平台相依面，遠大於輕量 CI gate 的涵蓋範圍。
- 舊版相依套件會逐漸累積 upstream 終止支援風險；若升級必須大幅更動應用程式，應在 v2 處理。

如需回報弱點，請盡可能採私下管道。公開 issue 不可附上 API key、逐字稿、音訊、模型產物或完整 production log。
