# v1 與 v2 雙軌維護政策

> [文件首頁](README.md) · 繁體中文 · [English](../en/versioning.md)

本 fork 同時維護兩個產品世代。本文的 **fork v1**、**fork v2** 是 fork
自身的世代名稱，並不等於上游的 `v1.0`、`v2.x` tag。

上游變更通常先進入 v2。v1 已透過 PR #5 完成**一次性完整同步上游至 `84912d5`**，
包含 `util/` → `core/` 遷移。兩個產品維護軌仍然分開：v1 必須保留自己的 server、
容器、API 契約與發布管道。

## 維護軌與權威 ref

| 維護軌 | 權威分支 | 上游血緣 | 變更政策 |
|---|---|---|---|
| fork v1 | `maintenance/v1` | PR #5 已同步上游至 `84912d5` 並遷移至 `core/` | 一次性完整上游同步已完成；保留 v1 server／容器／API 行為，恢復聚焦維護 |
| fork v2 | `master` | 已包含上游 `84912d5`（PR #4）；`fork-v2.0.0` 由此發布 | 積極開發跨平臺產品；使用短期 branch，並以 merge 同步上游 |
| v1 稽核快照 | `archive/v1-legacy` 與 tag `fork-pre-reset-20260525-1411` | reset 前最後 v1 tree：`b46ca74` | 不可變的復原／稽核點，不直接在此開發 |

上游進行大型 `util/` → `core/` 重構前，v1 與 v2 的 Git 歷史便已分岔。
已完成的 v1 同步在獨立分支導入上游變更，並將 v1 整合移植到 `core/`；
沒有整批匯入 v2 產品。PR #5 合併後，`maintenance/v1` 也使用 `core/` 架構。
例行 backport 必須針對接收分支實際使用的架構移植，並通過該世代的檢查。

## 支援矩陣

| 能力 | fork v1 | fork v2 |
|---|---|---|
| 原有 Windows 桌面流程 | 保留舊版行為 | 保留上游行為 |
| Linux server 容器 | 維護支援 | 主要部署路徑 |
| Windows 原生 server | 舊版／人工打包 | 通用入口與打包閘門 |
| 可腳本化 CLI | 僅舊版腳本 | 支援 Windows 與 Linux |
| 互動式 TUI | 不回移 | 支援 Windows 與 Linux |
| 瀏覽器工作台 | 不回移 | 支援現代 Windows／Linux 瀏覽器 |
| OpenAI 形式轉錄 API | 同步時保留既有 v1 契約 | 測試 `whisper-1` 轉錄契約，不支援能力會明確報錯 |
| 新功能 | 納入已核准一次性同步的上游變更；其餘維持聚焦維護 | 是 |

「支援」表示文件指定的入口至少有對應自動化閘門。硬體、終端與真實模型的
發布證據會分開列示；絕不以一個 Linux 容器測試冒充 Windows 執行證據。

## v1 同步與 backport 規則

已核准的完整同步以 `84912d5` 為目標，可依整合需求遷移架構與依賴，但必須保留
v1 server／容器／API 行為、通過獨立 v1 檢查，並經自己的 PR 整合。此次作業
不代表已授權正式環境部署或版本發布。

除這次完整同步外，每一項 v1 變更都必須符合以下條件：

1. 修正重大／安全問題、恢復模型資產，或維持既有外部契約。
2. 針對當下 v1 架構實作（同步前為 `util/`，完成後為 `core/`），不整個複製
   v2 產品 module。
3. 加入聚焦的 regression test 或可重跑的隔離 smoke test。
4. 除非舊預設不安全，否則不得改變 v1 預設行為。
5. 只能使用 v1 專屬 Git tag 發布。若要發布 image，必須另建並審查 v1 專用
   workflow／tag；目前 v2 tree 並未設定這項 image 自動發布。

持續性的產品功能開發與 UI 重設計屬於 v2；未來若要再對 v1 進行大範圍上游
同步，必須另行明確決定範圍。

## 版本與 image 命名

- v1 release tag：`fork-v1.<minor>.<patch>`
- v2 release tag：`fork-v2.<minor>.<patch>`
- Release candidate 加 SemVer pre-release suffix `-rc.<n>`，並標記為 GitHub
  pre-release；final tag 不會重用 RC tag
- 目前自動發布的 v2 image tag：不可變的 `sha-<full-git-sha>`，以及只對
  當下 `master` tip 做 guarded promotion 的 `latest`
- 現行 workflow 不發布移動式 `v1`／`v2` channel tag
- 此 tree 尚未自動發布 v1 image，且 v1 絕不可重用 v2 的 `latest`

上游 tag 僅代表上游版本，本 fork 不會用相同名稱重建 release tag。

### 目前候選版與發布邊界

最新已發布版為 **`fork-v2.0.0`**（2026-09-30）。**`fork-v2.1.0` 尚在未發布準備**，
不代表已建立 tag 或發布 RC。候選 runtime／套件 metadata 為 `2.1.0`
（`fork_server.__version__`、`pyproject.toml`）；`/health`、`/ready` 新增
`fork_version: "2.1.0"`，既有上游 `version: "2.6"` 不變，上游基底仍是 `84912d5`。

原始碼與 image workflow 的 canonical repo 同為 `DF-wu/CapsWriter-Offline-Container`；
舊名稱 `DF-wu/CapsWriter-Offline` 會重導向同一個 repo。發布 `master` 會觸發 image 自動發布與
guarded `latest` promotion；延後 GitHub Release 並不會延後此副作用。請在
[v2.1.0 發布草稿](../releases/fork-v2.1.0.md)分別記錄完整 merge SHA、相同 SHA 的
CI／套件 workflow、不可變 image tag／digest 及 provenance。版本字串與 mock 測試
不能替代實機驗收或延遲量測。

`myServices` recipe 必須另行 commit／push 至 Portainer 追蹤的 branch。本機
named-volume 模型修復仍待遠端發布，release 準備未變更實際 stack。只有 Portainer
拉到正確 recipe，並確認實際掛載與 readiness 後，才可移除舊 appdata models；
保留父目錄與 `hot-server.txt`。見 [boost／部署注意事項](../nvidia-boost.md)。

## 從 v1 移轉至 v2

請把移轉視為平行部署，不要做就地 Git merge：

1. 備份 `.env`、熱詞、角色檔與本機模型 cache。
2. 以不同 WebSocket／HTTP port 啟動 v2。
3. 執行 `/health`、`/ready`、模型列表，並以已知內容的中英文音訊轉錄。
4. 先把一個 CLI／TUI client 指向 v2，逐一驗證所需 response format。
5. 分批移動 client；在 rollback 期限結束前，讓 v1 保持停止但可復原。

設定名稱與能力差異會列在成對的 release notes。禁止用 v1 Python 原始碼覆蓋 v2。

## 同步上游

只有已合併到上游 `master` 的 commit 才會進入例行 v2 同步。大型未合併 PR
只能作為設計參考，不視為正式發布。目前 v2 同步保留 Python 3.10–3.12 與固定的
llama.cpp b7798 ABI 相容性，不會自動採用上游 Python 3.14 與 b10621 執行環境
假設。divergence guard 與 merge 流程請參閱
[上游同步指南](../upstream-sync-guide.md)。
