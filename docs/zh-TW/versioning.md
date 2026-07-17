# v1 與 v2 雙軌維護政策

> 繁體中文 · [English](../en/versioning.md)

本 fork 同時維護兩個產品世代。本文的 **fork v1**、**fork v2** 是 fork
自身的世代名稱，並不等於上游的 `v1.0`、`v2.x` tag。

![CapsWriter fork 雙軌維護流程：上游正式變更進入 v2，只有重大或安全修正才人工回移至隔離的 v1 維護分支](../assets/version-tracks.svg)

文字等價說明：上游已合併、已發布的提交進入 v2，並通過 Linux、Windows、
API、TUI、Web 與安全閘門。舊版 v1 絕不合併 v2；只有重大或安全修正可以人工
backport，且必須另行通過舊版容器、API 與模型資產檢查，才可發布 v1 版本。

## 維護軌與權威 ref

| 維護軌 | 權威分支 | 上游血緣 | 變更政策 |
|---|---|---|---|
| fork v1 | `maintenance/v1` | 上游 `v2.5-alpha` 加上 `3419171` 熱詞最佳化的邏輯快照 | 只收重大安全、相容性與模型資產修正 |
| fork v2 | feature PR 合併後為 `master`；開發分支為 `feature/universal-asr-client` | 截至 2026-07-16，包含上游 master `7d7fac3`（v2.6 與之後全部已合併提交） | 積極開發跨平臺產品，並以 merge 同步上游 |
| v1 稽核快照 | `archive/v1-legacy` 與 tag `fork-pre-reset-20260525-1411` | reset 前最後 v1 tree：`b46ca74` | 不可變的復原／稽核點，不直接在此開發 |

上游進行大型 `util/` → `core/` 重構前，v1 與 v2 的 Git 歷史便已分岔。
兩軌之間不支援 merge 或大量 cherry-pick；每個 backport 都必須針對目標架構
小幅人工移植並補測試。

## 支援矩陣

| 能力 | fork v1 | fork v2 |
|---|---|---|
| 原有 Windows 桌面流程 | 保留舊版行為 | 保留上游行為 |
| Linux server 容器 | 維護支援 | 主要部署路徑 |
| Windows 原生 server | 舊版／人工打包 | 通用入口與打包閘門 |
| 可腳本化 CLI | 僅舊版腳本 | 支援 Windows 與 Linux |
| 互動式 TUI | 不回移 | 支援 Windows 與 Linux |
| 瀏覽器工作台 | 不回移 | 支援現代 Windows／Linux 瀏覽器 |
| OpenAI 形式轉錄 API | 舊版相容；只收安全修正 | 測試 `whisper-1` 轉錄契約，不支援能力會明確報錯 |
| 新功能 | 否 | 是 |

「支援」表示文件指定的入口至少有對應自動化閘門。硬體、終端與真實模型的
發布證據會分開列示；絕不以一個 Linux 容器測試冒充 Windows 執行證據。

## v1 backport 規則

每一項 v1 變更都必須符合以下條件：

1. 修正重大／安全問題、恢復模型資產，或維持既有外部契約。
2. 針對 v1 `util/` 架構實作，不整個複製 v2 module。
3. 加入聚焦的 regression test 或可重跑的隔離 smoke test。
4. 除非舊預設不安全，否則不得改變 v1 預設行為。
5. 只能使用 v1 專屬 Git tag 發布。若要發布 image，必須另建並審查 v1 專用
   workflow／tag；目前 v2 tree 並未設定這項 image 自動發布。

功能開發、UI 重設計、依賴遷移與大範圍上游整合都屬於 v2。

## 版本與 image 命名

- v1 release tag：`fork-v1.<minor>.<patch>`
- v2 release tag：`fork-v2.<minor>.<patch>`
- 目前自動發布的 v2 image tag：不可變的 `sha-<full-git-sha>`，以及只對
  當下 `master` tip 做 guarded promotion 的 `latest`
- 現行 workflow 不發布移動式 `v1`／`v2` channel tag
- 此 tree 尚未自動發布 v1 image，且 v1 絕不可重用 v2 的 `latest`

上游 tag 僅代表上游版本，本 fork 不會用相同名稱重建 release tag。

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
只能作為設計參考，不視為正式發布。divergence guard 與 merge 流程請參閱
[上游同步指南](../upstream-sync-guide.md)。
