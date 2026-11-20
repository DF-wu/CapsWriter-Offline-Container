# 發布 fork v2

> [文件首頁](README.md) · 繁體中文 · [English](../en/releasing.md)

在已審閱的 `master` commit 上推送 `fork-vX.Y.Z` tag 即開始發布。
[`release.yml`](../../.github/workflows/release.yml) 會跑完發布 gate、從該 commit
建置 Windows 套件，並組好 **draft** GitHub Release；是否公開仍由人決定。

## Release matrix

| 附件 | 平台 | 適用對象 |
|---|---|---|
| `CapsWriter-Offline-Client-windows-x86_64.zip` | Windows 10/11 x64 | 只裝 Client，連到遠端或 container Server |
| `CapsWriter-Offline-windows-x86_64.zip` | Windows 10/11 x64 | Server 與 Client 在同一台離線電腦 |
| `CapsWriter-Offline-source.zip` | 不限 | tag commit 的 `git archive` |
| `SHA256SUMS` | 不限 | 三個 ZIP 的 checksum |
| Container image（不是 release 附件） | linux/amd64 | `ghcr.io/df-wu/capswriter-offline-server` 與 `-web`，每次推送 `master` 由 image workflow 發布；Linux Server 依 [Compose](../compose.md) 部署 |

刻意不提供 Windows 僅 Server 的 ZIP（完整版已涵蓋），也不提供 Linux tarball
（Linux 交付物就是 image）。

## Workflow 檢查內容

| Job | Runner | Gate |
|---|---|---|
| `resolve` | Ubuntu 24.04 | Tag 符合 `fork-vX.Y.Z[-suffix]`，且 commit 在 `master` 上 |
| `verify` | Ubuntu 24.04 | `scripts/verify_all.py --skip-web`（source、文件、上游差異、Server 與 scripts 測試）以及在隔離 hash-lock 環境執行的 HTTP API 功能性 contract |
| `desktop-tests` | Windows 2022 | 與 `portability.yml` 相同的桌面／打包 unit suites，加上免依賴 CLI 驗證 |
| `windows-package` | Windows 2022 | 以 [`windows-package.yml`](../../.github/workflows/windows-package.yml) 建置 `full` 與 `client`：hash-lock 建置、搬移、目錄與 reparse point 檢查、Client 版排除檢查、每個 EXE 的 `--artifact-self-check`，以及 `start_client.exe --status` smoke |
| `publish-draft` | Ubuntu 24.04 | 下載剛測過的 ZIP、加入 source ZIP、產生並重新驗證 `SHA256SUMS`、查詢 `sha-<commit>` image digest，建立或更新 draft |

只有 `publish-draft` 有 `contents: write`，且永遠不修改已公開的 release；若該 tag
的 release 已公開，job 會直接失敗。

## 步驟

1. 合併 release PR，等 `master` 上的 `ci`、`portability` 與兩個 image workflow 通過。
2. 在 merge commit 上打 tag 並推送：

   ```bash
   git tag -a fork-v2.3.0 -m "fork-v2.3.0" <master-commit>
   git push origin fork-v2.3.0
   ```

3. 等 `release` workflow 完成。要重建既有 tag 的 draft（例如 runner 偶發失敗），
   手動執行 workflow 並填入 `tag`。
4. 開啟 draft release，補上重點與升級說明、附上 `VALIDATION.md`，並核對 draft
   內的 image digest。若 image 顯示「not published yet」，等該 workflow 完成後重跑第 3 步。
5. 公開 draft。

`maintenance/v1` 上的 tag 不會執行此 workflow：v1 的 tag tree 沒有 `release.yml`，
v1 維持只發 source 的路線。

## 本機預演

打包部分可在 Windows 依[打包指南](../../assets/BUILD_GUIDE.md)
重現；設定 `CAPSWRITER_BUILD_VARIANT=client` 即可建置僅 Client 包。
