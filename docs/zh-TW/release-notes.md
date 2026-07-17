# Release notes 與 changelog

> [文件首頁](README.md) · [English](../en/release-notes.md) · [開始使用](getting-started.md)

## fork-v2.0.0-rc.1 — 跨平台 release candidate

Release candidate 日期：**2026-07-18**。`fork-v2.0.0-rc.1` 是 GitHub
pre-release，不是 final `fork-v2.0.0` support claim。Automated source、API、TUI、
Web、image 與 Windows package gates 已完成；下方 real-device／model qualification
仍須在 stable release 前補齊。

![Fork 維護流程：上游正式變更進入 active v2，只有重大或安全修正才人工回移到隔離 v1](../assets/version-tracks.svg)

文字等價說明：上游 release 會 merge 到 active fork v2，通過 Linux、Windows、
API、TUI、Web、security gates 後才發布。Legacy fork v1 絕不 merge v2；重大／
安全修正可人工 port，且必須通過獨立 legacy container、API、model-asset checks。

## Release 主題

此 snapshot 保留 upstream recognition engine，同時把過去偏 Linux-container 的
fork 文件與交付面，改成 evidence-based Windows + Linux product surface：

- Windows desktop／package entrypoint 持續支援，並加入選用 validated HTTP API。
- Linux desktop 加入有界 X11 hotkey，清楚拒絕 Wayland／headless，且不做不安全
  selective-suppression claim。
- Linux container/server 仍是主要 headless deployment。
- Web、standard-library CLI、hash-locked Textual TUI 提供本機 transcription
  service clients，但不取代 desktop app。

## Added

### Desktop portability

- Universal server entrypoint：HTTP API 關閉時保留 upstream desktop default；
  啟用時只套用 validated HTTP environment settings。
- Windows distribution 的 PyInstaller specification 已整合 universal server 與
  required server hidden imports。
- Platform-aware shortcut backend policy：native Windows、bounded X11 callback、
  actionable Wayland／headless unavailable。
- Pinned Ubuntu 24.04／Windows 2022、Python 3.10／3.12 portability matrix，驗證
  portable desktop／package contracts 與 no-GUI CLI。

### Server 與 API

- Linux Docker／Compose server：model bootstrap、GPU preference、CPU fallback、
  bounded backend probe、persistent model／hotword／log locations、readiness-aware
  health check。
- 選用 OpenAI 相容 `whisper-1` file transcription surface：health、readiness、
  models、五種 response format、explicit capability errors、bounded multipart／
  decode／admission／deadline、OpenAI-style error envelope。
- Dedicated exact-pin API contract environment：missing dependency、empty
  discovery、failure、skipped contract test 都會失敗。

### Clients

- React/Vite Web Console：recording／upload、readiness-aware preflight、cancellable
  transcription、五種 format、download、local browser TTS、runtime config、browser
  smoke、static Nginx image。
- Standard-library no-GUI CLI：health／readiness／models、single／batch transcription、
  atomic output、portable filename、local OS TTS、zipapp packaging。
- Bilingual Textual TUI：diagnostics、file transcription、optional bounded
  microphone、cancellation、atomic save、memory-only key。
- Fully resolved SHA-256 TUI dependency lock：由 Python 3.10、3.12 consume，並有
  strict no-skip Pilot／unit verifier。

### Release、security 與 documentation

- Root verification／cleanup orchestration、documentation link／accessibility
  checker、upstream-divergence guard、pinned workflow runner／action，以及有 gate、
  provenance／SBOM request 的 server／Web image publishing。
- Docker model download、ffmpeg helper、GUI recording／file transcription、worker
  shutdown、GPU boost、hotword Ollama、GGUF metadata read 都加入 safer default 與
  bounded subprocess／network cleanup。
- Paired English／繁體中文 getting-started、deployment、troubleshooting、support／
  security、versioning、portability、API、TUI、release docs；包含 accessible SVG
  diagrams 與真實 Textual capture。

## Changed behavior 與 compatibility notes

| 變更 | Operator／user 影響 |
|---|---|
| Root identity 改為 Windows + Linux | Windows user 留在本 fork 的 desktop/package docs；Linux 分成 X11 desktop 與 headless/client profiles |
| HTTP API 仍是 opt-in | Upgrade 不會自動取代既有 WebSocket／desktop behavior |
| 啟用的非 loopback API 需要 auth | 設定 key／key file；explicit insecure override 只限 isolated test network |
| API 驗證 unknown／unsupported field | 新 SDK field 不會在 local engine 忽略時看似成功 |
| `/v1/audio/translations` 明確 `501` | 使用 transcription；不宣稱 local translation |
| X11 強制關閉 shortcut suppression | 保留 listening，又不冒 whole-keyboard／pointer grab 風險 |
| Wayland／headless desktop hotkey 清楚 fail | 使用 X11 或 Web／CLI／TUI／file path |
| Web default key 發布需要兩個 setting | Configured key 不會意外寫入 public `/config.js` |
| TUI core install 使用 hash lock | 應重建 venv，不可混用任意 global／user packages |
| Log 預設省略 prompt／transcript | 只有明確 privacy／retention 決策後才啟用全文 log |

## Security fixes 與 hardening

- Authentication 與 declared body size 可在讀 upload 前拒絕；raw／file／decoded
  limits 與 bounded admission 限制 work。
- Cancellation／timeout 會透過 bounded cleanup 關閉 client stream、decoder
  process、pending API route、TUI-owned temp audio。
- Key-file 支援避免長期 command-line token；各 client contract 會 mask 或只在
  memory 持有 UI key。
- Container 預設 loopback publish、drop capabilities、啟用
  `no-new-privileges`，並從 build context 排除 local secret／model／archive。
- Client／server error 會限制 untrusted response／log preview，並 redact 反射的
  configured secret。
- Dependency lock、full-SHA action、read-only workflow permission、release gate、
  attestation 改善 supply-chain traceability。

完整 boundary 與 reporting path 請見[支援與安全](support-security.md)。

## 從早期 Linux-container-only fork presentation 移轉

1. 讓既有 deployment 停止但可復原。依 local policy 備份 `.env`、
   `hot-server.txt`、Compose override、key file、model／cache path、log。
2. 使用 fresh v2 checkout 或 immutable image，不可用舊 source tree 覆蓋。
3. Diff 目前 `.env.example`。HTTP API 仍預設關閉；明確啟用、設定 auth，並取消
   current Compose HTTP `ports:` mapping 的註解。
4. 確認 model／hardware selection。不應 request GPU 的 host 使用 explicit CPU
   settings。
5. 驗證 `/health`、`/ready`、`/v1/models`，以及每個所需 format 的小型 known
   transcription。
6. 一次移動一個 client：desktop、CLI、TUI、SDK，最後 Web；browser 設 exact
   CORS origins。
7. Rollback window 結束前保留 previous source／image／configuration。

需要 Windows desktop behavior 的 user 應留在本 fork 的
[桌面可攜性指南](desktop-portability.md)，不再被導向 repository 之外。

## 從 fork v1 移轉

Fork v1、v2 是架構與 Git history 都已分岔的 product generation。請用 parallel
deployment 移轉：

1. 備份 v1 configuration／model assets；
2. 在不同 ports 部署 v2；
3. 跑 readiness／model／known-audio checks；
4. 先把一個 client 指向 v2；
5. 逐步 migration，rollback window 內讓 v1 停止但可復原。

不可把 v2 merge 或大量 cherry-pick 到 `maintenance/v1`。詳見
[維護政策](versioning.md)。

## 已知限制

- CI 不下載每一個 production model，也不證明 recognition quality。
- Windows CI 會 hash-install、build、搬移、ZIP round-trip、檢查並 import-smoke
  兩個 packaged EXE；每個 shipped artifact 仍需真實 tray、shortcut、audio／
  FFmpeg、model／known-audio、hardware 與 exit test。
- Linux global shortcut 需要 X11；Wayland／headless desktop hotkey 不支援。X11
  無法安全 selectively suppress 單一 key。
- GPU usability、memory、performance、fallback 依 target driver／device／model，
  需要 hardware evidence。
- Browser microphone 需要 loopback 或 HTTPS 與 user permission；browser TTS
  依賴本機 browser／OS voices。
- TUI microphone 依賴選用 platform-native `sounddevice`／PortAudio 與 real input
  device；file mode 仍可用。
- Local API 刻意只實作 bounded transcription subset，不含 streaming、
  diarization、translation 或所有現行 OpenAI Audio features。

## Stable fork-v2.0.0 前仍需的 qualification

- Green portable Ubuntu／Windows matrix 與 isolated API／TUI jobs。
- Root verification、documentation、cleanup、Web browser／image smoke、
  supply-chain workflow source guards。
- 發布 binary 時實際 Windows package-job ZIP／digest 與 real desktop／hardware
  checks。
- 宣稱 Linux desktop shortcut 時的 real X11 check。
- Advertised model／CPU／GPU profiles 的 live readiness／known-audio results。
- Immutable image／source references、model／dependency provenance，以及適用的
  SBOM／attestation。
- Upgrade／rollback rehearsal 與 final known-limit review。

## Changelog sources

- 本頁記錄 fork v2 delivery／portability／API／client changelog。
- [`docs/CHANGELOG.md`](../CHANGELOG.md) 是本 fork 繼承的 upstream product／
  recognition history。
- [`docs/state-of-fork.md`](../state-of-fork.md) 是 reviewer 使用的 detailed
  implementation／verification inventory。
- Git tag 與 merge history 才是實際 released 內容的權威記錄；unreleased heading
  不是 release tag。

Fresh install 請讀[開始使用](getting-started.md)，upgrade 請讀
[deployment](deployment.md#upgrade-與-rollback)。
