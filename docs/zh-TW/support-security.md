# 支援與安全

> [文件首頁](README.md) · [English](../en/support-security.md) · [Release notes](release-notes.md)

本頁定義 fork v2 的「支援」意義與預設 security／privacy posture，並區分 automated
contract evidence 與 manual release evidence。

![CapsWriter verification pipeline：涵蓋 upstream divergence、portable desktop／CLI、server／API／Docker、Web／TUI gates 與 cleanup](../assets/verification-pipeline.svg)

文字等價說明：獨立 checks 會驗證 upstream drift、portable Python contract、以
hash 安裝的 Windows PyInstaller build、ZIP relocation 與 EXE import smoke、
server／API／Docker behavior、Web build／browser path、hash-locked TUI、文件與
generated-artifact cleanup。真實 model、audio、GPU、display、browser device、
tray／hook 與 known audio 仍是 automated CI 以外的明確 release evidence。

## 支援詞彙

- **支援：**有 documented entrypoint 與 automated portable contract；release
  operator 仍需蒐集表中列出的 hardware／artifact evidence。
- **有限支援：**預期用途可用，但某項重要行為因 platform 不同，並在下方說明。
- **不支援：**fork 刻意拒絕或未實作；清楚 fail／fallback 是預期結果。
- **沒有 release gate：**code 可能偶然可執行，但在有 pinned automated／manual
  gate 前，本專案不做 support claim。

## Platform 與 surface matrix

| Surface | 狀態 | Automated evidence | 必要 manual／release evidence |
|---|---|---|---|
| Windows desktop package | 支援 | Windows 2022／Python 3.12 hash-only dependency install、PyInstaller build、relocation／ZIP extraction、reparse-point inspection、兩個 EXE import smoke，以及四組 source matrix | 真實 Windows 的 tray、shortcut、microphone、FFmpeg、model/runtime asset、known audio、選用 API、CPU、DirectML 與 GPU profile，以及 clean child shutdown |
| Linux X11 desktop shortcut | 有限支援 | Backend detection、callback mapping、listener wiring、failure downgrade tests | 真實 X11 session、keyboard/mouse device、所選 shortcut 與 text-injection workflow |
| Linux Wayland global shortcut | 不支援 | Backend 回 unavailable，且不建構 listener | 使用 X11，或改用不需 global hotkey 的 Web／CLI／TUI／file workflow |
| Linux headless desktop shortcut | 不支援 | Headless backend 回 unavailable | 使用 container／source server 與非 desktop clients |
| Linux container server（`linux/amd64`） | 支援 | Dockerfile／Compose source guards、unit tests、health／readiness／image smoke gates | Selected model download/load、known audio、目標 CPU／GPU host、persistence、restart |
| Linux ARM64 container | 沒有 release gate | 沒有 ARM64 dependency lock、native runtime bundle、image build 或 model-load gate | 使用受支援的 `linux/amd64` host；不可從 portable Python-only check 推論支援 |
| CPU inference fallback | 支援 | Backend selection／fallback tests、configuration guards | 目標 CPU 上的 known-audio latency／quality |
| GPU acceleration | Selected backend／driver 可用時支援 | Probe／fallback 與 configuration tests | 實際 driver／device visibility、model load、memory、known audio、fallback observation |
| OpenAI 相容 transcription subset | 啟用時支援 | Dedicated exact-pin API contract no-skip job | Production release 的 live authenticated deployment 與 known audio |
| Web Console | 現代 Windows／Linux browser 支援 | Unit/type/build、browser mock smoke、static-image smoke | Target browser；使用 microphone 時的 permission／secure context；real API／model |
| 無 GUI CLI | Windows／Linux Python 3.10、3.12 支援 | 四組 OS／Python portability matrix 與 packaged zipapp smoke | 需要 `speak` 時的 target local TTS command／voice |
| Textual TUI core file workflow | Windows／Linux Python 3.10–3.12 支援 | Ubuntu 24.04／Windows 2022 × Python 3.10／3.12 四組 hash-locked no-skip Pilot matrix | Target terminal rendering；Windows release 需要 Windows terminal run |
| TUI 選用 microphone | Native stack／device 可用時支援 | Bounded recorder unit／Pilot contracts | Target OS 的 `sounddevice`、PortAudio、permission、real device、cleanup |
| macOS product | 沒有 release gate | 只有個別 portable helper 可能有 tests | 無 project-level desktop／server／client support claim |

「Windows + Linux fork」指的是上表的 documented surfaces，不代表每一種 display
server、browser、audio device 或 accelerator 都有相同行為。

## Security defaults

### Network exposure

- HTTP API 預設關閉。
- Compose 預設把 WebSocket、選用 HTTP、Web port 發布到 host loopback。
- 上游 source／desktop WebSocket 預設為 `0.0.0.0`；若 universal entrypoint
  只供本機使用，請設 `CAPSWRITER_SERVER_ADDR=127.0.0.1`。
- API bind 到非 loopback 時必須有 Bearer key／key file，除非明確設定
  `CAPSWRITER_HTTP_API_ALLOW_INSECURE_BIND=true`，且只用於 isolated trusted
  test network。
- CORS 使用 explicit allowlist；它不能取代 authentication。
- Client 會拒絕 URL credential、non-HTTP scheme、query、fragment；也不會帶
  credential 跟隨 redirect。

Remote access 應使用 maintained TLS reverse proxy 或 private overlay network，
保留 authentication，只發布必要 port，並維持 body／time／concurrency limits。

### Secrets

- 長期 server／CLI deployment 優先使用 `CAPSWRITER_HTTP_API_KEY_FILE` 或 secret
  mount／service-manager secret。
- TUI key 會遮罩且只存於記憶體；沒有 command-line key option。
- Web UI 內輸入的 key 只留 page memory。注入 `/config.js` 的 key 對所有可載入
  page 的人公開，因此 container 要求另設
  `CAPSWRITER_WEB_ALLOW_PUBLIC_API_KEY=true`。
- `.env`、key file、certificate、本機 archive 都排除於 Docker build context；
  不可 commit。
- Web 的 OpenAI-style、legacy `detail`、non-JSON、malformed-response 與 network
  error 會先轉成 control-safe 且有長度上限的文字，再遮蔽當次 request 使用的
  記憶體內 key，之後才寫入 React state 或畫面。Shared verification log 也會在
  可能反射 key 的位置 redact。

疑似 disclosure 後請 rotate key，並在 controlled restart 中更新 server／client。
不可把 active key 貼進 issue 或 transcript。

### Audio、transcript 與 log

- Asset 備妥後 inference 留在本機，但第一次 model／runtime bootstrap 可能連到
  configured download sources。
- HTTP prompt／transcript logging 預設關閉。啟用後會改變 privacy boundary，必須
  有 retention／access policy。
- TUI microphone WAV 使用 private temporary directory，會在 successful upload、
  cancel、replacement、normal exit 清除；failed upload 只為同 session retry 暫留。
- CLI／TUI output 寫到 user-selected path，並有 atomic replacement guard；TUI
  不會覆寫 source audio。
- Web settings 與最多 20 筆明文 transcript／raw history record 會放在 browser
  localStorage；這些 history 可能是敏感資料。手動輸入 API key 不會持久化。
  共用電腦應清除 site data；browser storage 不是 encrypted records system。
- Server log 預設位於 named volume；operator 負責 access、rotation、backup、delete。

沒有 explicit consent 與 redaction 時，不可把 private recording／transcript 當
public bug fixture。

### Resource 與 parser boundaries

API 會限制 raw／file upload size、decoded audio duration、active／pending requests、
response formats、multipart fields、ffmpeg output 與單一 end-to-end deadline。
Client 另有 timeout／response body limit。Unsupported field／capability 會被拒絕，
不會靜默忽略。

這些 control 可降低 accidental exhaustion 與 ambiguous behavior，但不會讓
unauthenticated public endpoint 變成安全 internet service。

### Container 與 supply chain

- Server container drop Linux capabilities 並啟用 `no-new-privileges`。
- Docker、TUI、Windows production-build Python dependency 使用 fully resolved
  SHA-256 hash lock。Server image lock 與 bundled llama runtime 針對
  `linux/amd64`；Windows lock 針對 CPython 3.12 x86-64 Windows，且只允許文件
  記載的 `srt` source exception；Web 使用 `package-lock.json` 與 `npm ci`。目前
  不宣稱 ARM64 server image 支援。
- CI runner 與 third-party action 有 pin；checkout 不保留 write credential。
- Server／Web publish workflow 先跑 gate，並要求 provenance／SBOM attestation。
  Server workflow 在 promotion `latest` 前，會 pull 並 smoke-test exact pushed
  digest 的 dependency consistency、runtime import、non-root execution 與
  entrypoint syntax。Consumer 仍應 pin immutable image digest，並驗證
  registry／attestation。
- Model asset 是 runtime data，不是 source-code unit test 能證明的項目。Controlled
  deployment 應記錄 source／version／checksum。

## Vulnerability reporting

若 fork repository 有 private vulnerability-reporting channel，請優先使用。內容
應包含 affected source commit／image digest、surface、minimal reproduction、
impact、proposed disclosure timeline；不可包含 live key、private audio 或 unrelated
environment dump。

若尚未啟用 private channel，請透過 repository maintainer／contact surface 聯絡
fork maintainer 並要求 private channel。Public issue 可以只說有 security report，
在 fix／disclosure plan 確認前不可放 exploit details。

若 vulnerability 來自 upstream engine／product，而非 fork 新增內容，也應與
upstream project 協調。Fork-specific API、container、portability、Web、CLI、TUI、
release-pipeline 問題則屬於本 fork。

## 一般 support request

提出前：

1. 依[疑難排解診斷順序](troubleshooting.md#診斷順序)操作。
2. 能做到時，請在 clean supported path 重現。
3. 附上 source commit／image digest、platform／session、Python／client、model／
   hardware selection、sanitized readiness response、bounded logs。
4. 說明 small known file 是否成功。
5. 移除 key、recording、transcript、prompt、personal path；只有 essential minimal
   redacted sample 可例外。

Hardware／model-quality report 需要 target hardware／model／audio evidence；只有 CI
output 不足。

## Release evidence

做 supported release claim 前應保留：

- portable Ubuntu／Windows matrix logs；
- isolated API contract 與四組 Ubuntu／Windows TUI no-skip logs；
- root verification、Web build／browser／image smoke、docs、cleanup logs；
- 發布 Windows binary 時實際上傳 ZIP、digest，以及 hash-install／build／
  relocation／兩個 EXE self-check workflow results；
- 宣稱 Linux desktop shortcut 時的 real X11 results；
- 每個 advertised CPU／GPU／model profile 的 live readiness 與 known-audio
  transcription；
- immutable source／image reference 與 dependency／model provenance；
- rollback instruction 與 known limitations。

Command 請見[驗證](../verification.md)，目前 snapshot 請見
[release notes](release-notes.md)。

## Lifecycle

Fork v2 為 active。Legacy fork v1 保持隔離，只接受重大、安全、相容性、model
asset 維護。不可把 v2 整體 merge 到 v1；應人工 port 小型 reviewed fix，並在
target generation 測試。詳見 [v1／v2 維護政策](versioning.md)。
