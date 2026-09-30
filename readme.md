<div align="center">

<img src="docs/assets/logo.png" width="84" alt="CapsWriter 圖示">

# CapsWriter-Offline v2

**按住 CapsLock 說話，放開就打字。全程離線的語音輸入法，也是可以放在 NAS 上共享的語音轉文字服務。**

繁體中文 · [English](README.en.md)

[![Release](https://img.shields.io/github/v/release/DF-wu/CapsWriter-Offline-Container?include_prereleases&label=release)](https://github.com/DF-wu/CapsWriter-Offline-Container/releases)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platforms](https://img.shields.io/badge/platforms-Windows%20%7C%20Linux-334155)](docs/zh-TW/desktop-portability.md)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker&logoColor=white)](docker-compose.yml)
[![OpenAI compatible](https://img.shields.io/badge/OpenAI%20audio-compatible-10A37F)](docs/zh-TW/openai-api.md)

</div>

![CapsWriter v2 架構：桌面、Web、CLI、TUI、OpenAI SDK 五種 Client 連到本機的 CapsWriter Server](docs/assets/overview.svg)

## 目錄

- [它能做什麼](#它能做什麼)
- [先搞懂兩個角色：Server 與 Client](#先搞懂兩個角色server-與-client)
- [我該選哪一種安裝方式？](#我該選哪一種安裝方式)
- [快速開始 A：Windows 桌面語音輸入](#快速開始-awindows-桌面語音輸入)
- [快速開始 B：Linux Docker Server](#快速開始-blinux-docker-server)
- [快速開始 C：Web／CLI／TUI／SDK Client](#快速開始-cwebclituisdk-client)
- [日常使用](#日常使用)
- [模型怎麼選](#模型怎麼選)
- [文件導覽](#文件導覽)
- [常見問題](#常見問題)
- [版本、Upstream 與授權](#版本upstream-與授權)

## 它能做什麼

| | 功能 | 說明 |
|---|---|---|
| 🎙️ | **聽寫輸入** | 在任何輸入框按住 CapsLock（或滑鼠側鍵）說話，放開後文字自動輸入。短按仍是原本的 CapsLock。 |
| 🔒 | **完全離線** | ASR、標點、數字轉換都在本機執行；音訊不會送到雲端。LLM 潤色是選用功能，可接本機 Ollama／LM Studio。 |
| 📁 | **檔案轉字幕** | 把音訊／影片拖到 Client 上，產生 `.srt`、`.txt`、`.json`；HTTP Client 另支援 `.vtt`。 |
| 🔥 | **熱詞與規則** | `hot.txt` 以音素模糊比對修正專有名詞；`hot-rule.txt` 支援正則替換。 |
| 🤖 | **LLM 角色** | 說「翻譯……」「助理……」等前綴，交給 LLM 翻譯、潤色或回答。 |
| 🐳 | **Docker Server** | Linux 一行指令啟動，模型自動下載；CPU 可用，NVIDIA／Intel／AMD GPU 選用加速。 |
| 🔌 | **OpenAI 相容 API** | `POST /v1/audio/transcriptions`（`whisper-1`），現有 OpenAI SDK 改個 base URL 就能用。 |
| 🖥️ | **多種 Client** | Windows／Linux X11 桌面、瀏覽器 Web Console、無 GUI CLI、Textual TUI。 |

<p align="center">
  <img src="assets/demo.png" width="860" alt="Windows 上的 CapsWriter：左邊是 Server 視窗顯示模型輸出，右邊是 Client 視窗顯示每次錄音的時長、延遲與辨識結果">
  <br><sub>Windows 實機畫面（取自 upstream）：左為 Server、右為 Client；短句轉錄延遲約 0.1–0.2 秒。</sub>
</p>

## 先搞懂兩個角色：Server 與 Client

CapsWriter 一定由兩個程式合作：

| 元件 | 負責 | 不負責 |
|---|---|---|
| **Server** | 載入 ASR 模型、解碼音訊、套用服務端熱詞、排程推論、產生逐字稿與字幕、回報健康狀態 | 不提供瀏覽器／終端機介面，也不操作使用者的剪貼簿或全域快捷鍵 |
| **Client** | 錄音或選檔、送出音訊、顯示／儲存結果；桌面 Client 另有托盤、快捷鍵、文字輸入 | 不載入模型、不做 ASR 推論 |

Server 提供兩個介面，給不同的 Client 使用：

| 介面 | 預設 | 誰在用 |
|---|---:|---|
| WebSocket `ws://127.0.0.1:6016` | 開啟 | Windows／Linux X11 桌面 Client |
| OpenAI 相容 HTTP `http://127.0.0.1:6017` | **關閉，需明確啟用** | Web Console、CLI、TUI、OpenAI SDK、curl |
| Web Console `http://127.0.0.1:8080` | 選用 | 只提供網頁；推論仍由 `:6017` 後面的 Server 執行 |

> [!TIP]
> Web、CLI、TUI **不是**另一套辨識引擎，它們都要連到「已啟用 HTTP API」的
> CapsWriter Server。桌面 Client 直接走 WebSocket，一般不需要開 HTTP API。
> 詳見 [Server 與 Client 分工](docs/zh-TW/server-and-clients.md)。

## 我該選哪一種安裝方式？

| 我想要…… | Server 裝在 | Client 用 | 從這裡開始 |
|---|---|---|---|
| 在自己的 Windows 電腦用語音打字 | 同一台 Windows（`start_server.exe`） | 桌面 Client（`start_client.exe`） | [快速開始 A](#快速開始-awindows-桌面語音輸入) |
| 家裡／公司有一台 Linux 主機或 NAS，多台電腦共用 | Linux Docker | 各電腦的桌面 Client、Web、CLI | [快速開始 B](#快速開始-blinux-docker-server) → [C](#快速開始-cwebclituisdk-client) |
| 用瀏覽器錄音或上傳檔案轉文字 | 任一已開 HTTP 的 Server | Web Console | [快速開始 C](#快速開始-cwebclituisdk-client) |
| 在腳本／SSH／CI 裡批次轉錄 | 任一已開 HTTP 的 Server | CLI | [CLI 指南](docs/zh-TW/cli-client.md) |
| 把現有 OpenAI Whisper 程式改成本機 | 任一已開 HTTP 的 Server | OpenAI SDK／curl | [API 指南](docs/zh-TW/openai-api.md) |
| Linux 桌面語音輸入 | 同一台 Linux（source） | Linux X11 桌面 Client | [開始使用：Linux X11](docs/zh-TW/getting-started.md#路徑-blinux-x11-desktop) |

## 快速開始 A：Windows 桌面語音輸入

### 1. 下載並解壓

到 [GitHub Releases](https://github.com/DF-wu/CapsWriter-Offline-Container/releases)
下載 `CapsWriter-Offline-windows-x86_64.zip`，**整個資料夾**解壓到一般路徑（例如
`D:\CapsWriter-Offline`）。可先用附帶的 `SHA256SUMS` 驗證檔案。

資料夾內有兩個程式，角色不能互換，也不要單獨搬出其中一個 EXE：

```text
CapsWriter-Offline/
├─ start_server.exe      ← Server：載入模型、辨識
├─ start_client.exe      ← Client：托盤、快捷鍵、錄音、打字
├─ config_server.py      ← Server 進階設定（模型、連接埠……）
├─ config_client.py      ← Client 進階設定（快捷鍵、熱詞、LLM……）
├─ hot.txt / hot-rule.txt / hot-server.txt
├─ LLM/                  ← LLM 角色
└─ models/               ← 模型放這裡（ZIP 內是空的）
```

### 2. 放入模型與 GGUF runtime

為了控制檔案大小與授權，ZIP 內**不含**模型、llama.cpp DLL 與 FFmpeg。預設模型
Qwen3-ASR 需要兩個檔案（都有固定 SHA-256）：

| 下載 | 解壓到 |
|---|---|
| [`Qwen3-ASR-1.7B-q5_k.zip`](https://github.com/HaujetZhao/CapsWriter-Offline/releases/download/models/Qwen3-ASR-1.7B-q5_k.zip)（約 1.8 GB） | `models/Qwen3-ASR/`，結果為 `models/Qwen3-ASR/Qwen3-ASR-1.7B/` |
| [`llama-b7798-bin-win-vulkan-x64.zip`](https://github.com/ggml-org/llama.cpp/releases/download/b7798/llama-b7798-bin-win-vulkan-x64.zip) | 其中的 `*.dll` 複製到 `core/server/engines/llama/bin/` |

[桌面可攜性指南](docs/zh-TW/desktop-portability.md#準備下載的-windows-package)
有一段可直接貼到 PowerShell 的指令，會自動下載、驗 hash、解壓到正確位置。
要轉錄影片／音檔，再把可信來源的 `ffmpeg.exe`（與 `ffprobe.exe`）放進 `PATH` 或
資料夾根目錄；只用麥克風聽寫則不需要。

### 3. 先開 Server，再開 Client

1. 雙擊 `start_server.exe`，等畫面出現「開始服務」與模型載入完成。
2. 雙擊 `start_client.exe`，右下角托盤會出現 CapsWriter 圖示。

第一次使用或要改設定時，執行 `start_client.exe --settings`，或在托盤右鍵選
**設定**：

<table>
  <tr>
    <td width="50%"><img src="docs/assets/desktop-settings-connection.png" alt="設定視窗的「連線」頁：Server 主機 127.0.0.1、WebSocket 連接埠 6016、測試 Server 連線按鈕、辨識語言 auto"></td>
    <td width="50%"><img src="docs/assets/desktop-settings-recording.png" alt="設定視窗的「錄音與快捷鍵」頁：麥克風選單、長按觸發時間 0.3 秒、快捷鍵表格列出 caps_lock 與滑鼠 x2"></td>
  </tr>
  <tr>
    <td><b>連線</b>：Server 在本機填 <code>127.0.0.1</code>；在其他主機就填它的 IP，按「測試 Server 連線」。</td>
    <td><b>錄音與快捷鍵</b>：選麥克風、調整長按時間、新增或停用快捷鍵（鍵盤或滑鼠側鍵）。</td>
  </tr>
</table>

<sub>以上為同一個 Tk 設定視窗在 Linux（Xvfb）上的實際渲染；Windows 上會套用系統原生外觀與字型。</sub>

儲存後重新啟動 Client 即生效。設定存在 `%LOCALAPPDATA%\CapsWriter\client-settings.json`，
只記錄你改過的欄位，其餘沿用 `config_client.py`。詳見[日常設定指南](docs/settings.md)。

### 4. 開始說話

![聽寫四步驟：點進輸入框、按住 CapsLock 說話、放開、文字自動輸入](docs/assets/dictation-flow.svg)

> [!NOTE]
> 發行 ZIP 由 GitHub Actions 的 `windows-package` job 以 hash lock 建置
> PyInstaller 套件，搬離 checkout 後壓縮／解壓、拒絕 reparse point，並讓兩個
> EXE 各自通過 `--artifact-self-check`。真實麥克風、托盤、快捷鍵、模型與 GPU
> 仍請在你的電腦上確認；想自行建置請看 [BUILD_GUIDE](assets/BUILD_GUIDE.md)。

## 快速開始 B：Linux Docker Server

需求：`linux/amd64`、Docker Engine 與 Compose plugin，以及數 GB 的模型空間。GPU 為選用。

```bash
git clone https://github.com/DF-wu/CapsWriter-Offline-Container.git
cd CapsWriter-Offline-Container
cp .env.example .env
cp hot-server.example.txt hot-server.txt
docker compose up -d capswriter-server
docker compose logs -f capswriter-server   # 第一次會下載模型，請耐心等候
```

啟動完成後，WebSocket `:6016` 就能給桌面 Client 使用（在設定視窗填這台主機的 IP）。

| 想要 | 追加的 Compose 檔 |
|---|---|
| NVIDIA GPU | `-f docker-compose.gpu.yml` |
| Intel／AMD 內顯（Vulkan） | `-f docker-compose.igpu.yml` |
| 較低延遲、較小的 Fun-ASR-Nano 模型 | `-f docker-compose.fun-asr.yml` |
| 自己管理 `./models` 目錄 | `-f docker-compose.models-bind.yml` |
| 在 Web 上管理 Server 設定 | `-f docker-compose.settings.yml`（見[日常設定](docs/settings.md)） |

例如 NVIDIA：`docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d capswriter-server`。
完整說明、升級與備份請看[部署指南](docs/zh-TW/deployment.md)。

### 啟用 HTTP API（給 Web／CLI／TUI／SDK）

在 `.env` 設定：

```dotenv
CAPSWRITER_HTTP_API_ENABLE=true
CAPSWRITER_HTTP_API_KEY=replace-with-a-long-random-token
CAPSWRITER_HTTP_API_PUBLISH_HOST=127.0.0.1
CAPSWRITER_HTTP_API_PORT=6017
CAPSWRITER_HTTP_API_CORS_ORIGINS=http://127.0.0.1:8080,http://localhost:8080,http://127.0.0.1:5173,http://localhost:5173
```

再到 [`docker-compose.yml`](docker-compose.yml) 取消第二個 port mapping 的註解：

```yaml
ports:
  - "127.0.0.1:6016:6016"
  - "127.0.0.1:6017:6017"
```

重建並檢查：`/health` 只代表程序活著，`/ready` 才代表模型已可接收音訊。

```bash
docker compose up -d --force-recreate capswriter-server
curl http://127.0.0.1:6017/health
curl http://127.0.0.1:6017/ready
```

> [!WARNING]
> 對區網或網際網路開放 HTTP 時，一定要保留 API key，並放在 TLS reverse proxy
> 或私有 overlay network（如 Tailscale）後面。見[支援與安全](docs/zh-TW/support-security.md)。

## 快速開始 C：Web／CLI／TUI／SDK Client

以下假設 Server 的 HTTP API 在 `http://127.0.0.1:6017`，key 為
`replace-with-a-long-random-token`。

### Web Console：瀏覽器錄音或上傳

```bash
CAPSWRITER_WEB_API_BASE=http://127.0.0.1:6017 \
  docker compose -f docker-compose.web.yml up -d --build capswriter-web
```

開啟 `http://127.0.0.1:8080`，在 API 欄位確認 `http://127.0.0.1:6017`，於遮罩欄位
貼上 key，就可以錄音或拖入檔案，並下載 text／json／srt／vtt。

![Web Console 實機截圖：左欄連線設定與 Server 診斷（Health、Ready、Router、FFmpeg 皆為 ok），中欄已上傳 zh.wav 並轉錄出「開放時間：早上九點至下午五點。」，右欄為 TTS 與歷史](docs/assets/web-console.png)

<sub>真實截圖：連到本機 Qwen3-ASR 1.7B（CPU）Server，轉錄 5.6 秒的中文測試音檔。</sub>

想改前端時用開發模式（需 Node.js 24）：

```bash
cd client/web
npm ci --no-audit --no-fund
npm run dev      # 開啟 http://127.0.0.1:5173
```

### CLI：腳本與批次

只需要 Python 3.10+ 標準函式庫：

```bash
export CAPSWRITER_API_BASE=http://127.0.0.1:6017
export CAPSWRITER_HTTP_API_KEY=replace-with-a-long-random-token
python client/cli/capswriter_cli.py ready
python client/cli/capswriter_cli.py transcribe meeting.wav --format text
python client/cli/capswriter_cli.py transcribe audio/*.mp3 --format srt --output-dir subs/
```

![CLI 實機輸出：health 回報 qwen_asr、中文音檔轉成文字、英文音檔轉成 SRT、兩個檔案批次輸出 VTT](docs/assets/cli-demo.svg)

### TUI：終端機工作台

```bash
python3.12 -m venv .venv-tui
.venv-tui/bin/python -m pip install \
  --require-hashes --only-binary=:all: \
  --requirement requirements/tui.lock
.venv-tui/bin/python -m client.tui --base-url http://127.0.0.1:6017
```

在「API key（只存於記憶體）」欄位貼上 key，按 **F5** 檢查 Server，輸入檔案路徑後
按 **Ctrl+T** 轉錄、**Ctrl+S** 儲存。

![真實 CapsWriter TUI：上方 Server 診斷全部正常，左下為 zh.wav 與參數，右下為轉錄結果「開放時間：早上九點至下午五點。」](docs/assets/tui-transcript.svg)

### OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:6017/v1", api_key="replace-with-a-long-random-token")
with open("meeting.wav", "rb") as f:
    print(client.audio.transcriptions.create(model="whisper-1", file=f).text)
```

支援 `text`、`json`、`verbose_json`、`srt`、`vtt`；不支援 translation、streaming、
diarization。完整範圍見 [OpenAI 相容 API](docs/zh-TW/openai-api.md)。

## 日常使用

完整圖文教學：**[使用教學](docs/zh-TW/usage.md)**。重點如下：

| 我想…… | 怎麼做 |
|---|---|
| 語音打字 | 按住 CapsLock 或滑鼠側鍵 X2 說話，放開即輸入；短按仍是原本功能 |
| 轉錄影片成字幕 | 把檔案拖到 `start_client.exe` 圖示上，同資料夾產生 `.srt`／`.txt`／`.json` |
| 修正字幕錯字 | 改好 `.txt` 後再拖回 Client，會沿用原時間軸重新產生 `.srt` |
| 讓專有名詞更準 | 在 `hot.txt` 每行寫一個詞，存檔 3 秒內自動生效 |
| 固定替換 | 在 `hot-rule.txt` 寫 `毫安時 = mAh` 或正則規則 |
| 說話前加「翻譯」 | 觸發 LLM 角色（需在 `LLM/*.py` 設定 provider 與 key） |
| 輸出繁體中文 | 設定視窗「輸出」頁勾選「轉為繁體中文」，可選 `zh-tw`／`zh-hk` |
| 看今天說過什麼 | 托盤右鍵 **日記**，依日期存成 Markdown 並附錄音 |
| 暫停 LLM 輸出 | 按 `Esc` |

托盤右鍵選單：**設定**、複製結果、日記、上下文、熱詞、清除記憶、重開音訊、重啟、退出。

## 模型怎麼選

| 模型（`model_type`） | 大小 | 特色 | Docker 自動下載 |
|---|---:|---|:---:|
| **Qwen3-ASR 1.7B**（`qwen_asr`，預設） | 1.3–1.8 GB | 準確率最高，中英混說佳；可用 Vulkan／CUDA 加速 | ✅ |
| **Fun-ASR-Nano**（`fun_asr_nano`） | 約 0.8 GB | 延遲低、支援服務端熱詞；CPU 也順 | ✅ |
| SenseVoice（`sensevoice`） | 約 0.4 GB | 輕量，多語（中英日韓粵） | 手動 |
| Paraformer（`paraformer`） | 約 0.5 GB（含標點模型） | 輕量中文 | 手動 |

Docker 以 `CAPSWRITER_MODEL_TYPE` 選模型；Windows 版改 `config_server.py` 的
`model_type`。模型來自 [upstream model release](https://github.com/HaujetZhao/CapsWriter-Offline/releases/tag/models)；
GPU 相關問題見 [显卡加速的若干问题](docs/显卡加速的若干问题.md)。

## 文件導覽

| 讀者 | 文件 |
|---|---|
| 第一次使用 | [Server 與 Client 分工](docs/zh-TW/server-and-clients.md) → [開始使用](docs/zh-TW/getting-started.md) → [使用教學](docs/zh-TW/usage.md) |
| Windows 使用者 | [桌面可攜性](docs/zh-TW/desktop-portability.md) · [日常設定](docs/settings.md) · [常見問題（upstream）](docs/常见问题.md) |
| 熱詞／角色／轉錄 | [熱詞](docs/热词功能如何使用.md) · [LLM 角色](docs/角色功能如何使用.md) · [檔案轉錄](docs/文件转录功能如何使用.md) · [辨識語言](docs/识别语言如何配置.md) |
| Server 維運 | [部署](docs/zh-TW/deployment.md) · [支援與安全](docs/zh-TW/support-security.md) · [疑難排解](docs/zh-TW/troubleshooting.md) |
| Client | [Web Console](docs/zh-TW/web-console.md) · [CLI](docs/zh-TW/cli-client.md) · [TUI](docs/zh-TW/tui.md) · [OpenAI 相容 API](docs/zh-TW/openai-api.md) |
| 版本與發行 | [Release notes](docs/zh-TW/release-notes.md) · [v1／v2 維護政策](docs/zh-TW/versioning.md) · [驗證](docs/verification.md) |
| 開發者 | [開發流程](docs/development.md) · [架構](docs/architecture.md) · [上游同步](docs/upstream-sync-guide.md) · [文件首頁](docs/zh-TW/README.md) |

## 常見問題

<details>
<summary><b>按了 CapsLock 沒反應？</b></summary>

先確認 Server 視窗已顯示模型載入完成，再看 Client 視窗是否顯示「已連接服務端」。
要在以系統管理員身分執行的程式（例如工作管理員、部分遊戲）中輸入，Client 也要以
系統管理員身分執行。日誌在 `logs/client_latest.log` 與 `logs/server_latest.log`。
</details>

<details>
<summary><b>Docker 一直停在 starting？</b></summary>

第一次啟動會下載模型（1–2 GB），healthcheck 預留 20 分鐘。用
`docker compose logs -f capswriter-server` 看進度；若啟用了 HTTP，`/ready` 會列出
還沒就緒的元件。見[疑難排解](docs/zh-TW/troubleshooting.md)。
</details>

<details>
<summary><b>Web Console 顯示 CORS 或 401？</b></summary>

CORS：把 Web 的來源（如 `http://127.0.0.1:8080`）加進
`CAPSWRITER_HTTP_API_CORS_ORIGINS` 後重建 Server。401：Web 輸入的 key 必須與
Server 的 `CAPSWRITER_HTTP_API_KEY` 相同。
</details>

<details>
<summary><b>Linux 桌面可以用快捷鍵嗎？</b></summary>

可以，但只支援 X11；Wayland 與 headless 沒有可靠的全域快捷鍵。X11 下無法只攔截
單一按鍵，所以 CapsLock 仍會切換大小寫，建議改用 F12 或滑鼠側鍵。
</details>

<details>
<summary><b>一定要 GPU 嗎？</b></summary>

不用。CPU 就能執行；Qwen3-ASR 在 CPU 上短句約數秒，想要 0.1–0.3 秒延遲再加 GPU，
或改用 Fun-ASR-Nano。
</details>

## 版本、Upstream 與授權

- **fork v2**（本分支 `master`）是持續開發的版本，release tag 為 `fork-v2.x.y`；
  **fork v1** 是另一條只做安全／相容性維護的分支。見[版本政策](docs/zh-TW/versioning.md)。
- Server／Web image 以不可變的 `sha-<commit>` tag 發布到 GHCR，並附 SBOM／provenance；
  `latest` 只指向通過檢查的 `master`。
- 本 fork 基於 [HaujetZhao/CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline)，
  模型、推論演算法與桌面體驗來自 upstream；fork 增加的是跨平台部署、HTTP API、
  多種 Client、安全邊界與發行流程。喜歡這個專案，也請支持
  [upstream 作者](https://github.com/HaujetZhao/CapsWriter-Offline)。

授權：[MIT](LICENSE)。
