# 無 GUI CLI 客戶端

> [文件首頁](README.md) · [English](../en/cli-client.md) · 繁體中文 · [語言選擇](../cli-client.md)

無 GUI 客戶端位於 [`client/cli`](../../client/cli)。這是一套只使用 Python
標準函式庫的客戶端，可連接與 OpenAI 相容的 CapsWriter HTTP API，並使用本機
作業系統的文字轉語音功能。它適合指令碼、SSH 工作階段、CI 冒煙測試、批次轉錄，
以及不需要桌面 GUI 或瀏覽器主控台的使用者。

![CapsWriter CLI 客戶端流程：命令列輸入連接 HTTP API 或本機文字轉語音引擎](../assets/cli-client-flow.svg)

## 功能

| 範疇 | 支援內容 |
|---|---|
| 伺服器檢查 | `health`、`ready`、`models` |
| 語音轉文字 | 以 `POST /v1/audio/transcriptions` 執行 `transcribe`，可處理一個或多個音訊檔案 |
| 格式 | `text`、`json`、`verbose_json`、`srt`、`vtt` |
| 批次輸出 | stdout、單一明確的 `--output`，或在 `--output-dir` 產生檔案 |
| 文字轉語音 | `speak` 可讀取直接輸入的文字、UTF-8 檔案或 stdin，並透過有執行時間上限的本機作業系統引擎朗讀 |
| 封裝 | 單一檔案的 Python zipapp（`capswriter-cli.pyz`） |
| 隔離性 | 不需要第三方 Python 相依套件；測試使用程序內的模擬 HTTP 伺服器 |

## 需求

- Linux 或 Windows 上的 Python 3.10 以上版本。
- 若要實際執行語音轉文字，需有運作中的 CapsWriter HTTP API。
- 選用的本機文字轉語音引擎：
  - Windows：一般桌面安裝中的 PowerShell `System.Speech`。
  - Linux：`spd-say`、`espeak-ng` 或 `espeak`。

CLI 不會安裝，也不需要任何全域套件。

## 封裝版 CLI

建立單一檔案的 zipapp：

```bash
python client/cli/scripts/build_zipapp.py
python client/cli/dist/capswriter-cli.pyz --help
```

成品會寫入 `client/cli/dist/capswriter-cli.pyz`。其中只含使用標準函式庫的 CLI
原始碼，可複製到裝有 Python 3.10 以上版本的 Linux 或 Windows 電腦：

```bash
python capswriter-cli.pyz health --base-url http://127.0.0.1:6017
python capswriter-cli.pyz ready --base-url http://127.0.0.1:6017
python capswriter-cli.pyz transcribe meeting.wav --format text
```

Git 會忽略 `client/cli/dist`，清理指令碼也會移除它。

## 伺服器設定

啟用 HTTP API：

```bash
CAPSWRITER_HTTP_API_ENABLE=true
CAPSWRITER_HTTP_API_BIND=0.0.0.0
CAPSWRITER_HTTP_API_PORT=6017
CAPSWRITER_HTTP_API_KEY=sk-local-dev
```

接著驗證：

```bash
python client/cli/capswriter_cli.py --help
python client/cli/capswriter_cli.py health --base-url http://127.0.0.1:6017 --key sk-local-dev
python client/cli/capswriter_cli.py ready --base-url http://127.0.0.1:6017 --key sk-local-dev
python client/cli/capswriter_cli.py models --base-url http://127.0.0.1:6017 --key sk-local-dev
```

在正式環境的 shell 與程序管理工具中，建議使用金鑰檔案，以免權杖直接出現在
命令歷史或程序參數中：

```bash
install -m 600 -D /dev/null /run/secrets/capswriter-http.key
printf '%s\n' 'sk-local-dev' > /run/secrets/capswriter-http.key
python client/cli/capswriter_cli.py ready \
  --base-url http://127.0.0.1:6017 \
  --key-file /run/secrets/capswriter-http.key
```

也可以使用環境變數：

```bash
export CAPSWRITER_API_BASE=http://127.0.0.1:6017
export CAPSWRITER_HTTP_API_KEY=sk-local-dev
python client/cli/capswriter_cli.py health
python client/cli/capswriter_cli.py ready
```

Windows PowerShell 的寫法如下：

```powershell
$env:CAPSWRITER_API_BASE = "http://127.0.0.1:6017"
$env:CAPSWRITER_HTTP_API_KEY = "sk-local-dev"
python client\cli\capswriter_cli.py health
python client\cli\capswriter_cli.py ready
```

若服務管理工具或 secret mount 能提供含有客戶端權杖的 UTF-8 檔案，CLI 應使用
`CAPSWRITER_HTTP_API_KEY_FILE`，而不是 `CAPSWRITER_HTTP_API_KEY`。去除空白後，
檔案內必須有非空白權杖。伺服器也接受同一變數作為 Bearer 權杖檔案；兩端都是
明確設定的 `CAPSWRITER_HTTP_API_KEY` 優先。

## 伺服器診斷

`health` 用來確認 HTTP 程序有回應。`ready` 的條件較嚴格：它會呼叫 `/ready`，
並印出 HTTP 工作路由器是否已綁定、`ffmpeg` 是否可用，以及目前有哪些操作限制
等部署診斷。將正式流量導入伺服器前，或容器健康檢查已通過但仍無法轉錄時，請
使用 `ready`。

## 轉錄

印出純文字：

```bash
python client/cli/capswriter_cli.py transcribe meeting.wav --format text
```

寫入單一輸出檔案：

```bash
python client/cli/capswriter_cli.py transcribe meeting.wav \
  --format verbose_json \
  --output meeting.transcript.json
```

`json` 與 `verbose_json` 輸出會寫成有效的 JSON，而不是純轉錄文字。

批次模式：

```bash
python client/cli/capswriter_cli.py transcribe audio/*.wav \
  --format srt \
  --output-dir transcripts/
```

`--output-dir` 會依每個音訊檔案的主檔名與回應格式產生檔名。產生的主檔名會經過
清理，確保可同時用於 Linux 與 Windows：控制字元和 Windows 不允許的檔名字元會
改為 `_`，開頭與結尾的空白或句點會移除，Windows 保留裝置名稱會加上後綴，
過長的名稱則會截短並加入短雜湊。若兩個輸入在清理或不分大小寫比較後會產生相同
目標路徑，CLI 會在傳送任何 HTTP 請求前失敗，因此批次作業不會無聲覆寫先前的
轉錄結果。

轉錄檔案會先寫入相同目錄中的暫存檔，再以不可分割的替換操作完成寫入。在
`--output-dir` 批次模式中，每個成功結果都會立即寫入，不必等整批完成；若後續
檔案失敗，先前輸出仍會保留。若最後替換失敗，原有轉錄檔會保持完整，暫存檔也會
移除。

為了相容性，語言與提示文字會傳給 HTTP API：

```bash
python client/cli/capswriter_cli.py transcribe meeting.wav \
  --language zh \
  --prompt "會議術語：CapsWriter, Qwen, FunASR"
```

伺服器會將 `zh`、`zh_CN`、`en`、`ja`、`ko`、`yue` 等常見別名正規化為 ASR
引擎使用的內部語言名稱。提示文字在換行正規化並限制為 2048 個字元後，會作為
辨識器的上下文傳入。

## 朗讀

朗讀直接輸入的文字：

```bash
python client/cli/capswriter_cli.py speak "CapsWriter transcription completed."
```

讀取 UTF-8 文字檔：

```bash
python client/cli/capswriter_cli.py speak transcript.txt --file
```

從 stdin 讀取文字，適合 shell 管線：

```bash
python client/cli/capswriter_cli.py transcribe meeting.wav --format text \
  | python client/cli/capswriter_cli.py speak --stdin
```

預覽將使用的本機命令：

```bash
python client/cli/capswriter_cli.py speak "test" --dry-run
```

`--file` 與 `--stdin` 互斥。`--stdin` 也會拒絕位置文字參數，避免指令碼意外忽略
輸入。

`speak` 命令不會呼叫 CapsWriter 伺服器，也不會把文字傳送到雲端服務；它會啟動
本機作業系統的文字轉語音引擎。`--tts-timeout` 會限制本機文字轉語音的執行時間，
預設為 `120` 秒，而且必須是正數。

## 結束代碼

| 代碼 | 意義 |
|---|---|
| `0` | 成功 |
| `1` | HTTP 失敗、不支援的格式、缺少檔案或文字輸入、無效的文字轉語音輸入組合、找不到本機文字轉語音引擎，或本機文字轉語音逾時 |

當伺服器回傳 OpenAI 樣式的 `{"error": ...}` JSON 時，CLI 會印出其中的
`error.message`，而不是傾印原始 JSON。為了相容舊版伺服器，它也會正規化舊式
FastAPI `{"detail": ...}` 回應。若代理伺服器或舊版伺服器回傳非 JSON 的 HTTP
錯誤內容，CLI 會印出 HTTP 狀態與有長度上限的單行內容預覽。若
health／readiness／models 呼叫或 JSON 轉錄回應格式錯誤，錯誤訊息會包含 HTTP
狀態和端點。CLI 也會在解析或顯示前限制每個 HTTP 回應本文；
`--max-response-mb` 預設為 `16` MiB，處理特別大的 verbose JSON 轉錄時可以提高。

HTTP 請求會直接連線：CLI 會忽略環境中的 `HTTP_PROXY`／`HTTPS_PROXY`，而且絕不
跟隨重新導向，避免已設定的 Bearer 權杖被轉送到其他端點。對等端提供的錯誤文字
會以不受控制字元影響的安全形式顯示、限制在 500 個字元內；若伺服器意外反射已
設定的權杖，CLI 會將它遮蔽。

`--timeout` 預設為 `600` 秒，與 `CAPSWRITER_HTTP_API_TASK_TIMEOUT` 一致。它必須
是正數；參數解析會在傳送任何 HTTP 請求前拒絕無效值。

## 驗證

執行隔離的驗證指令碼：

```bash
python client/cli/scripts/verify.py
```

它會依序執行：

1. `python -m compileall client/cli`
2. `python -m unittest discover -s client/cli/tests -v`
3. `python client/cli/scripts/build_zipapp.py`
4. `python client/cli/dist/capswriter-cli.pyz --help`
5. 使用 stdin 輸入，對封裝版執行 `speak --stdin --dry-run` 冒煙測試
6. `python client/cli/scripts/clean.py`

測試會啟動程序內的模擬 HTTP API，因此不需要真正的模型伺服器。即使先前步驟
失敗，清理步驟仍會移除 `__pycache__` 與 `.pyc` 檔案。

`client/cli/scripts/verify.py` 啟動的每個子程序都受
`CAPSWRITER_CLI_VERIFY_STEP_TIMEOUT` 限制，預設為 `600` 秒。速度特別慢的機器可
設定較大的正值：

```bash
CAPSWRITER_CLI_VERIFY_STEP_TIMEOUT=1200 python client/cli/scripts/verify.py
```

手動清理：

```bash
python client/cli/scripts/clean.py
```

## 實作要點

- Multipart 上傳使用 `urllib.request` 與動態產生的 boundary；本機檔名會先跳脫，
  再寫入 `Content-Disposition` 標頭。
- `--base-url` 接受絕對 `http://` 或 `https://` 根 URL，尾端可以有或沒有 `/v1`。
  它會保留 `https://host/capswriter/v1` 這類路徑前綴。傳送任何請求前，會拒絕
  URL 認證資訊、查詢字串、片段識別碼與非 HTTP 協定。
- HTTP 請求會略過環境代理設定、不跟隨重新導向，並將每個非 2xx 轉錄回應視為
  錯誤，使 Bearer 認證資訊只綁定到明確設定的 API 來源。
- `--key-file` 與 `CAPSWRITER_HTTP_API_KEY_FILE` 會讀取含有非空白 Bearer 權杖的
  UTF-8 檔案；一次性的本機診斷仍可用明確的 `--key` 覆蓋。
- `--timeout` 預設等於伺服器工作逾時（`600` 秒）；驗證為有限正浮點數後，會一致
  套用到 health／readiness／models 與轉錄請求。
- `--max-response-mb` 預設為 `16` MiB；驗證為有限正浮點數後，會在 JSON 解析或
  顯示轉錄前限制每個 HTTP 回應本文。也可用 `CAPSWRITER_CLI_MAX_RESPONSE_MB`
  設定。
- `--output` 與 `--output-dir` 會先在同目錄寫入暫存檔，再以不可分割的替換操作
  寫入轉錄檔；`--output-dir` 會立即寫入每個成功的批次項目。
- `--output-dir` 依回應格式對應輸出副檔名（`.txt`、`.json`、`.srt`、`.vtt`），
  清理產生的主檔名以相容 Linux／Windows，並在開始轉錄前拒絕重複的目標路徑。
- `--language` 與 `--prompt` 會傳送到 HTTP API；後端是否支援仍取決於所選模型。
- HTTP 錯誤會正規化 OpenAI 樣式的 `error.message`、舊式 `detail` 回應、非 JSON
  HTTP 錯誤本文，以及預期為 JSON 的端點所回傳之無效 JSON。顯示的對等端文字
  不受控制字元影響、有長度上限，並會遮蔽權杖。
- `speak` 接受直接文字、透過 `--file` 指定的 UTF-8 檔案，或透過 `--stdin` 讀取
  標準輸入；stdin 模式適合轉錄後接續文字轉語音的 shell 管線。
- `--tts-timeout` 預設為 `120` 秒，會驗證為正浮點數並套用到本機文字轉語音
  子程序。`--dry-run` 只印出所選命令，不會執行。
- Windows 文字轉語音使用 PowerShell `System.Speech`。
- Linux 文字轉語音依序優先使用 `spd-say`、`espeak-ng`、`espeak`。
