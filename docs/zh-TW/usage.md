# 使用教學

> [文件首頁](README.md) · 繁體中文 · [English](../en/usage.md) · [專案 README](../../README.zh-TW.md)

這份教學假設 Server 已經啟動（還沒有的話，請先看[開始使用](getting-started.md)）。
內容依「每天會做的事」排列，每一節都可以單獨閱讀。

![CapsWriter v2 架構：桌面、Web、CLI、TUI、OpenAI SDK 五種 Client 連到本機的 CapsWriter Server](../assets/overview.svg)

- [1. 確認 Server 已就緒](#1-確認-server-已就緒)
- [2. 語音打字（桌面 Client）](#2-語音打字桌面-client)
- [3. 設定視窗](#3-設定視窗)
- [4. 托盤選單](#4-托盤選單)
- [5. 影片／音檔轉字幕](#5-影片音檔轉字幕)
- [6. 熱詞、規則替換與自訂短語](#6-熱詞規則替換與自訂短語)
- [7. LLM 角色：翻譯、潤色、問答](#7-llm-角色翻譯潤色問答)
- [8. 日記與錄音存檔](#8-日記與錄音存檔)
- [9. Web Console](#9-web-console)
- [10. CLI](#10-cli)
- [11. TUI](#11-tui)
- [12. 在程式裡呼叫（OpenAI SDK／curl）](#12-在程式裡呼叫openai-sdkcurl)
- [13. 出問題時看哪裡](#13-出問題時看哪裡)

## 1. 確認 Server 已就緒

| Server 跑在 | 怎麼確認 |
|---|---|
| Windows `start_server.exe` | 視窗出現「模型文件检查通过」與「开始服务」 |
| Docker | `docker compose ps` 顯示 `healthy`；或看 `docker compose logs -f capswriter-server` |
| 已啟用 HTTP API | `curl http://127.0.0.1:6017/ready` 回傳 `"status": "ok"` |

`/health` 只代表程序活著；**`/ready` 成功才代表模型已載入、可以送音訊**。第一次
啟動 Docker 會下載 1–2 GB 模型，請耐心等候。

## 2. 語音打字（桌面 Client）

![聽寫四步驟：點進輸入框、按住 CapsLock 說話、放開、文字自動輸入](../assets/dictation-flow.svg)

1. 把游標放在任何可以打字的地方。
2. **按住** CapsLock（或滑鼠側鍵 X2）說話；按住超過 0.3 秒才會開始錄音。
3. **放開**按鍵，Server 辨識後文字會自動輸入到目前視窗。
4. 短按 CapsLock（少於 0.3 秒）仍是原本的大小寫切換，不受影響。

Client 視窗會顯示每一次的錄音長度、轉錄延遲與結果，例如：

```text
任务标识：67ae31e8-4c36-11f1-a3bb-9010576e74da
    录音时长：2.85s
    转录时延：0.17s  热词时延：0.00s
    识别结果：短音频可极速推理
```

### 說話小技巧

| 想要 | 怎麼說／怎麼做 |
|---|---|
| 加標點 | 句首或句尾說「逗號」「句號」「回車」，會轉成對應符號或換行 |
| 數字 | 直接說「一千八」「百分之二十」「三點五」，會自動轉成阿拉伯數字 |
| 短句不要句號 | 預設 8 個詞以內會去掉句尾的「，。」；在 `config_client.py` 調整 `trash_punc_thresh` |
| 長篇口述 | 可以一直按著；Client 每 60 秒自動切段（重疊 4 秒），Server 會去重拼接 |
| 單擊開始、再單擊停止 | 在設定視窗把該快捷鍵的「按住錄音」取消勾選 |
| 在管理員權限的程式裡輸入 | Client 也要以系統管理員身分執行 |
| 聊天軟體用貼上而非打字 | 設定視窗「輸出」頁勾選「使用剪貼簿貼上」，或把程式名加入 `paste_apps` |

> [!NOTE]
> Linux 只支援 X11 全域快捷鍵，而且 X11 無法只攔截單一按鍵，所以 CapsLock 仍會
> 切換大小寫；建議改用 F12 或滑鼠側鍵。Wayland 與 headless 環境不支援全域快捷鍵。

## 3. 設定視窗

執行 `start_client.exe --settings`（source 版為 `python scripts/dev.py client --settings`），
或在托盤右鍵選 **設定**。所有變更在**重新啟動 Client 後**生效。

| 連線 | 錄音與快捷鍵 | 輸出 |
|---|---|---|
| ![「連線」頁：Server 主機、WebSocket 連接埠 6016、測試 Server 連線、辨識語言、提示詞](../assets/desktop-settings-connection.png) | ![「錄音與快捷鍵」頁：麥克風、長按觸發時間、保留錄音檔、快捷鍵表格與編輯列](../assets/desktop-settings-recording.png) | ![「輸出」頁：剪貼簿貼上、還原剪貼簿、轉為繁體中文、繁體地區、短句去標點門檻、LLM 潤色](../assets/desktop-settings-output.png) |

<sub>同一個 Tk 設定視窗在 Linux（Xvfb）上的實際渲染；Windows 會使用系統原生外觀與字型。</sub>

- **連線**：Server 在同一台電腦填 `127.0.0.1`；在 NAS／其他電腦填它的 IP 或主機名稱，
  不要加 `ws://`。按「測試 Server 連線」只確認 WebSocket 連得上，不代表模型已就緒。
- **錄音與快捷鍵**：麥克風留空＝系統預設。快捷鍵可以是鍵盤（`caps_lock`、`f12`……）
  或滑鼠（`x1`、`x2`）；「阻擋原按鍵」勾選後，短按會自動補送原本的按鍵。
- **輸出**：想要繁體中文就勾選「轉為繁體中文」，地區可選 `zh-hant`、`zh-tw`、`zh-hk`。
- **設定來源**：顯示設定檔路徑；「恢復 Python 設定」會清除所有覆寫，回到 `config_client.py`。

設定存在 `%LOCALAPPDATA%\CapsWriter\client-settings.json`，只記錄你改過的欄位。
更多進階選項（UDP 廣播、檔案分段長度等）請直接編輯 [`config_client.py`](../../config_client.py)。
細節見[日常設定](../settings.md)。

## 4. 托盤選單

右鍵托盤圖示：

| 選單 | 用途 |
|---|---|
| 顯示／隱藏 | 顯示或隱藏 Client 主控台視窗（雙擊圖示也可以） |
| **設定** | 開啟上一節的設定視窗 |
| 複製結果 | 把最後一次辨識結果放進剪貼簿 |
| 日記 | 用檔案總管打開本月的日記資料夾 |
| 上下文 | 編輯送給模型的提示詞（人名、術語），幫助辨識 |
| 熱詞 | 直接開啟 `hot.txt` |
| 清除記憶 | 清除所有 LLM 角色的對話歷史 |
| 重開音訊 | 換了耳機／麥克風後重新初始化錄音裝置 |
| 重啟／退出 | 重新啟動或結束 Client |

## 5. 影片／音檔轉字幕

需要 `ffmpeg.exe`（與 `ffprobe.exe`）在 `PATH` 或 CapsWriter 資料夾內。

1. 把一個或多個檔案（mp3、wav、mp4、mkv、mov……）**拖到 `start_client.exe` 圖示上**。
2. Client 會顯示進度；完成後在**原檔案旁邊**產生：

| 檔案 | 內容 |
|---|---|
| `影片.srt` | 以句為單位的字幕 |
| `影片.txt` | 依標點分行的純文字 |
| `影片.json` | 字級時間戳 |
| `影片.merge.txt` | 整段不分行的長文（預設關閉，`file_save_merge = True` 開啟） |

**修正字幕錯字**：直接改 `影片.txt` 的錯字或分行，存檔後再把這個 `.txt` 拖回 Client，
會利用 `.json` 的時間戳重新對齊，產生新的 `.srt`。

在 NAS／伺服器上批次處理時，改用 [CLI](#10-cli) 或 [Web Console](#9-web-console)
會更方便。詳見 upstream 的[文件轉錄說明](../文件转录功能如何使用.md)。

## 6. 熱詞、規則替換與自訂短語

三個文字檔都在 CapsWriter 根目錄，存檔後約 3 秒自動重新載入，不需重啟。

### `hot.txt`：專有名詞（Client 端，強制替換）

依**發音**模糊比對，發音夠像就替換成第一個詞。用 `|` 列出別名，`~~~` 後面是例外詞：

```text
# 人名、產品名、術語
CapsWriter | Caps Rider
Claude | 克勞德 | Cloud ~~~ Weather | Sky
Qwen3-ASR | 千問 ASR
```

也能當「語音巨集」——第一個詞寫你想輸入的內容，後面寫觸發說法：

```text
hello@example.com | 我的信箱 | input my email
```

### `hot-rule.txt`：規則替換（正則表達式）

每行 `查找 = 替換`，等號兩邊要有空格：

```text
毫安時   =  mAh
(艾特)\s*(\w+)\s*(點)\s*(\w+)   =   @\2.\4
```

### `hot-server.txt`：Server 端熱詞

給 Fun-ASR-Nano 等支援熱詞的模型當「提示」，只會影響辨識傾向、不會強制替換。
Docker 會把這個檔案唯讀掛進容器。

完整格式與參數見[熱詞功能如何使用](../热词功能如何使用.md)。

## 7. LLM 角色：翻譯、潤色、問答

LLM 是**選用**功能，辨識本身仍完全離線。角色定義在 [`LLM/`](../../LLM/) 資料夾，
每個 `.py` 是一個角色，修改後自動重新載入。

| 角色檔 | 觸發方式 | 輸出 |
|---|---|---|
| `default.py` | 所有沒有前綴的句子 | 直接打字（潤色），預設關閉 |
| `翻译.py` | 句子以「翻譯」開頭 | Toast 彈窗 |
| `小助理.py` | 以「小助理」開頭 | Toast 彈窗 |
| `大助理.py` | 以「大助理」開頭 | Toast 彈窗，預設關閉 |

使用方式：

1. 打開角色檔，設定 `provider`（`ollama`、`lmstudio`、`openai`、`deepseek`……）、
   `model`，線上服務再填 `api_key`，把 `enabled` 改成 `True`。
2. 先用滑鼠選取一段文字，再按住 CapsLock 說「翻譯這段」，結果會顯示在 Toast 彈窗。
3. LLM 正在輸出時按 `Esc` 可以中斷；托盤「清除記憶」可清掉對話歷史。

> [!WARNING]
> 使用線上 LLM 時，辨識文字與（若啟用）選取的文字會送到該服務。重視隱私請改用
> 本機 Ollama／LM Studio。

詳見[角色功能如何使用](../角色功能如何使用.md)。

## 8. 日記與錄音存檔

每次辨識都會依日期寫入 `年/月/日.md`，錄音存在 `年/月/assets/`（有 FFmpeg 時壓成
mp3），Markdown 內附播放控制項。托盤 **日記** 可直接打開本月資料夾。不想保存錄音，
在設定視窗取消「保留錄音檔」。

## 9. Web Console

Web Console 是瀏覽器裡的轉錄工作台，需要 Server 已啟用 HTTP API（見
[OpenAI 相容 API](openai-api.md#啟用端點)）。

![Web Console 實機截圖：左欄連線設定與 Server 診斷（Health、Ready、Router、FFmpeg 皆為 ok），中欄已上傳 zh.wav 並轉錄出「開放時間：早上九點至下午五點。」，右欄為 TTS 與歷史](../assets/web-console.png)

<sub>真實截圖：連到本機 Qwen3-ASR 1.7B（CPU）Server，轉錄 5.6 秒的中文測試音檔。</sub>

1. 開啟 `http://127.0.0.1:8080`（或開發模式的 `:5173`）。
2. 在連線設定填 API 位址（例如 `http://127.0.0.1:6017`）與 API key，按「檢查服務」。
3. 按錄音按鈕說話，或把音訊檔拖進上傳區。
4. 選擇輸出格式（`text`、`json`、`verbose_json`、`srt`、`vtt`），完成後可複製或下載。
5. 「朗讀」使用瀏覽器本機的 TTS，不會送到 Server。

麥克風需要 `localhost` 或 HTTPS。部署、CORS 與安全注意事項見 [Web Console 指南](web-console.md)。

## 10. CLI

![CLI 實機輸出：health 回報 qwen_asr、中文音檔轉成文字、英文音檔轉成 SRT、兩個檔案批次輸出 VTT](../assets/cli-demo.svg)

```bash
export CAPSWRITER_API_BASE=http://127.0.0.1:6017
export CAPSWRITER_HTTP_API_KEY=replace-with-a-long-random-token

python client/cli/capswriter_cli.py ready                        # 檢查 Server
python client/cli/capswriter_cli.py transcribe talk.wav          # 印出文字
python client/cli/capswriter_cli.py transcribe talk.mp4 --format srt --output talk.srt
python client/cli/capswriter_cli.py transcribe audio/*.m4a --format vtt --output-dir subs/
python client/cli/capswriter_cli.py transcribe talk.wav --format text | \
  python client/cli/capswriter_cli.py speak --stdin              # 轉錄後朗讀
```

也可以打包成單一檔案：`python client/cli/scripts/build_zipapp.py`，產生
`client/cli/dist/capswriter-cli.pyz`，複製到任何有 Python 3.10+ 的電腦即可使用。
詳見 [CLI 指南](cli-client.md)。

## 11. TUI

![真實 CapsWriter TUI：上方 Server 診斷全部正常，左下為 zh.wav 與參數，右下為轉錄結果「開放時間：早上九點至下午五點。」](../assets/tui-transcript.svg)

| 按鍵 | 動作 |
|---|---|
| `F5` | 檢查 health、readiness 與模型 |
| `Ctrl+O` | 跳到音訊路徑欄位 |
| `F8` | 開始錄音；錄音中再按一次停止（需選用麥克風套件） |
| `F9` | 取消錄音並刪除暫存檔 |
| `Ctrl+T` | 開始轉錄 |
| `Esc` | 取消進行中的工作 |
| `Ctrl+S` | 儲存結果 |
| `Ctrl+L` | 切換 English／繁體中文 |
| `Ctrl+Q` | 離開 |

安裝與限制見 [TUI 指南](tui.md)。

## 12. 在程式裡呼叫（OpenAI SDK／curl）

```bash
curl http://127.0.0.1:6017/v1/audio/transcriptions \
  -H "Authorization: Bearer replace-with-a-long-random-token" \
  -F model=whisper-1 -F response_format=srt -F file=@talk.wav
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:6017/v1", api_key="replace-with-a-long-random-token")
with open("talk.wav", "rb") as f:
    result = client.audio.transcriptions.create(
        model="whisper-1", file=f, response_format="verbose_json", language="zh",
    )
print(result.text)
```

`model` 固定填 `whisper-1`（相容用 ID），實際模型由 Server 決定。支援範圍、
上傳上限與錯誤格式見 [OpenAI 相容 API](openai-api.md)。

## 13. 出問題時看哪裡

1. **先看 Server**：`/ready`、Server 視窗或 `docker compose logs`。
2. **再看 Client**：`logs/client_latest.log`；Server 端是 `logs/server_latest.log`。
3. 依症狀查[疑難排解](troubleshooting.md)與 upstream [常見問題](../常见问题.md)。

回報問題時請附上版本、作業系統、模型、硬體與相關日誌片段；**不要**附上 API key
或私人錄音。見[支援與安全](support-security.md)。
