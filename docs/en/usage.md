# Usage guide

> [Docs home](README.md) · English · [繁體中文](../zh-TW/usage.md) · [Project README](../../README.en.md)

This guide assumes a server is already running (if not, start with
[Getting started](getting-started.md)). It is organised by everyday tasks; every
section stands on its own. Screenshots show the Traditional Chinese UI; the text
explains each label.

![CapsWriter v2 architecture: desktop, Web, CLI, TUI and OpenAI SDK clients connect to a local CapsWriter Server](../assets/overview.svg)

- [1. Check that the server is ready](#1-check-that-the-server-is-ready)
- [2. Dictation (desktop client)](#2-dictation-desktop-client)
- [3. Settings window](#3-settings-window)
- [4. Tray menu](#4-tray-menu)
- [5. Audio and video to subtitles](#5-audio-and-video-to-subtitles)
- [6. Hotwords, rules and voice snippets](#6-hotwords-rules-and-voice-snippets)
- [7. LLM roles: translate, polish, ask](#7-llm-roles-translate-polish-ask)
- [8. Diary and saved recordings](#8-diary-and-saved-recordings)
- [9. Web Console](#9-web-console)
- [10. CLI](#10-cli)
- [11. TUI](#11-tui)
- [12. From code: OpenAI SDK and curl](#12-from-code-openai-sdk-and-curl)
- [13. When something goes wrong](#13-when-something-goes-wrong)

## 1. Check that the server is ready

| Server runs as | How to check |
|---|---|
| Windows `start_server.exe` | The window shows "模型文件检查通过" (model files OK) and "开始服务" (serving) |
| Docker | `docker compose ps` shows `healthy`, or follow `docker compose logs -f capswriter-server` |
| HTTP API enabled | `curl http://127.0.0.1:6017/ready` returns `"status": "ok"` |

`/health` only means the process is alive; **only `/ready` means the model is loaded
and audio can be sent**. The first Docker start downloads 1–2 GB of models.

## 2. Dictation (desktop client)

![Four dictation steps: click into a text field, hold CapsLock and speak, release, text is typed](../assets/dictation-flow.svg)

1. Put the cursor anywhere you can type.
2. **Hold** CapsLock (or mouse side button X2) and speak; recording starts after 0.3 s.
3. **Release** the key; the server recognises the audio and the text is typed into the
   active window.
4. A short CapsLock press (under 0.3 s) still toggles caps as usual.

The client console shows duration, latency and result for every recording:

```text
任务标识：67ae31e8-4c36-11f1-a3bb-9010576e74da   (task id)
    录音时长：2.85s                               (recording length)
    转录时延：0.17s  热词时延：0.00s              (ASR / hotword latency)
    识别结果：短音频可极速推理                    (result)
```

### Tips

| For | Do this |
|---|---|
| Punctuation | Say "逗号 / 句号 / 回车" (comma / full stop / newline) at the start or end of a sentence |
| Numbers | Speak them naturally ("一千八", "百分之二十"); inverse text normalisation writes digits |
| No trailing full stop on short phrases | Phrases of up to 8 words drop a trailing "，。" by default; tune `trash_punc_thresh` in `config_client.py` |
| Long dictation | Keep holding; the client cuts 60 s segments with 4 s overlap and the server merges them |
| Click to start, click to stop | Untick "hold to record" (按住錄音) for that shortcut in the settings window |
| Typing into elevated programs | Run the client as administrator too |
| Paste instead of typing in chat apps | Settings → Output → use clipboard paste, or add the program to `paste_apps` |

> [!NOTE]
> Linux supports global hotkeys on X11 only, and X11 cannot suppress a single key, so
> CapsLock will still toggle; F12 or a mouse side button works better. Wayland and
> headless sessions are unsupported for global hotkeys.

## 3. Settings window

Run `start_client.exe --settings` (from source: `python scripts/dev.py client --settings`)
or choose **設定 (Settings)** in the tray menu. Changes apply after **restarting the client**.

| Connection (連線) | Recording & shortcuts (錄音與快捷鍵) | Output (輸出) |
|---|---|---|
| ![Connection tab: server host, WebSocket port 6016, connection test, recognition language, prompt](../assets/desktop-settings-connection.png) | ![Recording tab: microphone, hold threshold, keep recordings, shortcut table and editor](../assets/desktop-settings-recording.png) | ![Output tab: clipboard paste, restore clipboard, Traditional Chinese conversion, locale, short-phrase punctuation threshold, LLM polishing](../assets/desktop-settings-output.png) |

<sub>Real rendering of the same Tk settings window on Linux (Xvfb); Windows uses its native theme and fonts.</sub>

- **Connection**: `127.0.0.1` when the server is on this PC; otherwise its IP or hostname,
  without `ws://`. The connection test only proves the WebSocket handshake, not model readiness.
- **Recording & shortcuts**: an empty microphone means the system default. Shortcuts can be
  keys (`caps_lock`, `f12`, …) or mouse buttons (`x1`, `x2`); with "block original key"
  (阻擋原按鍵) a short press re-sends the original key.
- **Output**: tick "convert to Traditional Chinese" (轉為繁體中文) and choose `zh-hant`,
  `zh-tw` or `zh-hk`.
- **Settings source** (設定來源): shows the settings file; "restore Python settings" clears
  every override and falls back to `config_client.py`.

Settings are stored in `%LOCALAPPDATA%\CapsWriter\client-settings.json` and contain only
the fields you changed. For advanced options (UDP broadcast, file segment length, …)
edit [`config_client.py`](../../config_client.py). See [daily settings](../settings.md).

## 4. Tray menu

| Item | Purpose |
|---|---|
| 显示/隐藏 (show/hide) | Toggle the client console (double-click the icon also works) |
| **設定 (Settings)** | Open the settings window above |
| 复制结果 (copy result) | Copy the last recognition result |
| 日记 (diary) | Open this month's diary folder |
| 上下文 (context) | Edit the prompt sent to the model (names, jargon) |
| 热词 (hotwords) | Open `hot.txt` |
| 清除记忆 (clear memory) | Clear every LLM role's conversation history |
| 重开音频 (restart audio) | Re-initialise the recording device after switching headsets |
| 重启 / 退出 (restart / exit) | Restart or quit the client |

## 5. Audio and video to subtitles

Requires `ffmpeg.exe` (and `ffprobe.exe`) on `PATH` or in the CapsWriter folder.

1. **Drop** one or more files (mp3, wav, mp4, mkv, mov, …) **onto the `start_client.exe` icon**.
2. The client shows progress and writes, **next to the original file**:

| File | Content |
|---|---|
| `video.srt` | Sentence-level subtitles |
| `video.txt` | Plain text split at punctuation |
| `video.json` | Character-level timestamps |
| `video.merge.txt` | One long paragraph (off by default; `file_save_merge = True`) |

**Fixing subtitle typos**: edit `video.txt`, save, and drop the `.txt` back on the
client — it re-aligns the text with the timestamps in `.json` and writes a new `.srt`.

For batch work on a NAS or server, the [CLI](#10-cli) or [Web Console](#9-web-console)
is more convenient.

## 6. Hotwords, rules and voice snippets

All three files live in the CapsWriter root and reload about 3 s after saving.

### `hot.txt`: names and jargon (client side, forced)

Matched by **pronunciation**; close enough means it is replaced by the first word.
List aliases with `|`; words after `~~~` are exceptions:

```text
# people, products, terms
CapsWriter | Caps Rider
Claude | 克劳德 | Cloud ~~~ Weather | Sky
```

It also works as a voice macro — put what you want typed first, then the trigger phrases:

```text
hello@example.com | 我的邮箱 | input my email
```

### `hot-rule.txt`: regex rules

One `pattern = replacement` per line, with spaces around the `=`:

```text
毫安时   =  mAh
(艾特)\s*(\w+)\s*(点)\s*(\w+)   =   @\2.\4
```

### `hot-server.txt`: server hotwords

Hints for models that support them (such as Fun-ASR-Nano); they bias recognition but
never force a replacement. Docker mounts the file read-only.

Full syntax and thresholds: upstream [hotword guide](../热词功能如何使用.md) (Simplified Chinese).

## 7. LLM roles: translate, polish, ask

LLM use is **optional**; recognition itself stays offline. Roles live in
[`LLM/`](../../LLM/), one `.py` file per role, reloaded automatically.

| Role file | Triggered by | Output |
|---|---|---|
| `default.py` | Every sentence without a prefix | Typed (polishing), off by default |
| `翻译.py` | Sentences starting with "翻译" (translate) | Toast pop-up |
| `小助理.py` | Starting with "小助理" (small assistant) | Toast pop-up |
| `大助理.py` | Starting with "大助理" (big assistant) | Toast pop-up, off by default |

1. Open the role file, set `provider` (`ollama`, `lmstudio`, `openai`, `deepseek`, …) and
   `model`, add `api_key` for online services, and set `enabled = True`.
2. Select some text, hold CapsLock and say "翻译这段" (translate this); the answer appears in a toast.
3. Press `Esc` to stop LLM output; tray "clear memory" wipes the conversation history.

> [!WARNING]
> Online LLM providers receive the recognised text and, if enabled, the selected text.
> Use local Ollama / LM Studio when privacy matters.

See the upstream [role guide](../角色功能如何使用.md) (Simplified Chinese).

## 8. Diary and saved recordings

Each recognition is appended to `year/month/day.md`; recordings are stored in
`year/month/assets/` (as mp3 when FFmpeg is available) with an audio player link in the
Markdown. Tray **diary** opens the current month. Untick "keep recordings" (保留錄音檔)
in settings to stop saving audio.

## 9. Web Console

A browser transcription workbench; the server needs the HTTP API enabled (see the
[OpenAI-compatible API](openai-api.md)).

![Real Web Console: connection settings and server diagnostics (Health, Ready, Router, FFmpeg all ok) on the left, an uploaded zh.wav transcribed to 開放時間：早上九點至下午五點。 in the middle, TTS and history on the right](../assets/web-console.png)

<sub>Real capture against a local Qwen3-ASR 1.7B (CPU) server transcribing a 5.6-second Chinese test clip.</sub>

1. Open `http://127.0.0.1:8080` (or `:5173` in development mode).
2. Enter the API root (e.g. `http://127.0.0.1:6017`) and API key, then check the service.
3. Press record and speak, or drop an audio file on the upload area.
4. Pick a format (`text`, `json`, `verbose_json`, `srt`, `vtt`) and copy or download the result.
5. Read-aloud uses the browser's local TTS; nothing is sent to the server.

The microphone requires `localhost` or HTTPS. Deployment, CORS and security are in the
[Web Console guide](web-console.md).

## 10. CLI

![Real CLI output: health reports qwen_asr, a Chinese clip as text, an English clip as SRT, and a two-file VTT batch](../assets/cli-demo.svg)

```bash
export CAPSWRITER_API_BASE=http://127.0.0.1:6017
export CAPSWRITER_HTTP_API_KEY=replace-with-a-long-random-token

python client/cli/capswriter_cli.py ready                        # check the server
python client/cli/capswriter_cli.py transcribe talk.wav          # print text
python client/cli/capswriter_cli.py transcribe talk.mp4 --format srt --output talk.srt
python client/cli/capswriter_cli.py transcribe audio/*.m4a --format vtt --output-dir subs/
python client/cli/capswriter_cli.py transcribe talk.wav --format text | \
  python client/cli/capswriter_cli.py speak --stdin              # transcribe, then read aloud
```

Build a single-file version with `python client/cli/scripts/build_zipapp.py`; copy
`client/cli/dist/capswriter-cli.pyz` to any machine with Python 3.10+. See the
[CLI guide](cli-client.md).

## 11. TUI

![Real CapsWriter TUI: server diagnostics all OK at the top, zh.wav and options bottom-left, the transcript 開放時間：早上九點至下午五點。 bottom-right](../assets/tui-transcript.svg)

| Key | Action |
|---|---|
| `F5` | Refresh health, readiness and models |
| `Ctrl+O` | Focus the audio path field |
| `F8` | Start recording; press again to stop (optional microphone stack) |
| `F9` | Cancel recording and delete the temporary file |
| `Ctrl+T` | Transcribe |
| `Esc` | Cancel the running job |
| `Ctrl+S` | Save the result |
| `Ctrl+L` | Switch English / Traditional Chinese |
| `Ctrl+Q` | Quit |

Installation and limits: [TUI guide](tui.md).

## 12. From code: OpenAI SDK and curl

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

`model` is always `whisper-1` (a compatibility ID); the server decides the real model.
Scope, upload limits and error format: [OpenAI-compatible API](openai-api.md).

## 13. When something goes wrong

1. **Server first**: `/ready`, the server window or `docker compose logs`.
2. **Then the client**: `logs/client_latest.log`; the server writes `logs/server_latest.log`.
3. Look up the symptom in [troubleshooting](troubleshooting.md).

When reporting a problem, include version, OS, model, hardware and relevant log lines —
**never** API keys or private recordings. See [support and security](support-security.md).
