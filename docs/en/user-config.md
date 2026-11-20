# Upgrade-safe settings (`user/`)

> [Documentation home](README.md) · [繁體中文](../zh-TW/user-config.md) · English

Release ZIPs never contain a `user/` folder. Keep your own settings there and
you can upgrade by extracting the new ZIP **over the old folder**: program files
are replaced and `user/` is left untouched.

```text
CapsWriter-Offline/
├─ start_client.exe, internal/, core/ …  ← replaced on upgrade
├─ config_client.py, hot.txt, LLM/ …     ← shipped defaults, replaced on upgrade
├─ models/                               ← not in the ZIP, kept
└─ user/                                 ← not in the ZIP, kept
   ├─ README.txt
   ├─ config_client.py                   ← only the client settings you change
   ├─ config_server.py                   ← only the server settings you change (full package)
   ├─ hot.txt, hot-rule.txt              ← your hotwords and regex rules
   ├─ hot-server.txt                     ← server hotwords (full package)
   └─ LLM/                               ← your LLM roles (API keys live here)
```

## How it is created

The first time `start_client.exe` (or `start_server.exe`) runs, it creates
`user/` and fills in what is missing. It never overwrites a file that exists:

| File | First run creates |
|---|---|
| `config_client.py`, `config_server.py` | A commented template that changes nothing |
| `hot.txt`, `hot-rule.txt`, `hot-server.txt` | A copy of the shipped default |
| `LLM/` | A copy of the shipped roles, only if `user/LLM/` does not exist yet |

From then on, edit the files in `user/`. The tray's **hotwords** entry and the
"add hotword" dialog already open and write `user/hot.txt`.

## Settings files hold only your changes

`user/config_client.py` and `user/config_server.py` contain only the values you
want to change. Use the same class and attribute names as the program's
`config_client.py` / `config_server.py`; every other value keeps following the
shipped defaults, including new settings added by later releases.

```python
# user/config_client.py
class ClientConfig:
    addr = '192.168.1.20'
    paste_apps = ['WeiXin.exe', 'Telegram.exe', 'Code.exe']
```

```python
# user/config_server.py
class ServerConfig:
    model_type = 'sensevoice'

class Qwen3ASRGGUFArgs:
    llm_use_gpu = False
```

A misspelled class or attribute is reported at startup and ignored, so a
setting removed in a later release does not stop the program. A file that is
not valid Python stops startup with the file name and the error. Only the
assigned attribute changes: values that the shipped file derives from another
attribute (for example model paths built from `ModelPaths.model_dir`) must be
set explicitly as well.

## Order of precedence

From lowest to highest:

1. The shipped `config_client.py` / `config_server.py`.
2. `user/config_client.py` / `user/config_server.py`.
3. Client daily settings saved by the settings window
   (`%LOCALAPPDATA%\CapsWriter\client-settings.json`, see [daily settings](../settings.md)).
4. Server environment variables (Docker and `CAPSWRITER_*`).

The settings window's **Sources** tab shows `user/config_client.py` for values
that come from your file. LLM roles merge by file name: `user/LLM/<name>.py`
replaces `LLM/<name>.py`, and the remaining shipped roles still load.

`CAPSWRITER_USER_DIR` moves the folder elsewhere (for example to a synced
drive). The Docker server does not use `user/`; configure it with environment
variables and mounted files as described in the [Compose guide](../compose.md).

## Upgrading

- **In place**: quit the client and server, extract the new ZIP over the old
  folder, and start again.
- **New folder**: extract the new ZIP elsewhere, then move `user/` (and
  `models/` for the full package) into it.

### Coming from a release without `user/`

Earlier releases kept your changes in the program files, which an upgrade
overwrites. Extract the new release into a **new folder** and import from the
old one once:

```powershell
.\start_client.exe --import-config "D:\CapsWriter-Offline-old"
```

The import writes only the settings that differ from the new defaults into
`user/config_client.py` and `user/config_server.py`. It copies `hot.txt`,
`hot-rule.txt`, `hot-server.txt` and LLM roles that differ from the shipped
versions. It never overwrites a `user/` file you have already edited (those are
reported as kept) and lists values it cannot convert automatically. The
client-only package skips server settings.
