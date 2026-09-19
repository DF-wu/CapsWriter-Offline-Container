"""Validated, per-user settings layered over the advanced Python configuration.

Importing this module never initializes audio, keyboard hooks, or model runtimes.
Only explicit overrides are persisted; all other settings follow config_client.py.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import tempfile

from .devices import valid_input_device


FIELDS = {
    "addr", "port", "shortcuts", "threshold", "input_device", "paste",
    "restore_clip", "traditional_convert", "traditional_locale", "save_audio",
    "trash_punc_thresh", "context", "language", "llm_enabled",
}
BOOLEAN_FIELDS = {"paste", "restore_clip", "traditional_convert", "save_audio", "llm_enabled"}
KEYBOARD_KEYS = {
    "ctrl_l", "ctrl_r", "shift", "shift_r", "alt_l", "alt_gr", "cmd", "cmd_r",
    "space", "enter", "tab", "backspace", "delete", "insert", "home", "end",
    "page_up", "page_down", "esc", "caps_lock", "num_lock", "scroll_lock",
    "print_screen", "pause", "menu", "up", "down", "left", "right",
} | {f"f{i}" for i in range(1, 25)}


class SettingsError(ValueError):
    def __init__(self, errors: dict[str, str]):
        self.errors = errors
        super().__init__("；".join(f"{key}: {value}" for key, value in errors.items()))


def settings_path() -> Path:
    explicit = os.environ.get("CAPSWRITER_CLIENT_SETTINGS")
    if explicit:
        return Path(explicit).expanduser().resolve()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    return base / "CapsWriter" / "client-settings.json"


def _valid_host(value: object) -> bool:
    if not isinstance(value, str) or not value or value != value.strip():
        return False
    bracketed = value.startswith("[") and value.endswith("]")
    host = value[1:-1] if bracketed else value
    try:
        address = ipaddress.ip_address(host)
        return not bracketed or address.version == 6
    except ValueError:
        return not bracketed and len(host) <= 253 and all(
            re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", part)
            for part in host.rstrip(".").split(".")
        )


def validate(overrides: object) -> dict:
    if not isinstance(overrides, dict):
        raise SettingsError({"settings": "設定必須是 JSON 物件"})
    errors = {}
    normalized = copy.deepcopy(overrides)
    for key, value in overrides.items():
        if key not in FIELDS:
            errors[key] = "不支援此日常設定；進階參數請保留於 config_client.py"
        elif key in BOOLEAN_FIELDS and type(value) is not bool:
            errors[key] = "請使用 true 或 false"
        elif key == "addr" and not _valid_host(value):
            errors[key] = "請填主機名稱或 IP，例如 axolotl；不要包含 ws://、連接埠或路徑"
        elif key == "port":
            if (type(value) not in (int, str) or not str(value).isascii()
                    or not 1 <= len(str(value)) <= 5
                    or not str(value).isdigit() or not 1 <= int(value) <= 65535):
                errors[key] = "連接埠必須是 1–65535 的整數"
            else:
                normalized[key] = str(value)
        elif key == "threshold":
            if type(value) not in (int, float) or not 0 <= value <= 5 or not math.isfinite(value):
                errors[key] = "請填 0–5 秒的數字"
        elif key == "trash_punc_thresh":
            if type(value) is not int or not 0 <= value <= 1000:
                errors[key] = "請填 0–1000 的整數；0 表示不依短句移除標點"
        elif key == "input_device":
            if not valid_input_device(value):
                errors[key] = "請選擇麥克風名稱與音訊介面；null 使用系統預設裝置"
        elif key == "traditional_locale" and value not in ("zh-hant", "zh-tw", "zh-hk"):
            errors[key] = "請選 zh-hant、zh-tw 或 zh-hk"
        elif key in ("context", "language"):
            if not isinstance(value, str) or len(value) > (4000 if key == "context" else 50):
                errors[key] = "文字過長或格式錯誤"
            elif key == "language" and not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", value):
                errors[key] = "請填語言代碼，例如 auto、chinese、english"
        elif key == "shortcuts":
            if not isinstance(value, list) or not 1 <= len(value) <= 16:
                errors[key] = "請設定 1–16 個快捷鍵"
                continue
            seen = set()
            for shortcut in value:
                if not isinstance(shortcut, dict) or set(shortcut) != {"key", "type", "suppress", "hold_mode", "enabled"}:
                    errors[key] = "每個快捷鍵需包含 key、type、suppress、hold_mode、enabled"
                    break
                button, kind = shortcut["key"], shortcut["type"]
                if not isinstance(button, str) or kind not in ("keyboard", "mouse"):
                    errors[key] = "快捷鍵類型或名稱無效"
                    break
                valid = button in ("x1", "x2") if kind == "mouse" else (
                    button in KEYBOARD_KEYS or len(button) == 1 and button.isprintable() and not button.isspace()
                )
                if not valid or any(type(shortcut[k]) is not bool for k in ("suppress", "hold_mode", "enabled")):
                    errors[key] = "使用單一按鍵，例如 caps_lock、f12，或滑鼠 x1／x2；開關需為布林值"
                    break
                identity = (kind, button)
                if identity in seen:
                    errors[key] = "快捷鍵不可重複"
                    break
                seen.add(identity)
            if not errors.get(key) and not any(item["enabled"] for item in value):
                errors[key] = "至少需啟用一個快捷鍵"
    if errors:
        raise SettingsError(errors)
    return normalized


def load_overrides(path: Path | None = None) -> dict:
    path = settings_path() if path is None else Path(path)
    if not path.exists():
        return {}
    try:
        if path.stat().st_size > 128 * 1024:
            raise ValueError("設定檔不可超過 128 KiB")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or set(data) != {"version", "overrides"} or type(data["version"]) is not int or data["version"] != 1:
            raise ValueError("不支援的設定格式或版本；預期 version: 1")
        return validate(data["overrides"])
    except SettingsError:
        raise
    except (OSError, ValueError) as exc:
        raise SettingsError({"settings": f"無法讀取 {path}：{exc}"}) from exc


@contextmanager
def _settings_lock(path: Path):
    """Use a persistent sidecar: replacing/deleting a locked inode breaks exclusion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt

            if lock.seek(0, os.SEEK_END) == 0:
                lock.write(b"\0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def save_overrides(overrides: dict, path: Path | None = None) -> None:
    """Replace all overrides (including explicit reset) under the writer lock."""
    normalized = validate(overrides)
    path = settings_path() if path is None else Path(path)
    with _settings_lock(path):
        _write_overrides(normalized, path)


def update_overrides(changes: dict, path: Path | None = None) -> dict:
    """Merge only edited fields into the latest file; last edit to a field wins."""
    normalized = validate(changes)
    path = settings_path() if path is None else Path(path)
    with _settings_lock(path):
        current = load_overrides(path)
        current.update(normalized)
        _write_overrides(current, path)
    return current


def _write_overrides(normalized: dict, path: Path) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            json.dump({"version": 1, "overrides": normalized}, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def effective_settings(config, overrides: dict) -> dict:
    base = {key: copy.deepcopy(getattr(config, key, None)) for key in sorted(FIELDS)}
    base.update(validate(overrides))
    return base


def apply_overrides(config, path: Path | None = None) -> dict:
    overrides = load_overrides(path)
    for key, value in overrides.items():
        setattr(config, key, copy.deepcopy(value))
    return overrides


def websocket_url(addr: str, port: str | int) -> str:
    values = validate({"addr": addr, "port": port})
    host = values["addr"]
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"ws://{host}:{values['port']}"


async def probe_connection(addr: str, port: str | int) -> None:
    """Verify a WebSocket handshake, without sending any microphone data."""
    import asyncio
    import inspect
    import websockets

    async def connect():
        kwargs = {"open_timeout": 4, "close_timeout": 1, "subprotocols": ["binary"], "max_size": 1024}
        if "proxy" in inspect.signature(websockets.connect).parameters:
            kwargs["proxy"] = None
        async with websockets.connect(websocket_url(addr, port), **kwargs):
            pass

    await asyncio.wait_for(connect(), timeout=6)
