"""Validated, restart-only daily settings; environment remains the final authority."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import threading
from collections.abc import Mapping

from fork_server.http_api.runtime_config import ConfigError, parse_api_key, parse_bool

MAX_SETTINGS_BYTES = 65_536


@dataclass(frozen=True)
class Field:
    key: str
    env: str
    label: str
    description: str
    type: str
    default: object
    minimum: int | float | None = None
    maximum: int | float | None = None
    choices: tuple[str, ...] = ()

    def validate(self, value):
        if self.type == "boolean":
            valid = isinstance(value, bool)
        elif self.type == "integer":
            valid = isinstance(value, int) and not isinstance(value, bool)
        elif self.type == "number":
            valid = (isinstance(value, (int, float)) and not isinstance(value, bool)
                     and (not isinstance(value, float) or math.isfinite(value)))
        else:
            valid = isinstance(value, str) and value in self.choices
        if not valid:
            raise ConfigError(f"{self.key}: invalid {self.type} value")
        if self.minimum is not None and value < self.minimum:
            raise ConfigError(f"{self.key}: must be >= {self.minimum}")
        if self.maximum is not None and value > self.maximum:
            raise ConfigError(f"{self.key}: must be <= {self.maximum}")
        return value

    def from_env(self, raw):
        try:
            if self.type == "boolean":
                return parse_bool({self.env: raw}, self.env, False)
            if self.type == "integer":
                return self.validate(int(raw))
            if self.type == "number":
                return self.validate(float(raw))
            return self.validate(raw.strip().lower())
        except (ValueError, TypeError) as exc:
            raise ConfigError(f"{self.env}: invalid value") from exc


FIELDS = (
    Field("inference_hardware", "CAPSWRITER_INFERENCE_HARDWARE", "容器運算硬體", "僅適用 Docker：auto 自動偵測並允許 CPU 回退，cpu 使用 CPU，gpu 優先嘗試 GPU，失敗時回退 CPU。", "string", "auto", choices=("auto", "cpu", "gpu")),
    Field("model_type", "CAPSWRITER_MODEL_TYPE", "辨識模型", "切換前請先準備該模型檔案。", "string", "qwen_asr", choices=("qwen_asr", "fun_asr_nano", "sensevoice", "paraformer")),
    Field("qwen_preset", "CAPSWRITER_QWEN_PRESET", "Qwen 運算模式", "僅適用 Qwen；進階 CUDA / Vulkan 環境變數仍優先。", "string", "default", choices=("default", "low_vram_gpu", "cpu_only")),
    Field("num_threads", "CAPSWRITER_NUM_THREADS", "CPU 執行緒", "適用 Qwen 與 Fun-ASR 的生成及批次處理；未設定時由模型自動決定。", "integer", None, 1, 256),
    Field("format_num", "CAPSWRITER_FORMAT_NUM", "數字格式化", "將辨識結果的中文數字轉為阿拉伯數字。", "boolean", True),
    Field("format_spell", "CAPSWRITER_FORMAT_SPELL", "中英空格", "調整中英文之間的空格。", "boolean", True),
    Field("max_upload_mb", "CAPSWRITER_HTTP_API_MAX_UPLOAD_MB", "檔案上限 (MB)", "HTTP 單次上傳的大小上限。", "integer", 100, 1, 1024),
    Field("max_audio_seconds", "CAPSWRITER_HTTP_API_MAX_AUDIO_SECONDS", "音訊上限 (秒)", "HTTP 單次音訊長度上限。", "number", 3600.0, 1, 14400),
    Field("task_timeout", "CAPSWRITER_HTTP_API_TASK_TIMEOUT", "HTTP 任務逾時 (秒)", "等待辨識完成的最長時間。", "number", 600.0, 1, 86400),
    Field("max_concurrent_requests", "CAPSWRITER_HTTP_API_MAX_CONCURRENT_REQUESTS", "HTTP 同時處理數", "含解碼及等待辨識；辨識工作仍依序執行。", "integer", 2, 1, 64),
    Field("max_pending_requests", "CAPSWRITER_HTTP_API_MAX_PENDING_REQUESTS", "HTTP 等待佇列", "繁忙時允許等待的請求數；0 表示不排隊。", "integer", 4, 0, 1024),
    Field("max_websocket_connections", "CAPSWRITER_SERVER_MAX_WEBSOCKET_CONNECTIONS", "語音輸入連線數", "同時允許的 WebSocket 連線上限。", "integer", 8, 1, 1024),
    Field("max_websocket_task_seconds", "CAPSWRITER_SERVER_MAX_WEBSOCKET_TASK_SECONDS", "語音輸入上限 (秒)", "WebSocket 單次錄音任務長度上限。", "number", 3600.0, 1, 86400),
)
FIELD_MAP = {field.key: field for field in FIELDS}


class RevisionConflict(Exception):
    """The file changed since the caller last read it."""


def decode_settings_json(raw):
    """Reject non-JSON numeric constants and safely bound nesting failures."""
    def reject_constant(_value):
        raise ValueError("Non-finite JSON number")

    try:
        return json.loads(raw, parse_constant=reject_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ConfigError("Settings must be valid JSON") from exc


def validate_values(values):
    if not isinstance(values, dict):
        raise ConfigError("settings values must be an object")
    result = {}
    for key, value in values.items():
        if key not in FIELD_MAP:
            raise ConfigError(f"Unknown setting: {key}")
        if value is not None:
            result[key] = FIELD_MAP[key].validate(value)
    return result


def _revision(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class SettingsStore:
    def __init__(self, env: Mapping[str, str], defaults=None, *, profile="docker"):
        self.fields = tuple(
            replace(field, choices=("qwen_asr", "fun_asr_nano"))
            if field.key == "model_type" and profile == "docker" else field
            for field in FIELDS if profile == "docker" or field.key not in {"inference_hardware", "qwen_preset"}
        )
        self.field_map = {field.key: field for field in self.fields}
        self.env = dict(env)
        injected_keys = set(env.get("CAPSWRITER_SETTINGS_INJECTED_KEYS", "").split(","))
        for field in self.fields:
            if field.env in injected_keys:
                self.env.pop(field.env, None)
        if not self.env.get("CAPSWRITER_INFERENCE_HARDWARE", "").strip() and self.env.get("CAPSWRITER_GPU_MODE", "").strip():
            self.env["CAPSWRITER_INFERENCE_HARDWARE"] = self.env["CAPSWRITER_GPU_MODE"]
        self.defaults = {field.key: field.default for field in FIELDS}
        self.defaults.update(defaults or {})
        self.enabled = parse_bool(env, "CAPSWRITER_SETTINGS_ENABLE", False)
        raw_path = env.get("CAPSWRITER_SETTINGS_PATH", "").strip()
        self.path = Path(raw_path).expanduser().absolute() if raw_path else None
        if self.enabled and self.path is None:
            raise ConfigError("CAPSWRITER_SETTINGS_PATH is required when settings management is enabled")
        if self.enabled and not parse_api_key(env):
            raise ConfigError("Settings management requires CAPSWRITER_HTTP_API_KEY or CAPSWRITER_HTTP_API_KEY_FILE")
        self._lock = threading.RLock()
        saved = self._read()
        self.effective = self._resolve(saved)
        self.sources = {field.key: "environment" if self.env.get(field.env, "").strip()
                        else "saved" if field.key in saved else "default" for field in self.fields}

    def _read(self):
        if self.path is None:
            return {}
        try:
            with self.path.open("rb") as handle:
                raw = handle.read(MAX_SETTINGS_BYTES + 1)
        except FileNotFoundError:
            return {}
        except OSError as exc:
            raise ConfigError("Unable to read server settings file") from exc
        if len(raw) > MAX_SETTINGS_BYTES:
            raise ConfigError("Server settings file exceeds size limit")
        try:
            data = decode_settings_json(raw)
        except ConfigError as exc:
            raise ConfigError("Server settings file is not valid JSON") from exc
        if (not isinstance(data, dict) or set(data) != {"version", "values"}
                or type(data["version"]) is not int or data["version"] != 1):
            raise ConfigError("Server settings file requires version 1 and values")
        return self._validate(data["values"])

    def _validate(self, values):
        result = validate_values(values)
        for key, value in values.items():
            if key not in self.field_map:
                raise ConfigError(f"Unknown setting for this server: {key}")
            if value is not None:
                self.field_map[key].validate(value)
        return result

    def _resolve(self, saved):
        return {
            field.key: FIELD_MAP[field.key].from_env(self.env[field.env]) if self.env.get(field.env, "").strip()
            else saved.get(field.key, self.defaults[field.key])
            for field in self.fields
        }

    def environment(self):
        """A private merged mapping, never mutate the process environment."""
        result = dict(self.env)
        for field in self.fields:
            value = self.effective[field.key]
            if value is not None and not result.get(field.env, "").strip():
                result[field.env] = str(value).lower() if isinstance(value, bool) else str(value)
        return result

    def snapshot(self):
        with self._lock:
            return self._snapshot(self._read())

    def _snapshot(self, saved):
        next_values = self._resolve(saved)
        fields = []
        for field in self.fields:
            overridden = bool(self.env.get(field.env, "").strip())
            item = {
                "key": field.key, "label": field.label, "description": field.description,
                "type": field.type, "value": self.effective[field.key],
                "saved_value": saved.get(field.key), "default": self.defaults[field.key],
                "source": self.sources[field.key],
                "next_value": next_values[field.key],
                "next_source": "environment" if overridden else "saved" if field.key in saved else "default",
                "environment_variable": field.env, "overridden_by_environment": overridden,
                "restart_required": next_values[field.key] != self.effective[field.key],
                "requires_restart": True,
            }
            if field.choices:
                item["choices"] = list(field.choices)
            if field.minimum is not None:
                item["minimum"] = field.minimum
            if field.maximum is not None:
                item["maximum"] = field.maximum
            fields.append(item)
        return {"enabled": self.enabled, "restart_required": any(f["restart_required"] for f in fields), "revision": _revision(saved), "fields": fields}

    @contextmanager
    def _file_lock(self):
        # Settings file and its lock must share a persistent, writable directory.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_name(self.path.name + ".lock").open("a+b") as lock:
            if os.name == "nt":
                import msvcrt
                lock.seek(0)
                if not lock.read(1):
                    lock.write(b"0")
                    lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def update(self, values, revision):
        if not self.enabled:
            raise ConfigError("Server settings management is disabled")
        changes = self._validate(values)
        if not isinstance(revision, str) or not revision:
            raise ConfigError("revision is required; read settings before saving")
        with self._lock, self._file_lock():
            saved = self._read()
            if revision != _revision(saved):
                raise RevisionConflict("Settings changed; reload before saving")
            saved.update(changes)
            for key, value in values.items():
                if value is None:
                    saved.pop(key, None)
            self._resolve(saved)
            content = json.dumps({"version": 1, "values": saved}, ensure_ascii=False, indent=2) + "\n"
            descriptor, filename = tempfile.mkstemp(prefix=".capswriter-settings-", dir=self.path.parent)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(filename, self.path)
                if os.name != "nt":
                    directory = os.open(self.path.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
            finally:
                if os.path.exists(filename):
                    os.unlink(filename)
            return self._snapshot(saved)
