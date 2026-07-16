# coding: utf-8
"""Validated runtime configuration for the optional HTTP API sidecar."""

from __future__ import annotations

from dataclasses import dataclass
from ipaddress import ip_address
import math
from pathlib import Path
from typing import Iterable, Mapping
from urllib.parse import urlsplit


TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}
ALLOW_INSECURE_BIND_ENV = "CAPSWRITER_HTTP_API_ALLOW_INSECURE_BIND"
API_KEY_ENV = "CAPSWRITER_HTTP_API_KEY"
API_KEY_FILE_ENV = "CAPSWRITER_HTTP_API_KEY_FILE"
MAX_HTTP_UPLOAD_MB = 1024
MAX_HTTP_AUDIO_SECONDS = 14_400.0
MAX_HTTP_TASK_TIMEOUT_SECONDS = 86_400.0
MAX_HTTP_CONCURRENT_REQUESTS = 64
MAX_HTTP_PENDING_REQUESTS = 1024


class ConfigError(ValueError):
    """Raised when an explicit HTTP API environment value is invalid."""


@dataclass(frozen=True)
class HttpApiSettings:
    enable: bool = False
    bind: str = "127.0.0.1"
    port: int = 6017
    api_key: str = ""
    max_upload_mb: int = 100
    max_audio_seconds: float = 3600.0
    task_timeout: float = 600.0
    max_concurrent_requests: int = 2
    max_pending_requests: int = 4
    cors_origins: tuple[str, ...] = ()
    allow_insecure_bind: bool = False
    log_transcripts: bool = False


def _env(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def parse_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = _env(env, name)
    if raw is None:
        return default
    value = raw.lower()
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    raise ConfigError(f"{name} must be one of: true/false, yes/no, on/off, 1/0")


def parse_int_range(
    env: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    raw = _env(env, name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{name} must be <= {maximum}")
    return value


def parse_float_range(
    env: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float:
    raw = _env(env, name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number") from exc
    if not math.isfinite(value):
        raise ConfigError(f"{name} must be >= {minimum:g}")
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum:g}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{name} must be <= {maximum:g}")
    return value


def normalize_cors_origins(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = value.split(",")
    else:
        items = [str(item) for item in value]

    origins: list[str] = []
    for item in items:
        origin = item.strip().rstrip("/")
        if not origin:
            continue
        if origin == "*":
            origins.append(origin)
            continue
        origins.append(normalize_http_origin(origin))
    return tuple(origins)


def normalize_http_origin(value: str) -> str:
    """Return a canonical browser HTTP(S) origin or raise ``ConfigError``."""
    origin = str(value).strip().rstrip("/")
    parsed = urlsplit(origin)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ConfigError(
            "CAPSWRITER_HTTP_API_CORS_ORIGINS entries must be http(s) origins"
        )
    if parsed.username is not None or parsed.password is not None:
        raise ConfigError(
            "CAPSWRITER_HTTP_API_CORS_ORIGINS entries must not include credentials"
        )
    if parsed.path or parsed.query or parsed.fragment:
        raise ConfigError(
            "CAPSWRITER_HTTP_API_CORS_ORIGINS entries must not include paths"
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise ConfigError(
            "CAPSWRITER_HTTP_API_CORS_ORIGINS entries must use valid ports"
        ) from exc
    host = parsed.hostname
    if not host:
        raise ConfigError(
            "CAPSWRITER_HTTP_API_CORS_ORIGINS entries must include a host"
        )
    host = host.casefold()
    if ":" in host:
        host = f"[{host}]"
    scheme = parsed.scheme.casefold()
    default_port = 80 if scheme == "http" else 443
    port_suffix = f":{port}" if port is not None and port != default_port else ""
    return f"{scheme}://{host}{port_suffix}"


def read_secret_file(path: str, env_name: str) -> str:
    try:
        value = Path(path).read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ConfigError(f"{env_name} could not be read: {exc}") from exc
    if not value:
        raise ConfigError(f"{env_name} must not point to an empty file")
    return value


def parse_api_key(env: Mapping[str, str]) -> str:
    explicit_key = _env(env, API_KEY_ENV)
    if explicit_key is not None:
        return explicit_key
    key_file = _env(env, API_KEY_FILE_ENV)
    if key_file is None:
        return ""
    return read_secret_file(key_file, API_KEY_FILE_ENV)


def _normalize_bind_host(bind: str) -> str:
    host = bind.strip()
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    return host.lower()


def bind_requires_auth(bind: str) -> bool:
    host = _normalize_bind_host(bind)
    if host == "localhost":
        return False
    try:
        return not ip_address(host).is_loopback
    except ValueError:
        return True


def validate_http_api_settings(settings: HttpApiSettings) -> HttpApiSettings:
    if (
        settings.enable
        and bind_requires_auth(settings.bind)
        and not settings.api_key
        and not settings.allow_insecure_bind
    ):
        raise ConfigError(
            f"{API_KEY_ENV} or {API_KEY_FILE_ENV} is required when "
            "CAPSWRITER_HTTP_API_ENABLE=true and CAPSWRITER_HTTP_API_BIND is "
            f"not loopback; set {ALLOW_INSECURE_BIND_ENV}=true only on a "
            "trusted network"
        )
    return settings


def parse_http_api_env(env: Mapping[str, str]) -> HttpApiSettings:
    settings = HttpApiSettings(
        enable=parse_bool(env, "CAPSWRITER_HTTP_API_ENABLE", False),
        bind=_env(env, "CAPSWRITER_HTTP_API_BIND") or "127.0.0.1",
        port=parse_int_range(
            env,
            "CAPSWRITER_HTTP_API_PORT",
            6017,
            minimum=1,
            maximum=65535,
        ),
        api_key=parse_api_key(env),
        max_upload_mb=parse_int_range(
            env,
            "CAPSWRITER_HTTP_API_MAX_UPLOAD_MB",
            100,
            minimum=1,
            maximum=MAX_HTTP_UPLOAD_MB,
        ),
        max_audio_seconds=parse_float_range(
            env,
            "CAPSWRITER_HTTP_API_MAX_AUDIO_SECONDS",
            3600.0,
            minimum=1.0,
            maximum=MAX_HTTP_AUDIO_SECONDS,
        ),
        task_timeout=parse_float_range(
            env,
            "CAPSWRITER_HTTP_API_TASK_TIMEOUT",
            600.0,
            minimum=1.0,
            maximum=MAX_HTTP_TASK_TIMEOUT_SECONDS,
        ),
        max_concurrent_requests=parse_int_range(
            env,
            "CAPSWRITER_HTTP_API_MAX_CONCURRENT_REQUESTS",
            2,
            minimum=1,
            maximum=MAX_HTTP_CONCURRENT_REQUESTS,
        ),
        max_pending_requests=parse_int_range(
            env,
            "CAPSWRITER_HTTP_API_MAX_PENDING_REQUESTS",
            4,
            minimum=0,
            maximum=MAX_HTTP_PENDING_REQUESTS,
        ),
        cors_origins=normalize_cors_origins(
            _env(env, "CAPSWRITER_HTTP_API_CORS_ORIGINS")
        ),
        allow_insecure_bind=parse_bool(env, ALLOW_INSECURE_BIND_ENV, False),
        log_transcripts=parse_bool(
            env,
            "CAPSWRITER_HTTP_API_LOG_TRANSCRIPTS",
            False,
        ),
    )
    return validate_http_api_settings(settings)
