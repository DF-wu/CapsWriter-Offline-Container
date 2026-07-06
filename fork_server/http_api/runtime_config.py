# coding: utf-8
"""Validated runtime configuration for the optional HTTP API sidecar."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping
from urllib.parse import urlsplit


TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}


class ConfigError(ValueError):
    """Raised when an explicit HTTP API environment value is invalid."""


@dataclass(frozen=True)
class HttpApiSettings:
    enable: bool = False
    bind: str = "127.0.0.1"
    port: int = 6017
    api_key: str = ""
    max_upload_mb: int = 100
    task_timeout: float = 600.0
    cors_origins: tuple[str, ...] = ()


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
) -> float:
    raw = _env(env, name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum:g}")
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
        parsed = urlsplit(origin)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ConfigError(
                "CAPSWRITER_HTTP_API_CORS_ORIGINS entries must be http(s) origins"
            )
        if parsed.path or parsed.query or parsed.fragment:
            raise ConfigError(
                "CAPSWRITER_HTTP_API_CORS_ORIGINS entries must not include paths"
            )
        origins.append(origin)
    return tuple(origins)


def parse_http_api_env(env: Mapping[str, str]) -> HttpApiSettings:
    return HttpApiSettings(
        enable=parse_bool(env, "CAPSWRITER_HTTP_API_ENABLE", False),
        bind=_env(env, "CAPSWRITER_HTTP_API_BIND") or "127.0.0.1",
        port=parse_int_range(
            env,
            "CAPSWRITER_HTTP_API_PORT",
            6017,
            minimum=1,
            maximum=65535,
        ),
        api_key=_env(env, "CAPSWRITER_HTTP_API_KEY") or "",
        max_upload_mb=parse_int_range(
            env,
            "CAPSWRITER_HTTP_API_MAX_UPLOAD_MB",
            100,
            minimum=1,
        ),
        task_timeout=parse_float_range(
            env,
            "CAPSWRITER_HTTP_API_TASK_TIMEOUT",
            600.0,
            minimum=1.0,
        ),
        cors_origins=normalize_cors_origins(
            _env(env, "CAPSWRITER_HTTP_API_CORS_ORIGINS")
        ),
    )
