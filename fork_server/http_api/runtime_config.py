# coding: utf-8
"""Validated runtime configuration for the optional HTTP API sidecar."""

from __future__ import annotations

from dataclasses import dataclass
from ipaddress import ip_address
from typing import Iterable, Mapping
from urllib.parse import urlsplit


TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}
ALLOW_INSECURE_BIND_ENV = "CAPSWRITER_HTTP_API_ALLOW_INSECURE_BIND"


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
    max_concurrent_requests: int = 2
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
            "CAPSWRITER_HTTP_API_KEY is required when "
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
        max_concurrent_requests=parse_int_range(
            env,
            "CAPSWRITER_HTTP_API_MAX_CONCURRENT_REQUESTS",
            2,
            minimum=1,
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
