# coding: utf-8
"""Desktop-safe server entrypoint with the optional fork HTTP runtime.

Unlike ``start_server_docker.py``, this entrypoint deliberately does not apply
container defaults to the upstream server configuration. The tray, model, and
accelerator settings therefore keep their ``config_server.py`` values on
Windows. The fork attaches validated ``CAPSWRITER_HTTP_API_*`` settings and,
when explicitly set, a desktop-safe ``CAPSWRITER_SERVER_ADDR`` override.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from multiprocessing import freeze_support
from typing import Any

from fork_server.http_api.runtime_config import (
    ConfigError,
    HttpApiSettings,
    parse_http_api_env,
)
from fork_server.runtime_limits import (
    parse_max_websocket_connections,
    parse_max_websocket_task_seconds,
)


HTTP_SETTING_ATTRS = {
    "enable": "http_api_enable",
    "bind": "http_api_bind",
    "port": "http_api_port",
    "api_key": "http_api_key",
    "max_upload_mb": "http_api_max_upload_mb",
    "max_audio_seconds": "http_api_max_audio_seconds",
    "task_timeout": "http_api_task_timeout",
    "max_concurrent_requests": "http_api_max_concurrent_requests",
    "max_pending_requests": "http_api_max_pending_requests",
    "cors_origins": "http_api_cors_origins",
    "allow_insecure_bind": "http_api_allow_insecure_bind",
    "log_transcripts": "http_api_log_transcripts",
}
SERVER_ADDR_ENV = "CAPSWRITER_SERVER_ADDR"


def apply_http_api_settings(
    settings: HttpApiSettings,
    config_cls: type[Any],
) -> None:
    """Attach only HTTP sidecar settings to an upstream config class."""

    for setting_name, config_name in HTTP_SETTING_ATTRS.items():
        value = getattr(settings, setting_name)
        if setting_name == "cors_origins":
            value = list(value)
        setattr(config_cls, config_name, value)


def apply_server_addr_override(
    environ: Mapping[str, str],
    config_cls: type[Any],
) -> None:
    """Apply an explicit WebSocket bind without changing the upstream default."""

    if SERVER_ADDR_ENV not in environ:
        return
    value = environ[SERVER_ADDR_ENV].strip()
    if (
        not value
        or len(value) > 255
        or any(character.isspace() or ord(character) < 32 for character in value)
        or any(character in value for character in "/\\?#")
    ):
        raise ConfigError(
            f"{SERVER_ADDR_ENV} must be a host or IP address without a scheme or path"
        )
    setattr(config_cls, "addr", value)


def configure_http_api(
    environ: Mapping[str, str] | None = None,
    config_cls: type[Any] | None = None,
) -> HttpApiSettings:
    """Validate opt-in fork settings without changing other desktop defaults."""

    if config_cls is None:
        from config_server import ServerConfig

        config_cls = ServerConfig
    selected_environ = os.environ if environ is None else environ
    apply_server_addr_override(selected_environ, config_cls)
    setattr(
        config_cls,
        "max_websocket_connections",
        parse_max_websocket_connections(selected_environ),
    )
    setattr(
        config_cls,
        "max_websocket_task_seconds",
        parse_max_websocket_task_seconds(selected_environ),
    )
    settings = parse_http_api_env(selected_environ)
    apply_http_api_settings(settings, config_cls)
    return settings


def main(argv: list[str] | None = None) -> int | None:
    selected_args = sys.argv[1:] if argv is None else argv
    if selected_args == ["--artifact-self-check"]:
        from artifact_self_check import run_artifact_self_check

        return run_artifact_self_check("server")

    try:
        settings = configure_http_api()
    except ConfigError as exc:
        print(f"CapsWriter configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    if settings.enable:
        from fork_server.bootstrap import create_server

        server = create_server()
    else:
        # Use the upstream class itself for the default desktop path. This is
        # stronger drift protection than running the fork subclass's copied
        # lifecycle when the HTTP sidecar is not requested.
        from core.server.app import CapsWriterServer

        server = CapsWriterServer()
    server.start()


if __name__ == "__main__":
    # Keep the upstream Windows/PyInstaller multiprocessing behavior.
    freeze_support()
    status = main()
    if status is not None:
        raise SystemExit(status)
