# coding: utf-8
"""Dependency-light parsing for shared server resource limits."""

from __future__ import annotations

from collections.abc import Mapping

from fork_server.http_api.runtime_config import parse_float_range, parse_int_range


SERVER_MAX_WEBSOCKET_CONNECTIONS_ENV = (
    "CAPSWRITER_SERVER_MAX_WEBSOCKET_CONNECTIONS"
)
DEFAULT_SERVER_MAX_WEBSOCKET_CONNECTIONS = 8
MAX_SERVER_MAX_WEBSOCKET_CONNECTIONS = 1024

SERVER_MAX_WEBSOCKET_TASK_SECONDS_ENV = (
    "CAPSWRITER_SERVER_MAX_WEBSOCKET_TASK_SECONDS"
)
DEFAULT_SERVER_MAX_WEBSOCKET_TASK_SECONDS = 3600.0
MAX_SERVER_MAX_WEBSOCKET_TASK_SECONDS = 86_400.0


def parse_max_websocket_connections(environ: Mapping[str, str]) -> int:
    return parse_int_range(
        environ,
        SERVER_MAX_WEBSOCKET_CONNECTIONS_ENV,
        DEFAULT_SERVER_MAX_WEBSOCKET_CONNECTIONS,
        minimum=1,
        maximum=MAX_SERVER_MAX_WEBSOCKET_CONNECTIONS,
    )


def parse_max_websocket_task_seconds(environ: Mapping[str, str]) -> float:
    return parse_float_range(
        environ,
        SERVER_MAX_WEBSOCKET_TASK_SECONDS_ENV,
        DEFAULT_SERVER_MAX_WEBSOCKET_TASK_SECONDS,
        minimum=1.0,
        maximum=MAX_SERVER_MAX_WEBSOCKET_TASK_SECONDS,
    )
