# coding: utf-8
"""Readiness payload helpers for the HTTP API sidecar."""

from __future__ import annotations

from typing import Any

from .auth import auth_enabled as _auth_enabled


def readiness_auth_enabled(api_key: str | None) -> bool:
    return _auth_enabled(api_key)


def build_readiness(
    *,
    model: str,
    version: str,
    task_router_bound: bool,
    ffmpeg_available: bool,
    auth_enabled: bool,
    max_upload_mb: int,
    task_timeout: float,
    max_concurrent_requests: int,
    cors_origins: list[str],
    log_transcripts: bool = False,
) -> tuple[dict[str, Any], int]:
    """Return ``(payload, status_code)`` for ``GET /ready``."""
    checks = {
        "task_router_bound": bool(task_router_bound),
        "ffmpeg_available": bool(ffmpeg_available),
    }
    ready = all(checks.values())
    payload = {
        "status": "ok" if ready else "degraded",
        "model": model,
        "version": version,
        "checks": checks,
        "config": {
            "auth_enabled": bool(auth_enabled),
            "max_upload_mb": int(max_upload_mb),
            "task_timeout": float(task_timeout),
            "max_concurrent_requests": int(max_concurrent_requests),
            "cors_enabled": bool(cors_origins),
            "cors_origins_count": len(cors_origins),
            "log_transcripts": bool(log_transcripts),
        },
    }
    return payload, 200 if ready else 503
