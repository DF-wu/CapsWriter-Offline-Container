# coding: utf-8
"""HTTP API request limit helpers.

This module intentionally has no FastAPI dependency so it can be tested in the
repository verification gate without installing the server runtime stack.
"""

from __future__ import annotations

import math
from typing import Any, Protocol


DEFAULT_MAX_UPLOAD_MB = 100
DEFAULT_TASK_TIMEOUT_SECONDS = 600.0
MIN_TASK_TIMEOUT_SECONDS = 1.0
UPLOAD_READ_CHUNK_BYTES = 1024 * 1024


class UploadTooLargeError(Exception):
    """Raised when an upload exceeds the configured byte limit."""


class AsyncReadable(Protocol):
    async def read(self, size: int = -1) -> bytes:
        ...


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _positive_float(value: Any, default: float, minimum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed if parsed >= minimum else default


def upload_limit_bytes(
    value: Any,
    default_mb: int = DEFAULT_MAX_UPLOAD_MB,
) -> tuple[int, int]:
    """Return ``(byte_limit, normalized_mb)`` for upload size enforcement."""
    mb = _positive_int(value, default_mb)
    return mb * 1024 * 1024, mb


def task_timeout_seconds(
    value: Any,
    default: float = DEFAULT_TASK_TIMEOUT_SECONDS,
    minimum: float = MIN_TASK_TIMEOUT_SECONDS,
) -> float:
    """Return a usable recognition timeout in seconds."""
    return _positive_float(value, default, minimum)


async def read_upload_limited(
    upload: AsyncReadable,
    max_bytes: int,
    chunk_size: int = UPLOAD_READ_CHUNK_BYTES,
) -> bytes:
    """Read an async upload stream while enforcing a byte limit."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise UploadTooLargeError
        chunks.append(chunk)
    return b"".join(chunks)
