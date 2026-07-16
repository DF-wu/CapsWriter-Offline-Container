# coding: utf-8
"""HTTP API request limit helpers.

This module intentionally has no FastAPI dependency so it can be tested in the
repository verification gate without installing the server runtime stack.
"""

from __future__ import annotations

import math
from typing import Any, Protocol

try:
    # When the server stack is installed, inheriting from Starlette's parser
    # exception makes MultiPartParser close every partially-created spool before
    # Request converts the sentinel into an HTTP exception.  Dependency-light
    # verification still imports this module without Starlette.
    from starlette.formparsers import MultiPartException as _MultipartErrorBase
except ImportError:  # pragma: no cover - exercised by the dependency-light suite
    _MultipartErrorBase = Exception


DEFAULT_MAX_UPLOAD_MB = 100
DEFAULT_MAX_AUDIO_SECONDS = 3600.0
DEFAULT_TASK_TIMEOUT_SECONDS = 600.0
MAX_AUDIO_SECONDS = 14_400.0
MAX_TASK_TIMEOUT_SECONDS = 86_400.0
MIN_TASK_TIMEOUT_SECONDS = 1.0
UPLOAD_READ_CHUNK_BYTES = 1024 * 1024
MULTIPART_OVERHEAD_BYTES = 1024 * 1024
REQUEST_BODY_TOO_LARGE_MESSAGE = "CapsWriter multipart request body limit exceeded"


class UploadTooLargeError(Exception):
    """Raised when an upload exceeds the configured byte limit."""


class RequestBodyTooLargeError(_MultipartErrorBase):
    """Raised by the ASGI receive guard before multipart parsing can continue."""

    def __init__(self) -> None:
        super().__init__(REQUEST_BODY_TOO_LARGE_MESSAGE)


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
    maximum: float = MAX_TASK_TIMEOUT_SECONDS,
) -> float:
    """Return a usable recognition timeout in seconds."""
    timeout = _positive_float(value, default, minimum)
    return timeout if timeout <= maximum else default


def audio_limit_bytes(
    value: Any,
    default_seconds: float = DEFAULT_MAX_AUDIO_SECONDS,
) -> tuple[int, float]:
    """Return the decoded PCM byte limit and normalized duration limit."""
    seconds = _positive_float(value, default_seconds, 1.0)
    if seconds > MAX_AUDIO_SECONDS:
        seconds = default_seconds
    # The server pipeline consumes 16 kHz, mono, float32 PCM.
    try:
        byte_limit = int(seconds * 16_000 * 4)
    except (OverflowError, ValueError):
        seconds = DEFAULT_MAX_AUDIO_SECONDS
        byte_limit = int(seconds * 16_000 * 4)
    return byte_limit, seconds


def request_body_limit_bytes(
    upload_bytes: int,
    overhead_bytes: int = MULTIPART_OVERHEAD_BYTES,
) -> int:
    """Bound the complete multipart body while leaving room for headers/fields."""
    upload_limit = _positive_int(upload_bytes, DEFAULT_MAX_UPLOAD_MB * 1024 * 1024)
    overhead = _positive_int(overhead_bytes, MULTIPART_OVERHEAD_BYTES)
    return upload_limit + overhead


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
