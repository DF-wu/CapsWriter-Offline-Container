# coding: utf-8
"""Dependency-light HTTP upload limit helpers for the legacy server."""

from typing import Protocol


UPLOAD_READ_CHUNK_BYTES = 1024 * 1024


class UploadTooLargeError(Exception):
    """Raised as soon as a streamed upload exceeds its configured limit."""


class AsyncReadable(Protocol):
    async def read(self, size: int = -1) -> bytes:
        ...


async def read_upload_limited(
    upload: AsyncReadable,
    max_bytes: int,
    chunk_size: int = UPLOAD_READ_CHUNK_BYTES,
) -> bytes:
    """Read an async upload in bounded chunks, failing before reading to EOF."""
    if max_bytes < 1 or chunk_size < 1:
        raise ValueError("max_bytes and chunk_size must be positive")

    chunks = []
    total = 0
    while True:
        chunk = await upload.read(min(chunk_size, max_bytes - total + 1))
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > max_bytes:
            raise UploadTooLargeError
        chunks.append(chunk)
