# coding: utf-8

from __future__ import annotations

import asyncio
import unittest

from fork_server.http_api.limits import (
    UploadTooLargeError,
    audio_limit_bytes,
    read_upload_limited,
    request_body_limit_bytes,
    task_timeout_seconds,
    upload_limit_bytes,
)


class FakeUpload:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)
        self.calls = 0

    async def read(self, size: int = -1) -> bytes:
        del size
        self.calls += 1
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


class HttpLimitTest(unittest.TestCase):
    def test_upload_limit_bytes_normalizes_invalid_values(self) -> None:
        self.assertEqual(upload_limit_bytes("2"), (2 * 1024 * 1024, 2))
        self.assertEqual(upload_limit_bytes(0), (100 * 1024 * 1024, 100))
        self.assertEqual(upload_limit_bytes(-5), (100 * 1024 * 1024, 100))
        self.assertEqual(upload_limit_bytes("bad"), (100 * 1024 * 1024, 100))

    def test_task_timeout_seconds_requires_positive_value(self) -> None:
        self.assertEqual(task_timeout_seconds("2.5"), 2.5)
        self.assertEqual(task_timeout_seconds(0), 600.0)
        self.assertEqual(task_timeout_seconds("bad"), 600.0)
        self.assertEqual(task_timeout_seconds("nan"), 600.0)
        self.assertEqual(task_timeout_seconds("inf"), 600.0)
        self.assertEqual(task_timeout_seconds("1e308"), 600.0)

    def test_decoded_audio_and_multipart_limits_are_bounded(self) -> None:
        pcm_bytes, seconds = audio_limit_bytes("2.5")
        self.assertEqual(seconds, 2.5)
        self.assertEqual(pcm_bytes, 160_000)
        self.assertEqual(request_body_limit_bytes(100, 20), 120)
        self.assertEqual(audio_limit_bytes("1e308"), (230_400_000, 3600.0))

    def test_read_upload_limited_returns_combined_bytes(self) -> None:
        upload = FakeUpload([b"hello", b" ", b"world"])
        data = asyncio.run(read_upload_limited(upload, max_bytes=32, chunk_size=4))
        self.assertEqual(data, b"hello world")

    def test_read_upload_limited_stops_at_limit(self) -> None:
        upload = FakeUpload([b"1234", b"5678", b"should-not-read"])
        with self.assertRaises(UploadTooLargeError):
            asyncio.run(read_upload_limited(upload, max_bytes=7, chunk_size=4))
        self.assertEqual(upload.calls, 2)


if __name__ == "__main__":
    unittest.main()
