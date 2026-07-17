from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from client.tui import api as api_module
from client.tui.api import (
    MAX_RESPONSE_BYTES,
    MAX_TIMEOUT_SECONDS,
    ApiError,
    CapsWriterApi,
    ResponseTooLarge,
    normalize_base_url,
)


class SlowStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.closed = True
        yield b""

    async def aclose(self) -> None:
        self.closed = True


class SlowTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.stream = SlowStream()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=self.stream, request=request)


class ChunkStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b"12"
        yield b"345"


class VirtualClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class SlowDripStream(httpx.AsyncByteStream):
    def __init__(self, clock: VirtualClock) -> None:
        self.clock = clock
        self.closed = False

    async def __aiter__(self):
        for chunk in (b'{', b'"', b'status', b'":"ok"}'):
            self.clock.advance(0.4)
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


class CapsWriterApiTest(unittest.IsolatedAsyncioTestCase):
    def test_normalize_base_url_accepts_v1_and_path_prefix(self) -> None:
        self.assertEqual(normalize_base_url("http://127.0.0.1:6017/v1/"), "http://127.0.0.1:6017")
        self.assertEqual(
            normalize_base_url("https://example.test/capswriter/v1"),
            "https://example.test/capswriter",
        )

    def test_normalize_base_url_rejects_unsafe_shapes(self) -> None:
        for value in (
            "ftp://example.test",
            "example.test",
            "https://user:pass@example.test",
            "https://example.test?token=x",
            "https://example.test/#fragment",
            "http://example.test:99999",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_base_url(value)

    def test_constructor_bounds_timeout_and_response_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            CapsWriterApi("http://localhost", diagnostic_timeout=MAX_TIMEOUT_SECONDS + 1)
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            CapsWriterApi("http://localhost", max_response_bytes=MAX_RESPONSE_BYTES + 1)
        with self.assertRaisesRegex(ValueError, "positive"):
            CapsWriterApi("http://localhost", transcription_timeout=0)

    async def test_diagnostics_preserve_prefix_and_send_masked_bearer_token(self) -> None:
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.url.path.endswith("/health"):
                return httpx.Response(200, json={"status": "ok", "model": "mock"})
            if request.url.path.endswith("/ready"):
                return httpx.Response(503, json={"status": "starting"})
            return httpx.Response(200, json={"object": "list", "data": [{"id": "mock"}]})

        api = CapsWriterApi(
            "https://example.test/prefix/v1",
            "sk-secret",
            transport=httpx.MockTransport(handler),
        )
        health, ready, models = await asyncio.gather(api.health(), api.ready(), api.models())

        self.assertTrue(health.ok)
        self.assertEqual(ready.status, 503)
        self.assertFalse(ready.ok)
        self.assertEqual(models.payload["data"][0]["id"], "mock")
        self.assertEqual(
            [request.url.path for request in requests],
            ["/prefix/health", "/prefix/ready", "/prefix/v1/models"],
        )
        self.assertTrue(all(request.headers["authorization"] == "Bearer sk-secret" for request in requests))

    async def test_transcription_streams_multipart_and_renders_text(self) -> None:
        captured: dict[str, object] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["path"] = request.url.path
            captured["authorization"] = request.headers.get("authorization")
            captured["body"] = await request.aread()
            return httpx.Response(200, text="hello 世界")

        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory, "sample.wav")
            audio.write_bytes(b"RIFFmock-wave")
            result = await CapsWriterApi(
                "http://localhost:6017",
                "secret",
                transport=httpx.MockTransport(handler),
            ).transcribe(
                audio,
                response_format="text",
                model="whisper-1",
                language="zh",
                prompt="CapsWriter",
            )

        body = captured["body"]
        self.assertIsInstance(body, bytes)
        self.assertEqual(captured["path"], "/v1/audio/transcriptions")
        self.assertEqual(captured["authorization"], "Bearer secret")
        self.assertIn(b'filename="sample.wav"', body)
        self.assertIn(b'name="response_format"', body)
        self.assertIn(b"CapsWriter", body)
        self.assertEqual(result.text, "hello 世界")

    async def test_json_transcription_is_pretty_and_unicode_safe(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            await request.aread()
            return httpx.Response(200, json={"text": "繁體中文"})

        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory, "sample.wav")
            audio.write_bytes(b"audio")
            result = await CapsWriterApi(
                "http://localhost", transport=httpx.MockTransport(handler)
            ).transcribe(audio, response_format="verbose_json")
        self.assertEqual(json.loads(result.text), {"text": "繁體中文"})
        self.assertIn("繁體中文", result.text)

    async def test_openai_error_is_bounded_and_reflected_secret_is_redacted(self) -> None:
        secret = "sk-never-show"

        async def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": {"message": f"invalid {secret}"}})

        api = CapsWriterApi("http://localhost", secret, transport=httpx.MockTransport(handler))
        with self.assertRaises(ApiError) as context:
            await api.health()
        self.assertNotIn(secret, str(context.exception))
        self.assertIn("[REDACTED]", str(context.exception))

    async def test_openai_error_redacts_secret_at_both_preview_boundaries(self) -> None:
        secret = "sk-" + ("s" * 64)
        exposed_prefix = secret[:-1]
        messages = (
            ("x" * 490) + "Bearer " + secret + ("y" * 700),
            "Denied" + (" " * 1995) + secret,
        )

        for reflected in messages:
            with self.subTest(reflected_length=len(reflected)):
                async def handler(
                    _request: httpx.Request,
                    message: str = reflected,
                ) -> httpx.Response:
                    return httpx.Response(
                        401,
                        json={"error": {"message": message}},
                    )

                api = CapsWriterApi(
                    "http://localhost",
                    secret,
                    transport=httpx.MockTransport(handler),
                )
                with self.assertRaises(ApiError) as context:
                    await api.health()

                rendered = str(context.exception)
                self.assertNotIn(exposed_prefix, rendered)
                self.assertNotIn(secret, rendered)

    async def test_declared_response_larger_than_cap_is_rejected(self) -> None:
        async def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"12345", headers={"content-length": "5"})

        api = CapsWriterApi(
            "http://localhost", max_response_bytes=4, transport=httpx.MockTransport(handler)
        )
        with self.assertRaises(ResponseTooLarge):
            await api.health()

    async def test_streaming_response_larger_than_cap_is_rejected_without_length(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, stream=ChunkStream(), request=request)

        api = CapsWriterApi(
            "http://localhost", max_response_bytes=4, transport=httpx.MockTransport(handler)
        )
        with self.assertRaises(ResponseTooLarge):
            await api.health()

    async def test_real_socket_read_timeout_is_bounded(self) -> None:
        connections: list[asyncio.StreamWriter] = []

        async def stall(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            connections.append(writer)
            try:
                await reader.read()
            finally:
                writer.close()

        server = await asyncio.start_server(stall, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            api = CapsWriterApi(
                f"http://127.0.0.1:{port}", diagnostic_timeout=0.05
            )
            with self.assertRaisesRegex(ApiError, "timed out after 0.05 seconds"):
                await api.health()
        finally:
            server.close()
            await server.wait_closed()
            for writer in connections:
                writer.close()
                await writer.wait_closed()

    async def test_slow_drip_body_obeys_absolute_request_deadline(self) -> None:
        clock = VirtualClock()
        stream = SlowDripStream(clock)

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, stream=stream, request=request)

        api = CapsWriterApi(
            "http://localhost",
            diagnostic_timeout=1.0,
            transport=httpx.MockTransport(handler),
        )
        with (
            patch.object(api_module, "_monotonic", new=clock),
            self.assertRaisesRegex(
                ApiError,
                "request timed out after 1 seconds",
            ),
        ):
            await api.health()

        self.assertTrue(stream.closed)
        self.assertAlmostEqual(clock.value, 1.2)

    async def test_invalid_json_diagnostic_is_actionable(self) -> None:
        async def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="not json")

        api = CapsWriterApi("http://localhost", transport=httpx.MockTransport(handler))
        with self.assertRaisesRegex(ApiError, "expected a JSON response from /health"):
            await api.health()

    async def test_redirect_is_not_followed_with_api_key(self) -> None:
        calls = 0

        async def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(307, headers={"location": "https://other.test/health"})

        api = CapsWriterApi("http://localhost", "secret", transport=httpx.MockTransport(handler))
        with self.assertRaisesRegex(ApiError, "HTTP 307"):
            await api.health()
        self.assertEqual(calls, 1)

    async def test_cancelling_request_closes_response_stream(self) -> None:
        transport = SlowTransport()
        api = CapsWriterApi("http://localhost", transport=transport)
        task = asyncio.create_task(api.health())
        await asyncio.wait_for(transport.stream.started.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(transport.stream.closed)


if __name__ == "__main__":
    unittest.main()
