import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from config_server import ServerConfig as Config
from util.constants import AudioFormat
from util.server.http_api import (
    _check_auth,
    _check_transcription_content_type,
    create_app,
)
from util.server.server_classes import Result
from util.server.task_router import router as task_router


class HttpAuthenticationTests(unittest.TestCase):
    def test_auth_is_disabled_only_when_configured_key_is_empty(self):
        with patch.object(Config, "http_api_key", ""):
            _check_auth(None)

    def test_bearer_scheme_is_case_insensitive(self):
        with patch.object(Config, "http_api_key", "sëcret"):
            _check_auth("bearer sëcret")

    def test_missing_malformed_and_wrong_credentials_are_rejected(self):
        with patch.object(Config, "http_api_key", "expected"):
            for header in (None, "Basic expected", "Bearer", "Bearer wrong extra"):
                with self.subTest(header=header):
                    with self.assertRaises(HTTPException) as raised:
                        _check_auth(header)
                    self.assertEqual(raised.exception.status_code, 401)

            with self.assertRaises(HTTPException) as raised:
                _check_auth("Bearer wrong")
            self.assertEqual(raised.exception.status_code, 401)
            self.assertEqual(raised.exception.detail, "Invalid API key")


class HttpApplicationSmokeTests(unittest.TestCase):
    def test_only_multipart_transcription_bodies_reach_form_parser(self):
        _check_transcription_content_type(
            'Multipart/Form-Data; boundary="capswriter"'
        )
        for content_type in (None, "", "application/x-www-form-urlencoded"):
            with self.subTest(content_type=content_type):
                with self.assertRaises(HTTPException) as raised:
                    _check_transcription_content_type(content_type)
                self.assertEqual(raised.exception.status_code, 415)

    def test_expected_legacy_routes_are_registered(self):
        app = create_app()
        routes = {(route.path, tuple(sorted(route.methods or ()))) for route in app.routes}

        self.assertIn(("/health", ("GET",)), routes)
        self.assertIn(("/v1/models", ("GET",)), routes)
        self.assertIn(("/v1/audio/transcriptions", ("POST",)), routes)
        self.assertIn(("/v1/audio/translations", ("POST",)), routes)


class HttpMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    async def _request(self, headers):
        app = create_app()
        receive_calls = []
        sent = []

        async def receive():
            receive_calls.append(True)
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/audio/transcriptions",
            "raw_path": b"/v1/audio/transcriptions",
            "query_string": b"",
            "headers": headers,
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 6017),
        }
        await app(scope, receive, send)
        response_start = next(
            message for message in sent if message["type"] == "http.response.start"
        )
        return response_start["status"], receive_calls

    async def test_auth_rejection_happens_before_body_read(self):
        with patch.object(Config, "http_api_key", "expected"):
            status, receive_calls = await self._request(
                [(b"content-type", b"multipart/form-data; boundary=x")]
            )

        self.assertEqual(status, 401)
        self.assertEqual(receive_calls, [])

    async def test_non_multipart_rejection_happens_before_body_read(self):
        with patch.object(Config, "http_api_key", ""):
            status, receive_calls = await self._request(
                [(b"content-type", b"application/x-www-form-urlencoded")]
            )

        self.assertEqual(status, 415)
        self.assertEqual(receive_calls, [])

    async def test_valid_multipart_request_reaches_endpoint(self):
        app = create_app()
        boundary = b"capswriter-boundary"
        body = b"\r\n".join(
            (
                b"--" + boundary,
                b'Content-Disposition: form-data; name="file"; filename="audio.wav"',
                b"Content-Type: audio/wav",
                b"",
                b"audio-bytes",
                b"--" + boundary + b"--",
                b"",
            )
        )
        sent = []
        body_sent = False

        async def receive():
            nonlocal body_sent
            if body_sent:
                return {"type": "http.disconnect"}
            body_sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message):
            sent.append(message)

        def register_completed(task_id):
            future = asyncio.get_running_loop().create_future()
            future.set_result(
                Result(
                    task_id,
                    f"http:{task_id}",
                    "file",
                    duration=0.1,
                    text="hello",
                    text_accu="hello",
                    is_final=True,
                )
            )
            return future

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/audio/transcriptions",
            "raw_path": b"/v1/audio/transcriptions",
            "query_string": b"",
            "headers": [
                (
                    b"content-type",
                    b"multipart/form-data; boundary=" + boundary,
                ),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 6017),
        }
        pcm = b"\x00" * AudioFormat.seconds_to_bytes(0.1)
        with (
            patch.object(Config, "http_api_key", ""),
            patch(
                "util.server.http_api.decode_to_pcm",
                new=AsyncMock(return_value=pcm),
            ),
            patch("util.server.http_api._split_and_submit"),
            patch.object(task_router, "register", side_effect=register_completed),
        ):
            await app(scope, receive, send)

        response_start = next(
            message for message in sent if message["type"] == "http.response.start"
        )
        response_body = b"".join(
            message.get("body", b"")
            for message in sent
            if message["type"] == "http.response.body"
        )
        self.assertEqual(response_start["status"], 200)
        self.assertEqual(json.loads(response_body), {"text": "hello"})


if __name__ == "__main__":
    unittest.main()
