# coding: utf-8

from __future__ import annotations

import asyncio
import importlib.util
import json
import tempfile
from unittest.mock import patch
import unittest

from fork_server.http_api.body_limit import RequestBodyLimitMiddleware


SERVER_DEPS_AVAILABLE = (
    importlib.util.find_spec("fastapi") is not None
    and importlib.util.find_spec("multipart") is not None
)


class RequestBodyLimitMiddlewareTest(unittest.TestCase):
    def test_chunked_body_stops_at_limit_with_openai_error(self) -> None:
        async def scenario():
            downstream_reached_end = False

            async def downstream(scope, receive, send):
                nonlocal downstream_reached_end
                del scope
                while True:
                    message = await receive()
                    if not message.get("more_body", False):
                        break
                downstream_reached_end = True
                await send({"type": "http.response.start", "status": 204, "headers": []})
                await send({"type": "http.response.body", "body": b""})

            incoming = iter(
                [
                    {"type": "http.request", "body": b"1234", "more_body": True},
                    {"type": "http.request", "body": b"5678", "more_body": False},
                ]
            )
            sent = []

            async def receive():
                return next(incoming)

            async def send(message):
                sent.append(message)

            middleware = RequestBodyLimitMiddleware(downstream, max_body_bytes=7)
            await middleware(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/v1/audio/transcriptions",
                },
                receive,
                send,
            )

            self.assertFalse(downstream_reached_end)
            self.assertEqual(sent[0]["status"], 413)
            headers = dict(sent[0]["headers"])
            self.assertEqual(headers[b"connection"], b"close")
            payload = json.loads(sent[1]["body"])
            self.assertEqual(payload["error"]["type"], "invalid_request_error")
            self.assertEqual(payload["error"]["code"], "request_too_large")

        asyncio.run(scenario())

    @unittest.skipUnless(
        SERVER_DEPS_AVAILABLE,
        "FastAPI multipart dependencies are not installed",
    )
    def test_partial_multipart_spool_is_closed_when_stream_crosses_limit(self) -> None:
        async def scenario() -> None:
            from fastapi import FastAPI, Request
            import starlette.formparsers as formparsers
            from starlette.exceptions import HTTPException as StarletteHTTPException

            from fork_server.http_api.errors import http_exception_handler
            from fork_server.http_api.multipart_form import (
                close_form_files,
                parse_multipart_form,
            )

            boundary = b"CapsWriterSpoolCleanupBoundary"
            first_chunk = b"".join(
                [
                    b"--" + boundary + b"\r\n",
                    b'Content-Disposition: form-data; name="file"; filename="sample.wav"\r\n',
                    b"Content-Type: audio/wav\r\n\r\n",
                    b"a" * 64,
                ]
            )
            second_chunk = b"".join(
                [
                    b"b" * 64,
                    b"\r\n--" + boundary + b"--\r\n",
                ]
            )
            incoming = iter(
                [
                    {
                        "type": "http.request",
                        "body": first_chunk,
                        "more_body": True,
                    },
                    {
                        "type": "http.request",
                        "body": second_chunk,
                        "more_body": False,
                    },
                ]
            )
            sent = []
            created_files = []
            original_spooled_file = tempfile.SpooledTemporaryFile

            def recording_spooled_file(*args, **kwargs):
                file = original_spooled_file(*args, **kwargs)
                created_files.append(file)
                return file

            async def receive():
                return next(incoming)

            async def send(message):
                sent.append(message)

            app = FastAPI()
            app.add_middleware(
                RequestBodyLimitMiddleware,
                max_body_bytes=len(first_chunk) + 1,
            )
            app.add_exception_handler(
                StarletteHTTPException,
                http_exception_handler,
            )

            async def parse_upload(request):
                form = await parse_multipart_form(
                    request,
                    max_files=1,
                    max_fields=12,
                )
                try:
                    return {"unexpected": True}
                finally:
                    close_form_files(form)

            parse_upload.__annotations__["request"] = Request
            app.post("/v1/audio/transcriptions")(parse_upload)

            scope = {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.3"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/v1/audio/transcriptions",
                "raw_path": b"/v1/audio/transcriptions",
                "query_string": b"",
                "root_path": "",
                "headers": [
                    (
                        b"content-type",
                        b"multipart/form-data; boundary=" + boundary,
                    )
                ],
                "client": ("127.0.0.1", 12345),
                "server": ("testserver", 80),
                "state": {},
            }

            with patch.object(
                formparsers,
                "SpooledTemporaryFile",
                side_effect=recording_spooled_file,
            ):
                await app(scope, receive, send)

            start = next(
                message
                for message in sent
                if message["type"] == "http.response.start"
            )
            response_body = b"".join(
                message.get("body", b"")
                for message in sent
                if message["type"] == "http.response.body"
            )
            self.assertEqual(start["status"], 413)
            self.assertEqual(
                json.loads(response_body)["error"]["code"],
                "request_too_large",
            )
            self.assertEqual(len(created_files), 1)
            self.assertTrue(created_files[0].closed)

        asyncio.run(scenario())

    @unittest.skipUnless(
        SERVER_DEPS_AVAILABLE,
        "FastAPI multipart dependencies are not installed",
    )
    def test_partial_spool_closes_on_disconnect_and_deadline_cancellation(self) -> None:
        async def run_case(kind: str) -> None:
            from starlette.requests import ClientDisconnect, Request
            import starlette.formparsers as formparsers

            from fork_server.http_api.multipart_form import parse_multipart_form

            boundary = b"CapsWriterInterruptedUpload"
            first_chunk = b"".join(
                [
                    b"--" + boundary + b"\r\n",
                    b'Content-Disposition: form-data; name="file"; filename="sample.wav"\r\n',
                    b"Content-Type: audio/wav\r\n\r\n",
                    b"a" * 64,
                ]
            )
            calls = 0
            created_files = []
            original_spooled_file = tempfile.SpooledTemporaryFile

            def recording_spooled_file(*args, **kwargs):
                file = original_spooled_file(*args, **kwargs)
                created_files.append(file)
                return file

            async def receive():
                nonlocal calls
                calls += 1
                if calls == 1:
                    return {
                        "type": "http.request",
                        "body": first_chunk,
                        "more_body": True,
                    }
                if kind == "disconnect":
                    return {"type": "http.disconnect"}
                await asyncio.sleep(60)
                return {"type": "http.disconnect"}

            request = Request(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/v1/audio/transcriptions",
                    "headers": [
                        (
                            b"content-type",
                            b"multipart/form-data; boundary=" + boundary,
                        )
                    ],
                },
                receive,
            )

            with patch.object(
                formparsers,
                "SpooledTemporaryFile",
                side_effect=recording_spooled_file,
            ):
                if kind == "disconnect":
                    with self.assertRaises(ClientDisconnect):
                        await parse_multipart_form(
                            request,
                            max_files=1,
                            max_fields=12,
                        )
                else:
                    with self.assertRaises(asyncio.TimeoutError):
                        await asyncio.wait_for(
                            parse_multipart_form(
                                request,
                                max_files=1,
                                max_fields=12,
                            ),
                            timeout=0.01,
                        )

            self.assertEqual(len(created_files), 1)
            self.assertTrue(created_files[0].closed)

        asyncio.run(run_case("disconnect"))
        asyncio.run(run_case("timeout"))

    def test_non_transcription_path_is_not_intercepted(self) -> None:
        async def scenario():
            called = False

            async def downstream(scope, receive, send):
                nonlocal called
                del scope, receive
                called = True
                await send({"type": "http.response.start", "status": 204, "headers": []})
                await send({"type": "http.response.body", "body": b""})

            async def receive():
                return {"type": "http.request", "body": b"x" * 100}

            async def send(message):
                del message

            middleware = RequestBodyLimitMiddleware(downstream, max_body_bytes=1)
            await middleware(
                {"type": "http", "method": "GET", "path": "/health"},
                receive,
                send,
            )
            self.assertTrue(called)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
