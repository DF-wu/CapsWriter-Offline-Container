# coding: utf-8
"""Real-socket regression for rejecting an unread request body."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
import importlib.util
import socket
from unittest.mock import patch
import unittest


SERVER_DEPS_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("fastapi", "multipart", "uvicorn")
)


@unittest.skipUnless(
    SERVER_DEPS_AVAILABLE,
    "Uvicorn/FastAPI contract dependencies are not installed",
)
class UvicornPreBodyRejectionTest(unittest.TestCase):
    def test_disconnected_admission_waiter_leaves_queue_immediately(self) -> None:
        from fork_server.http_api import api
        import uvicorn

        async def scenario() -> None:
            with ExitStack() as stack:
                for name, value in {
                    "http_api_key": "",
                    "http_api_cors_origins": [],
                    "http_api_max_concurrent_requests": 1,
                    "http_api_max_pending_requests": 1,
                    "http_api_max_upload_mb": 10,
                    "http_api_max_audio_seconds": 3600,
                    "http_api_task_timeout": 30.0,
                    "http_api_log_transcripts": False,
                }.items():
                    stack.enter_context(
                        patch.object(api.Config, name, value, create=True)
                    )
                stack.enter_context(
                    patch.object(
                        api.task_router,
                        "recognizer_process_alive",
                        return_value=True,
                    )
                )

                app = api.create_app()
                listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind(("127.0.0.1", 0))
                listener.listen()
                listener.setblocking(False)

                config = uvicorn.Config(
                    app,
                    log_config=None,
                    access_log=False,
                    lifespan="off",
                )
                server = uvicorn.Server(config)
                server_task = asyncio.create_task(server.serve(sockets=[listener]))
                writer = None
                try:
                    for _attempt in range(200):
                        if server.started:
                            break
                        await asyncio.sleep(0.005)
                    self.assertTrue(server.started)

                    async with app.state.transcription_admission.slot():
                        _reader, writer = await asyncio.open_connection(
                            "127.0.0.1",
                            listener.getsockname()[1],
                        )
                        writer.write(
                            b"POST /v1/audio/transcriptions HTTP/1.1\r\n"
                            b"Host: 127.0.0.1\r\n"
                            b"Content-Type: multipart/form-data; boundary=waiting\r\n"
                            b"Content-Length: 100000\r\n"
                            b"\r\n"
                        )
                        await writer.drain()

                        for _attempt in range(200):
                            if app.state.transcription_admission.waiting == 1:
                                break
                            await asyncio.sleep(0.005)
                        self.assertEqual(
                            app.state.transcription_admission.waiting,
                            1,
                        )

                        writer.close()
                        await writer.wait_closed()
                        writer = None
                        for _attempt in range(200):
                            if app.state.transcription_admission.waiting == 0:
                                break
                            await asyncio.sleep(0.005)
                        self.assertEqual(
                            app.state.transcription_admission.waiting,
                            0,
                        )
                finally:
                    if writer is not None:
                        writer.close()
                        await writer.wait_closed()
                    server.should_exit = True
                    await asyncio.wait_for(server_task, timeout=5.0)
                    listener.close()

        asyncio.run(scenario())

    def _assert_rejection_closes_socket(
        self,
        *,
        api_key: str,
        request_bytes: bytes,
        expected_status: int,
        task_timeout: float = 5.0,
    ) -> None:
        from fork_server.http_api import api
        import uvicorn

        async def scenario() -> None:
            with ExitStack() as stack:
                for name, value in {
                    "http_api_key": api_key,
                    "http_api_cors_origins": [],
                    "http_api_max_concurrent_requests": 2,
                    "http_api_max_pending_requests": 4,
                    "http_api_max_upload_mb": 10,
                    "http_api_max_audio_seconds": 3600,
                    "http_api_task_timeout": task_timeout,
                    "http_api_log_transcripts": False,
                }.items():
                    stack.enter_context(
                        patch.object(api.Config, name, value, create=True)
                    )
                stack.enter_context(
                    patch.object(
                        api.task_router,
                        "recognizer_process_alive",
                        return_value=True,
                    )
                )

                app = api.create_app()
                listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind(("127.0.0.1", 0))
                listener.listen()
                listener.setblocking(False)
                port = listener.getsockname()[1]

                config = uvicorn.Config(
                    app,
                    log_config=None,
                    access_log=False,
                    lifespan="off",
                )
                server = uvicorn.Server(config)
                server_task = asyncio.create_task(server.serve(sockets=[listener]))
                writer = None
                try:
                    for _attempt in range(200):
                        if server.started:
                            break
                        await asyncio.sleep(0.005)
                    self.assertTrue(server.started)

                    reader, writer = await asyncio.open_connection(
                        "127.0.0.1",
                        port,
                    )
                    writer.write(request_bytes)
                    await writer.drain()

                    header_block = await asyncio.wait_for(
                        reader.readuntil(b"\r\n\r\n"),
                        timeout=2.0,
                    )
                    header_lines = header_block.decode("latin-1").split("\r\n")
                    self.assertIn(f" {expected_status} ", header_lines[0])
                    headers = {
                        key.strip().casefold(): value.strip().casefold()
                        for line in header_lines[1:]
                        if ":" in line
                        for key, value in [line.split(":", 1)]
                    }
                    self.assertEqual(headers.get("connection"), "close")

                    content_length = int(headers["content-length"])
                    await asyncio.wait_for(
                        reader.readexactly(content_length),
                        timeout=2.0,
                    )
                    self.assertEqual(
                        await asyncio.wait_for(reader.read(1), timeout=2.0),
                        b"",
                    )
                finally:
                    if writer is not None:
                        writer.close()
                        await writer.wait_closed()
                    server.should_exit = True
                    await asyncio.wait_for(server_task, timeout=5.0)
                    listener.close()

        asyncio.run(scenario())

    def test_unauthorized_slow_body_closes_keep_alive_socket(self) -> None:
        self._assert_rejection_closes_socket(
            api_key="sk-required",
            request_bytes=(
                b"POST /v1/audio/transcriptions HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Content-Type: multipart/form-data; boundary=slow\r\n"
                b"Content-Length: 100\r\n"
                b"\r\n"
            ),
            expected_status=401,
        )

    def test_malformed_slow_multipart_closes_keep_alive_socket(self) -> None:
        self._assert_rejection_closes_socket(
            api_key="",
            request_bytes=(
                b"POST /v1/audio/transcriptions HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Content-Type: multipart/form-data; boundary=expected\r\n"
                b"Content-Length: 100000\r\n"
                b"\r\n"
                b"--wrong-boundary\r\n"
            ),
            expected_status=400,
        )

    def test_slow_upload_deadline_closes_keep_alive_socket(self) -> None:
        self._assert_rejection_closes_socket(
            api_key="",
            request_bytes=(
                b"POST /v1/audio/transcriptions HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Content-Type: multipart/form-data; boundary=slow\r\n"
                b"Content-Length: 100\r\n"
                b"\r\n"
            ),
            expected_status=504,
            task_timeout=1.0,
        )

    def test_unsupported_translation_closes_unread_body_socket(self) -> None:
        self._assert_rejection_closes_socket(
            api_key="",
            request_bytes=(
                b"POST /v1/audio/translations HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Content-Type: multipart/form-data; boundary=slow\r\n"
                b"Content-Length: 100\r\n"
                b"\r\n"
            ),
            expected_status=501,
        )


if __name__ == "__main__":
    unittest.main()
