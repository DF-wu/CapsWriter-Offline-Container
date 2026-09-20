# coding: utf-8

from __future__ import annotations

import json
import os
import sys
import threading
import unittest
from http.client import HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_DIR))

import healthcheck  # noqa: E402


class ReadyHandler(BaseHTTPRequestHandler):
    status_code = 200
    payload = {"status": "ok"}

    def log_message(self, *_args) -> None:
        return

    def do_GET(self) -> None:
        if self.path != "/ready":
            self.send_response(404)
            self.end_headers()
            return
        data = json.dumps(self.payload).encode("utf-8")
        self.send_response(self.status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class HealthcheckTest(unittest.TestCase):
    def _serve(self, handler_cls):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.shutdown)
        return server

    def test_normalize_loopback_host_for_container_binds(self) -> None:
        self.assertEqual(healthcheck.normalize_loopback_host("0.0.0.0"), "127.0.0.1")
        self.assertEqual(healthcheck.normalize_loopback_host("::"), "::1")
        self.assertEqual(healthcheck.normalize_loopback_host("127.0.0.1"), "127.0.0.1")
        self.assertEqual(healthcheck.normalize_loopback_host("[::1]"), "::1")
        self.assertEqual(healthcheck.format_host_header("::1", 6016), "[::1]:6016")

    def test_env_enabled_accepts_common_truthy_values(self) -> None:
        for value in ("1", "true", "TRUE", "yes", "on"):
            with self.subTest(value=value), patch.dict(os.environ, {"X_FLAG": value}):
                self.assertTrue(healthcheck.env_enabled("X_FLAG"))
        with patch.dict(os.environ, {"X_FLAG": "false"}):
            self.assertFalse(healthcheck.env_enabled("X_FLAG"))

    def test_env_port_uses_default_for_missing_or_blank_values(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(healthcheck.env_port("X_PORT", 1234), 1234)
        with patch.dict(os.environ, {"X_PORT": "  "}):
            self.assertEqual(healthcheck.env_port("X_PORT", 1234), 1234)

    def test_env_port_rejects_invalid_values(self) -> None:
        for value in ("abc", "0", "65536"):
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {"X_PORT": value}),
            ):
                self.assertIsNone(healthcheck.env_port("X_PORT", 1234))
        with patch.dict(os.environ, {"X_PORT": "6017"}):
            self.assertEqual(healthcheck.env_port("X_PORT", 1234), 6017)

    def test_websocket_check_rejects_invalid_port_without_socket(self) -> None:
        with patch.dict(os.environ, {"CAPSWRITER_SERVER_PORT": "not-a-port"}):
            with patch.object(
                healthcheck.socket,
                "create_connection",
            ) as connection:
                self.assertFalse(healthcheck.check_websocket_port())
        connection.assert_not_called()

    def test_websocket_check_completes_valid_upgrade_and_masked_close(self) -> None:
        expected_accept = healthcheck.base64.b64encode(
            healthcheck.hashlib.sha1(
                (
                    healthcheck.WEBSOCKET_KEY + healthcheck.WEBSOCKET_GUID
                ).encode("ascii")
            ).digest()
        ).decode("ascii")

        class FakeSocket:
            def __init__(self) -> None:
                response = (
                    "HTTP/1.1 101 Switching Protocols\r\n"
                    "Upgrade: websocket\r\n"
                    "Connection: Upgrade\r\n"
                    f"Sec-WebSocket-Accept: {expected_accept}\r\n"
                    "Sec-WebSocket-Protocol: binary\r\n"
                    "\r\n"
                ).encode("ascii")
                self.responses = [response[:35], response[35:], b"\x88\x00"]
                self.sent = []

            def __enter__(self):
                return self

            def __exit__(self, *_exc_info):
                return False

            def settimeout(self, _timeout) -> None:
                return

            def sendall(self, data) -> None:
                self.sent.append(data)

            def recv(self, _size):
                return self.responses.pop(0)

        fake_socket = FakeSocket()
        with (
            patch.object(
                healthcheck.socket,
                "create_connection",
                return_value=fake_socket,
            ) as connection,
            patch.dict(
                os.environ,
                {
                    "CAPSWRITER_SERVER_ADDR": "::1",
                    "CAPSWRITER_SERVER_PORT": "16016",
                },
            ),
        ):
            self.assertTrue(healthcheck.check_websocket_port())

        self.assertEqual(connection.call_args.args[0], ("::1", 16016))
        self.assertGreater(connection.call_args.kwargs["timeout"], 0)
        request = fake_socket.sent[0]
        self.assertIn(b"Host: [::1]:16016\r\n", request)
        self.assertIn(b"Upgrade: websocket\r\n", request)
        self.assertIn(b"Connection: Upgrade\r\n", request)
        self.assertIn(b"Sec-WebSocket-Protocol: binary\r\n", request)
        self.assertIn(
            f"Sec-WebSocket-Key: {healthcheck.WEBSOCKET_KEY}\r\n".encode("ascii"),
            request,
        )
        self.assertEqual(fake_socket.sent[-1], b"\x88\x80\x00\x00\x00\x00")

    def test_websocket_check_rejects_non_upgrade_response(self) -> None:
        class FakeSocket:
            def __enter__(self):
                return self

            def __exit__(self, *_exc_info):
                return False

            def settimeout(self, _timeout) -> None:
                return

            def connect(self, _address) -> None:
                return

            def sendall(self, _data) -> None:
                return

            def recv(self, _size):
                return b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"

        with patch.object(
            healthcheck.socket,
            "create_connection",
            return_value=FakeSocket(),
        ):
            self.assertFalse(healthcheck.check_websocket_port())

    def test_websocket_check_enforces_absolute_deadline_before_connect(self) -> None:
        with (
            patch.object(healthcheck.time, "monotonic", side_effect=[10.0, 14.0]),
            patch.object(healthcheck.socket, "create_connection") as connection,
        ):
            self.assertFalse(healthcheck.check_websocket_port())
        connection.assert_not_called()

    def test_http_readiness_accepts_ok_payload(self) -> None:
        server = self._serve(ReadyHandler)
        with patch.dict(
            os.environ,
            {
                "CAPSWRITER_HTTP_API_BIND": "0.0.0.0",
                "CAPSWRITER_HTTP_API_PORT": str(server.server_port),
            },
        ):
            self.assertTrue(healthcheck.check_http_readiness())

    def test_http_readiness_rejects_degraded_payload(self) -> None:
        class DegradedHandler(ReadyHandler):
            status_code = 503
            payload = {"status": "degraded"}

        server = self._serve(DegradedHandler)
        with patch.dict(
            os.environ,
            {
                "CAPSWRITER_HTTP_API_BIND": "127.0.0.1",
                "CAPSWRITER_HTTP_API_PORT": str(server.server_port),
            },
        ):
            self.assertFalse(healthcheck.check_http_readiness())

    def test_http_readiness_rejects_non_json_payload(self) -> None:
        class BadJsonHandler(ReadyHandler):
            def do_GET(self) -> None:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"not-json")

        server = self._serve(BadJsonHandler)
        with patch.dict(
            os.environ,
            {
                "CAPSWRITER_HTTP_API_BIND": "127.0.0.1",
                "CAPSWRITER_HTTP_API_PORT": str(server.server_port),
            },
        ):
            self.assertFalse(healthcheck.check_http_readiness())

    def test_http_readiness_rejects_http_protocol_error(self) -> None:
        class BrokenConnection:
            def __init__(self, *_args, **_kwargs) -> None:
                return

            def request(self, *_args, **_kwargs) -> None:
                raise HTTPException("bad response")

            def close(self) -> None:
                return

        with patch.object(healthcheck, "HTTPConnection", BrokenConnection):
            self.assertFalse(healthcheck.check_http_readiness())

    def test_http_readiness_rejects_invalid_port_without_request(self) -> None:
        with patch.dict(os.environ, {"CAPSWRITER_HTTP_API_PORT": "not-a-port"}):
            with patch.object(healthcheck, "HTTPConnection") as connection:
                self.assertFalse(healthcheck.check_http_readiness())
        connection.assert_not_called()


if __name__ == "__main__":
    unittest.main()
