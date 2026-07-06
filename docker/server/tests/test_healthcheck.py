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
        self.assertEqual(healthcheck.normalize_loopback_host("::"), "127.0.0.1")
        self.assertEqual(healthcheck.normalize_loopback_host("127.0.0.1"), "127.0.0.1")

    def test_env_enabled_accepts_common_truthy_values(self) -> None:
        for value in ("1", "true", "TRUE", "yes", "on"):
            with self.subTest(value=value), patch.dict(os.environ, {"X_FLAG": value}):
                self.assertTrue(healthcheck.env_enabled("X_FLAG"))
        with patch.dict(os.environ, {"X_FLAG": "false"}):
            self.assertFalse(healthcheck.env_enabled("X_FLAG"))

    def test_env_port_rejects_invalid_values(self) -> None:
        for value in ("", "abc", "0", "65536"):
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {"X_PORT": value}),
            ):
                self.assertIsNone(healthcheck.env_port("X_PORT", 1234))
        with patch.dict(os.environ, {"X_PORT": "6017"}):
            self.assertEqual(healthcheck.env_port("X_PORT", 1234), 6017)

    def test_websocket_check_rejects_invalid_port_without_socket(self) -> None:
        with patch.dict(os.environ, {"CAPSWRITER_SERVER_PORT": "not-a-port"}):
            with patch.object(healthcheck.socket, "socket") as socket_factory:
                self.assertFalse(healthcheck.check_websocket_port())
        socket_factory.assert_not_called()

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
