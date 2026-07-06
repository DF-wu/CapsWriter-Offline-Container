# coding: utf-8

from __future__ import annotations

import io
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import check_http_api  # noqa: E402


class UnauthorizedHealthHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def do_GET(self):
        if self.path == "/health":
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"Invalid API key"}}')
            return
        self.send_response(404)
        self.end_headers()


class CheckHttpApiMultipartTest(unittest.TestCase):
    def test_multipart_header_value_escapes_control_characters(self) -> None:
        value = 'sample"\\\r\nX-Injected: yes.wav'

        self.assertEqual(
            check_http_api.multipart_header_value(value),
            'sample\\"\\\\  X-Injected: yes.wav',
        )

    def test_build_multipart_body_uses_escaped_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            audio.write_bytes(b"RIFF")

            with patch.object(
                check_http_api,
                "multipart_header_value",
                return_value="escaped.wav",
            ) as escape:
                body, boundary = check_http_api._build_multipart_body(
                    str(audio),
                    "text",
                    boundary="test-boundary",
                )

        escape.assert_called_once_with("sample.wav")
        self.assertEqual(boundary, "test-boundary")
        self.assertIn(b'filename="escaped.wav"', body)
        self.assertIn(b"RIFF", body)
        self.assertIn(b"\r\n\r\ntext\r\n", body)


class CheckHttpApiMainTest(unittest.TestCase):
    def test_health_401_reports_api_key_hint_not_connection_failure(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), UnauthorizedHealthHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                patch.object(check_http_api.shutil, "which", return_value="/usr/bin/ffmpeg"),
                patch.object(
                    sys,
                    "argv",
                    [
                        "check_http_api.py",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(server.server_port),
                    ],
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                code = check_http_api.main()
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

        output = stdout.getvalue() + stderr.getvalue()
        self.assertEqual(code, 1)
        self.assertIn("HTTP 401", output)
        self.assertIn("需要 API key", output)
        self.assertNotIn("无法连接", output)


if __name__ == "__main__":
    unittest.main()
