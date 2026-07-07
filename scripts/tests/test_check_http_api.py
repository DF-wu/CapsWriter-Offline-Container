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
from unittest.mock import Mock, patch

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

    def test_build_multipart_stream_reports_content_length(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            audio.write_bytes(b"RIFF")
            body, boundary, content_length = check_http_api._build_multipart_stream(
                str(audio),
                "text",
                boundary="test-boundary",
                chunk_size=2,
            )
            chunks = list(body)

        self.assertEqual(boundary, "test-boundary")
        self.assertEqual(content_length, sum(len(chunk) for chunk in chunks))
        self.assertIn(b"RI", chunks)
        self.assertIn(b"FF", chunks)
        self.assertIn(b"\r\n\r\ntext\r\n", b"".join(chunks))

    def test_api_post_uses_configured_timeout(self) -> None:
        class Response:
            headers = {"Content-Type": "text/plain; charset=utf-8"}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b"ok"

        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            audio.write_bytes(b"RIFF")
            urlopen = Mock(return_value=Response())

            with patch.object(check_http_api.urllib.request, "urlopen", urlopen):
                result = check_http_api._api_post(
                    "http://127.0.0.1:6017",
                    "/v1/audio/transcriptions",
                    str(audio),
                    "text",
                    "sk-local-dev",
                    7.5,
                )

        self.assertEqual(result["_raw_text"], "ok")
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 7.5)

    def test_api_post_uses_streaming_multipart(self) -> None:
        class Response:
            headers = {"Content-Type": "text/plain; charset=utf-8"}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b"ok"

        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            audio.write_bytes(b"RIFF")
            urlopen = Mock(return_value=Response())

            with (
                patch.object(check_http_api.urllib.request, "urlopen", urlopen),
                patch.object(
                    check_http_api,
                    "_build_multipart_body",
                    side_effect=AssertionError("byte body helper called"),
                ),
            ):
                result = check_http_api._api_post(
                    "http://127.0.0.1:6017",
                    "/v1/audio/transcriptions",
                    str(audio),
                    "text",
                    "",
                    7.5,
                )

            request = urlopen.call_args.args[0]
            self.assertEqual(result["_raw_text"], "ok")
            self.assertNotIsInstance(request.data, bytes)
            self.assertEqual(
                int(request.get_header("Content-length")),
                sum(len(chunk) for chunk in request.data),
            )

    def test_positive_float_rejects_non_positive_timeout(self) -> None:
        self.assertEqual(check_http_api.positive_float("2.5"), 2.5)
        with self.assertRaises(check_http_api.argparse.ArgumentTypeError):
            check_http_api.positive_float("0")


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
