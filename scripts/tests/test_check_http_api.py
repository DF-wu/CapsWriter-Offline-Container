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


class TranscriptionHandler(BaseHTTPRequestHandler):
    request_body = b""
    content_length = 0

    def log_message(self, *_args):
        return

    def do_POST(self):
        if self.path != "/v1/audio/transcriptions":
            self.send_response(404)
            self.end_headers()
            return
        self.__class__.content_length = int(self.headers.get("Content-Length", "0"))
        self.__class__.request_body = self.rfile.read(self.__class__.content_length)
        body = b"mock transcript"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class OversizedTranscriptionResponseHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        body = b"abcd"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class UnauthorizedTranscriptionHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def _json(self, status, payload):
        body = payload.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, '{"status":"ok","model":"mock_asr","version":"dev"}')
            return
        if self.path == "/v1/models":
            self._json(200, '{"data":[{"id":"mock_asr"}]}')
            return
        if self.path == "/ready":
            self._json(
                200,
                '{"status":"ok","checks":{"ffmpeg_available":true,"task_router_bound":true}}',
            )
            return
        self._json(404, '{"error":"not found"}')

    def do_POST(self):
        if self.path == "/v1/audio/transcriptions":
            self._json(401, '{"error":{"message":"Invalid API key"}}')
            return
        self._json(404, '{"error":"not found"}')


class CheckHttpApiMultipartTest(unittest.TestCase):
    def test_resolve_api_key_reads_key_file_when_key_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            key_file = Path(tmp) / "capswriter.key"
            key_file.write_text("sk-from-file\n", encoding="utf-8")

            self.assertEqual(
                check_http_api.resolve_api_key("", str(key_file)),
                "sk-from-file",
            )

    def test_resolve_api_key_prefers_explicit_key_over_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            key_file = Path(tmp) / "capswriter.key"
            key_file.write_text("sk-from-file\n", encoding="utf-8")

            self.assertEqual(
                check_http_api.resolve_api_key("sk-explicit", str(key_file)),
                "sk-explicit",
            )

    def test_resolve_api_key_rejects_empty_key_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            key_file = Path(tmp) / "capswriter.key"
            key_file.write_text("\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must not be empty"):
                check_http_api.resolve_api_key("", str(key_file))

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

    def test_api_get_rejects_oversized_response_body(self) -> None:
        class FakeResponse:
            def __init__(self) -> None:
                self.requested_sizes = []

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, size=-1):
                self.requested_sizes.append(size)
                return b"xxxx"

        response = FakeResponse()
        with (
            patch.object(check_http_api, "MAX_RESPONSE_BODY_BYTES", 3),
            patch.object(check_http_api.urllib.request, "urlopen", return_value=response),
        ):
            with self.assertRaisesRegex(ValueError, "exceeded 3 bytes"):
                check_http_api._api_get("http://127.0.0.1:6017", "/health", "")

        self.assertEqual(response.requested_sizes, [4])

    def test_http_error_preview_rejects_oversized_response_body(self) -> None:
        error = check_http_api.urllib.error.HTTPError(
            "http://127.0.0.1:6017/ready",
            503,
            "Service Unavailable",
            {},
            io.BytesIO(b"xxxx"),
        )

        with patch.object(check_http_api, "MAX_RESPONSE_BODY_BYTES", 3):
            preview = check_http_api._http_error_preview(error)

        self.assertEqual(preview, "HTTP response body exceeded 3 bytes")

    def test_api_post_uses_configured_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            audio.write_bytes(b"RIFF")
            transport = Mock(return_value=("text/plain; charset=utf-8", b"ok"))

            with patch.object(check_http_api, "_http_post_stream", transport):
                result = check_http_api._api_post(
                    "http://127.0.0.1:6017",
                    "/v1/audio/transcriptions",
                    str(audio),
                    "text",
                    "sk-local-dev",
                    7.5,
                )

        self.assertEqual(result["_raw_text"], "ok")
        self.assertEqual(transport.call_args.args[3], 7.5)

    def test_api_post_uses_streaming_multipart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            audio.write_bytes(b"RIFF")
            transport = Mock(return_value=("text/plain; charset=utf-8", b"ok"))

            with (
                patch.object(check_http_api, "_http_post_stream", transport),
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

            _url, body, headers, _timeout = transport.call_args.args
            self.assertEqual(result["_raw_text"], "ok")
            self.assertNotIsInstance(body, bytes)
            self.assertEqual(
                int(headers["Content-Length"]),
                sum(len(chunk) for chunk in body),
            )

    def test_api_post_streaming_body_reaches_http_server(self) -> None:
        TranscriptionHandler.request_body = b""
        TranscriptionHandler.content_length = 0
        server = ThreadingHTTPServer(("127.0.0.1", 0), TranscriptionHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                audio = Path(tmp) / "sample.wav"
                audio.write_bytes(b"RIFF")
                result = check_http_api._api_post(
                    f"http://127.0.0.1:{server.server_port}",
                    "/v1/audio/transcriptions",
                    str(audio),
                    "text",
                    "",
                    7.5,
                )
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

        self.assertEqual(result["_raw_text"], "mock transcript")
        self.assertEqual(
            TranscriptionHandler.content_length,
            len(TranscriptionHandler.request_body),
        )
        self.assertIn(b"RIFF", TranscriptionHandler.request_body)
        self.assertIn(
            b'\r\nContent-Disposition: form-data; name="response_format"',
            TranscriptionHandler.request_body,
        )
        self.assertIn(b"\r\n\r\ntext\r\n", TranscriptionHandler.request_body)

    def test_api_post_rejects_oversized_response_body(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), OversizedTranscriptionResponseHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                audio = Path(tmp) / "sample.wav"
                audio.write_bytes(b"RIFF")
                with patch.object(check_http_api, "MAX_RESPONSE_BODY_BYTES", 3):
                    with self.assertRaisesRegex(ValueError, "exceeded 3 bytes"):
                        check_http_api._api_post(
                            f"http://127.0.0.1:{server.server_port}",
                            "/v1/audio/transcriptions",
                            str(audio),
                            "text",
                            "",
                            7.5,
                        )
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_positive_float_rejects_non_positive_timeout(self) -> None:
        self.assertEqual(check_http_api.positive_float("2.5"), 2.5)
        with self.assertRaises(check_http_api.argparse.ArgumentTypeError):
            check_http_api.positive_float("0")

    def test_port_number_rejects_out_of_range_values(self) -> None:
        self.assertEqual(check_http_api.port_number("6017"), 6017)
        with self.assertRaisesRegex(check_http_api.argparse.ArgumentTypeError, "1..65535"):
            check_http_api.port_number("0")
        with self.assertRaisesRegex(check_http_api.argparse.ArgumentTypeError, "1..65535"):
            check_http_api.port_number("70000")

    def test_host_name_rejects_urls_credentials_and_paths(self) -> None:
        with self.assertRaisesRegex(check_http_api.argparse.ArgumentTypeError, "not a URL"):
            check_http_api.host_name("http://127.0.0.1")
        with self.assertRaisesRegex(check_http_api.argparse.ArgumentTypeError, "not a URL"):
            check_http_api.host_name("user@127.0.0.1")
        with self.assertRaisesRegex(check_http_api.argparse.ArgumentTypeError, "not a URL"):
            check_http_api.host_name("127.0.0.1/v1")

    def test_host_name_brackets_ipv6_literals(self) -> None:
        self.assertEqual(check_http_api.host_name("::1"), "[::1]")
        self.assertEqual(check_http_api.host_name("[::1]"), "[::1]")
        with self.assertRaisesRegex(check_http_api.argparse.ArgumentTypeError, "valid address"):
            check_http_api.host_name("[not-ipv6]")


class CheckHttpApiMainTest(unittest.TestCase):
    def test_main_uses_server_aligned_default_transcription_timeout(self) -> None:
        observed_timeouts = []

        def fake_get(_base, path, _api_key):
            if path == "/health":
                return {"status": "ok", "model": "mock_asr", "version": "dev"}
            if path == "/v1/models":
                return {"data": [{"id": "mock_asr"}]}
            if path == "/ready":
                return {
                    "status": "ok",
                    "checks": {
                        "ffmpeg_available": True,
                        "task_router_bound": True,
                    },
                }
            raise AssertionError(path)

        def fake_post(_base, _path, _audio, _fmt, _api_key, timeout):
            observed_timeouts.append(timeout)
            return {"text": "mock transcript"}

        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            audio.write_bytes(b"RIFF")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                patch.object(check_http_api.shutil, "which", return_value="/usr/bin/ffmpeg"),
                patch.object(check_http_api, "_api_get", side_effect=fake_get),
                patch.object(check_http_api, "_api_post", side_effect=fake_post),
                patch.object(
                    sys,
                    "argv",
                    [
                        "check_http_api.py",
                        "--audio",
                        str(audio),
                    ],
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                code = check_http_api.main()

        self.assertEqual(code, 0)
        self.assertEqual(observed_timeouts, [600.0, 600.0, 600.0])

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

    def test_transcription_401_reports_api_key_hint_not_broken_pipe(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), UnauthorizedTranscriptionHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                audio = Path(tmp) / "sample.wav"
                audio.write_bytes(b"RIFF")
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
                            "--audio",
                            str(audio),
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
        self.assertNotIn("Broken pipe", output)
        self.assertNotIn("Connection reset", output)


if __name__ == "__main__":
    unittest.main()
