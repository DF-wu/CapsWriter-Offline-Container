import io
import json
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from client.cli import capswriter_cli as cli  # noqa: E402


class MockCapsWriterHandler(BaseHTTPRequestHandler):
    ready_status = 200
    ready_payload = {
        "status": "ok",
        "model": "mock_asr",
        "version": "dev",
        "checks": {
            "task_router_bound": True,
            "ffmpeg_available": True,
        },
        "config": {
            "auth_enabled": False,
            "max_upload_mb": 100,
            "task_timeout": 600.0,
            "max_concurrent_requests": 2,
            "cors_enabled": False,
            "cors_origins_count": 0,
        },
    }

    def log_message(self, *_args):
        return

    def _json(self, status, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json;charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _text(self, status, payload, content_type="text/plain;charset=utf-8"):
        data = payload.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok", "model": "mock_asr", "version": "dev"})
            return
        if self.path == "/ready":
            self._json(self.ready_status, self.ready_payload)
            return
        if self.path == "/v1/models":
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "mock_asr",
                            "object": "model",
                            "owned_by": "capswriter-offline",
                            "created": 0,
                        }
                    ],
                },
            )
            return
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/audio/transcriptions":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8", errors="replace")
        if 'name="response_format"\r\n\r\ntext' in body:
            self._text(200, "mock cli transcript")
            return
        self._json(200, {"text": "mock cli transcript"})


class ErrorCapsWriterHandler(MockCapsWriterHandler):
    def do_POST(self):
        if self.path == "/v1/audio/transcriptions":
            self._json(
                401,
                {
                    "error": {
                        "message": "Invalid API key",
                        "type": "authentication_error",
                        "param": None,
                        "code": None,
                    }
                },
            )
            return
        super().do_POST()


class CapsWriterCliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), MockCapsWriterHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=5)
        cls.server.server_close()

    def test_normalize_base_url_accepts_v1(self):
        self.assertEqual(
            cli.normalize_base_url("http://localhost:6017/v1/"),
            "http://localhost:6017",
        )

    def test_build_multipart_contains_audio_and_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            audio.write_bytes(b"RIFF")
            body, boundary = cli.build_multipart(
                audio,
                {"model": "whisper-1", "response_format": "text"},
                boundary="test-boundary",
            )
            self.assertEqual(boundary, "test-boundary")
            self.assertIn(b'filename="sample.wav"', body)
            self.assertIn(b"whisper-1", body)
            self.assertIn(b"RIFF", body)

    def test_build_multipart_escapes_filename_header_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / 'sample"\r\nX-Injected: yes.wav'
            audio.write_bytes(b"RIFF")
            body, _boundary = cli.build_multipart(
                audio,
                {"model": "whisper-1"},
                boundary="test-boundary",
            )
            header = body.split(b"\r\n\r\n", 1)[0]

            self.assertIn(b'filename="sample\\"  X-Injected: yes.wav"', header)
            self.assertNotIn(b"\r\nX-Injected:", header)

    def test_health_and_transcribe_against_mock_server(self):
        config = cli.ApiConfig(base_url=self.base_url, timeout=5)
        self.assertEqual(cli.http_get_json(config, "/health")["model"], "mock_asr")
        self.assertEqual(cli.http_get_json(config, "/ready")["status"], "ok")

        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            audio.write_bytes(b"RIFF")
            result = cli.transcribe_file(config, audio, response_format="text")
            self.assertEqual(cli.render_transcription(result, "text"), "mock cli transcript")

    def test_main_writes_transcription_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            output = Path(tmp) / "out.txt"
            audio.write_bytes(b"RIFF")
            code = cli.main(
                [
                    "transcribe",
                    "--base-url",
                    self.base_url,
                    "--timeout",
                    "5",
                    "--format",
                    "text",
                    "--output",
                    str(output),
                    str(audio),
                ]
            )
            self.assertEqual(code, 0)
            self.assertEqual(output.read_text(encoding="utf-8"), "mock cli transcript")

    def test_main_ready_prints_diagnostics(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = cli.main(
                [
                    "ready",
                    "--base-url",
                    self.base_url,
                    "--timeout",
                    "5",
                ]
            )

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["checks"]["ffmpeg_available"])

    def test_main_ready_prints_degraded_payload_on_http_error(self):
        class DegradedReadyHandler(MockCapsWriterHandler):
            ready_status = 503
            ready_payload = {
                **MockCapsWriterHandler.ready_payload,
                "status": "degraded",
                "checks": {
                    "task_router_bound": False,
                    "ffmpeg_available": True,
                },
            }

        server = ThreadingHTTPServer(("127.0.0.1", 0), DegradedReadyHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = cli.main(
                    [
                        "ready",
                        "--base-url",
                        f"http://127.0.0.1:{server.server_port}",
                        "--timeout",
                        "5",
                    ]
                )
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

        self.assertEqual(code, 1)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "degraded")
        self.assertFalse(payload["checks"]["task_router_bound"])

    def test_main_ready_reports_non_json_http_error(self):
        class PlainTextReadyHandler(MockCapsWriterHandler):
            def do_GET(self):
                if self.path == "/ready":
                    self._text(502, "Bad gateway", "text/plain")
                    return
                super().do_GET()

        server = ThreadingHTTPServer(("127.0.0.1", 0), PlainTextReadyHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = cli.main(
                    [
                        "ready",
                        "--base-url",
                        f"http://127.0.0.1:{server.server_port}",
                        "--timeout",
                        "5",
                    ]
                )
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

        self.assertEqual(code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("capswriter-cli: HTTP 502: Bad gateway", stderr.getvalue())

    def test_transcribe_file_raises_api_error_with_openai_message(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), ErrorCapsWriterHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            config = cli.ApiConfig(
                base_url=f"http://127.0.0.1:{server.server_port}",
                timeout=5,
            )
            with tempfile.TemporaryDirectory() as tmp:
                audio = Path(tmp) / "sample.wav"
                audio.write_bytes(b"RIFF")
                with self.assertRaises(cli.ApiError) as ctx:
                    cli.transcribe_file(config, audio, response_format="text")
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

        self.assertEqual(ctx.exception.status, 401)
        self.assertEqual(ctx.exception.message, "Invalid API key")
        self.assertEqual(str(ctx.exception), "HTTP 401: Invalid API key")

    def test_error_message_from_body_accepts_legacy_detail_payload(self):
        self.assertEqual(
            cli.error_message_from_body(b'{"detail":"Not Found"}'),
            "Not Found",
        )

    def test_tts_command_selection_linux(self):
        command = cli.select_tts_command(
            "hello",
            platform_name="Linux",
            which=lambda name: "/usr/bin/spd-say" if name == "spd-say" else None,
            rate=10,
        )
        self.assertEqual(command, ["spd-say", "--rate", "10", "hello"])

    def test_tts_command_selection_windows(self):
        command = cli.select_tts_command(
            "hello",
            platform_name="Windows",
            which=lambda _name: None,
            rate=2,
        )
        self.assertEqual(command[:3], ["powershell", "-NoProfile", "-Command"])
        self.assertIn("$s.Rate = 2", command[3])
        self.assertIn("$s.Speak('hello')", command[3])


if __name__ == "__main__":
    unittest.main()
