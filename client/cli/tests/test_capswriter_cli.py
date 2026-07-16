import io
import json
import os
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

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


class OversizedTranscriptionHandler(MockCapsWriterHandler):
    def do_POST(self):
        if self.path == "/v1/audio/transcriptions":
            self._text(200, "abcd")
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

    def test_normalize_base_url_accepts_path_prefix_with_v1(self):
        self.assertEqual(
            cli.normalize_base_url("https://asr.example.test/capswriter/v1/"),
            "https://asr.example.test/capswriter",
        )

    def test_normalize_base_url_rejects_unsupported_scheme(self):
        with self.assertRaisesRegex(ValueError, "http:// or https://"):
            cli.normalize_base_url("ftp://asr.example.test")

    def test_normalize_base_url_rejects_url_credentials(self):
        with self.assertRaisesRegex(ValueError, "username or password"):
            cli.normalize_base_url("https://user:secret@asr.example.test")

    def test_normalize_base_url_rejects_query_or_fragment(self):
        with self.assertRaisesRegex(ValueError, "query or fragment"):
            cli.normalize_base_url("https://asr.example.test/v1?token=secret")
        with self.assertRaisesRegex(ValueError, "query or fragment"):
            cli.normalize_base_url("https://asr.example.test/v1#ready")

    def test_main_rejects_invalid_base_url_before_request(self):
        with redirect_stderr(io.StringIO()) as stderr:
            status = cli.main(["health", "--base-url", "ftp://asr.example.test"])

        self.assertEqual(status, 1)
        self.assertIn("API base URL must be an absolute http:// or https:// URL", stderr.getvalue())

    def test_positive_float_rejects_non_positive_timeout(self):
        self.assertEqual(cli.positive_float("2.5"), 2.5)
        with self.assertRaises(cli.argparse.ArgumentTypeError):
            cli.positive_float("0")
        with self.assertRaises(cli.argparse.ArgumentTypeError):
            cli.positive_float("inf")

    def test_parser_rejects_non_positive_timeout(self):
        with redirect_stderr(io.StringIO()) as stderr, self.assertRaises(SystemExit) as ctx:
            cli.build_parser().parse_args(["health", "--timeout", "0"])

        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("must be > 0", stderr.getvalue())

    def test_config_default_timeout_matches_server_task_timeout(self):
        args = cli.build_parser().parse_args(["health"])

        self.assertEqual(cli._config(args).timeout, 600.0)
        self.assertEqual(
            cli._config(args).max_response_bytes,
            int(cli.DEFAULT_MAX_RESPONSE_MB * 1024 * 1024),
        )

    def test_config_sets_response_limit_from_argument(self):
        args = cli.build_parser().parse_args(["health", "--max-response-mb", "0.5"])

        self.assertEqual(cli._config(args).max_response_bytes, 512 * 1024)

    def test_config_reads_api_key_file_when_key_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            key_file = Path(tmp) / "capswriter.key"
            key_file.write_text("sk-from-file\n", encoding="utf-8")
            args = cli.build_parser().parse_args(["health", "--key-file", str(key_file)])

            self.assertEqual(cli._config(args).api_key, "sk-from-file")

    def test_config_prefers_explicit_key_over_key_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            key_file = Path(tmp) / "capswriter.key"
            key_file.write_text("sk-from-file\n", encoding="utf-8")
            args = cli.build_parser().parse_args(
                ["health", "--key", "sk-explicit", "--key-file", str(key_file)]
            )

        self.assertEqual(cli._config(args).api_key, "sk-explicit")

    def test_config_uses_key_file_environment_variable(self):
        with tempfile.TemporaryDirectory() as tmp:
            key_file = Path(tmp) / "capswriter.key"
            key_file.write_text("sk-from-env-file\n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "CAPSWRITER_HTTP_API_KEY": "",
                    "CAPSWRITER_HTTP_API_KEY_FILE": str(key_file),
                },
            ):
                args = cli.build_parser().parse_args(["health"])

                self.assertEqual(cli._config(args).api_key, "sk-from-env-file")

    def test_config_rejects_empty_api_key_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            key_file = Path(tmp) / "capswriter.key"
            key_file.write_text("\n", encoding="utf-8")
            args = cli.build_parser().parse_args(["health", "--key-file", str(key_file)])

            with self.assertRaisesRegex(ValueError, "must not be empty"):
                cli._config(args)

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

    def test_build_multipart_stream_reports_content_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            audio.write_bytes(b"RIFF")
            body, boundary, content_length = cli.build_multipart_stream(
                audio,
                {"model": "whisper-1", "response_format": "text", "prompt": ""},
                boundary="test-boundary",
                chunk_size=2,
            )
            chunks = list(body)

            self.assertEqual(boundary, "test-boundary")
            self.assertEqual(content_length, sum(len(chunk) for chunk in chunks))
            self.assertIn(b"RI", chunks)
            self.assertIn(b"FF", chunks)
            self.assertIn(b"whisper-1", b"".join(chunks))

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

    def test_diagnostic_redirect_is_rejected_without_forwarding_api_key(self):
        observed_authorization = []

        class RedirectTargetHandler(MockCapsWriterHandler):
            def do_GET(self):
                observed_authorization.append(self.headers.get("Authorization"))
                self._json(200, {"status": "unexpected redirect"})

        target = ThreadingHTTPServer(("127.0.0.1", 0), RedirectTargetHandler)

        class RedirectSourceHandler(MockCapsWriterHandler):
            def do_GET(self):
                self.send_response(302)
                self.send_header(
                    "Location",
                    f"http://127.0.0.1:{target.server_port}/stolen",
                )
                self.send_header("Content-Length", "0")
                self.end_headers()

        source = ThreadingHTTPServer(("127.0.0.1", 0), RedirectSourceHandler)
        target_thread = threading.Thread(target=target.serve_forever, daemon=True)
        source_thread = threading.Thread(target=source.serve_forever, daemon=True)
        target_thread.start()
        source_thread.start()
        try:
            config = cli.ApiConfig(
                base_url=f"http://127.0.0.1:{source.server_port}",
                api_key="never-forward-me",
                timeout=5,
            )
            with self.assertRaises(cli.ApiError) as ctx:
                cli.http_get_json(config, "/health")
        finally:
            source.shutdown()
            target.shutdown()
            source_thread.join(timeout=5)
            target_thread.join(timeout=5)
            source.server_close()
            target.server_close()

        self.assertEqual(ctx.exception.status, 302)
        self.assertEqual(observed_authorization, [])

    def test_transcribe_file_streams_audio_without_reading_whole_file(self):
        config = cli.ApiConfig(base_url=self.base_url, timeout=5)
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            audio.write_bytes(b"RIFF")
            with patch.object(Path, "read_bytes", side_effect=AssertionError("read_bytes called")):
                result = cli.transcribe_file(config, audio, response_format="text")

        self.assertEqual(cli.render_transcription(result, "text"), "mock cli transcript")

    def test_read_response_rejects_oversized_body(self):
        class FakeResponse:
            status = 200
            headers = {"Content-Type": "text/plain"}

            def __init__(self):
                self.requested_sizes = []

            def read(self, size=-1):
                self.requested_sizes.append(size)
                return b"abcd"

        response = FakeResponse()

        with self.assertRaisesRegex(ValueError, "exceeded 3 bytes"):
            cli._read_response(response, 3)

        self.assertEqual(response.requested_sizes, [4])

    def test_slow_drip_response_obeys_absolute_request_deadline(self):
        class VirtualClock:
            def __init__(self):
                self.value = 0.0

            def __call__(self):
                return self.value

            def advance(self, seconds):
                self.value += seconds

        clock = VirtualClock()

        class SlowDripResponse:
            status = 200
            headers = {"Content-Type": "application/json"}

            def __init__(self):
                self.body = b'{"status":"ok"}'
                self.offset = 0
                self.read_timeouts = []
                self.closed = False

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.closed = True

            def settimeout(self, timeout):
                self.read_timeouts.append(timeout)

            def read1(self, _size=-1):
                clock.advance(0.4)
                chunk = self.body[self.offset : self.offset + 1]
                self.offset += len(chunk)
                return chunk

            def read(self, _size=-1):
                clock.advance(2.0)
                return self.body

        response = SlowDripResponse()
        config = cli.ApiConfig(base_url="http://localhost", timeout=1.0)

        with (
            patch.object(cli, "_monotonic", new=clock),
            patch.object(cli, "_open_direct", return_value=response),
            self.assertRaisesRegex(
                TimeoutError,
                "HTTP request timed out after 1 seconds",
            ),
        ):
            cli.http_get_json(config, "/health")

        self.assertTrue(response.closed)
        self.assertEqual(response.offset, 3)
        self.assertEqual(len(response.read_timeouts), 3)
        self.assertAlmostEqual(response.read_timeouts[0], 1.0)
        self.assertAlmostEqual(response.read_timeouts[1], 0.6)
        self.assertAlmostEqual(response.read_timeouts[2], 0.2)

    def test_http_error_result_rejects_oversized_body(self):
        exc = cli.error.HTTPError(
            "http://127.0.0.1:6017/health",
            502,
            "Bad Gateway",
            {},
            io.BytesIO(b"abcd"),
        )

        with self.assertRaisesRegex(ValueError, "exceeded 3 bytes"):
            cli._result_from_http_error(exc, 3)

    def test_transcribe_file_rejects_oversized_response_body(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), OversizedTranscriptionHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            config = cli.ApiConfig(
                base_url=f"http://127.0.0.1:{server.server_port}",
                timeout=5,
                max_response_bytes=3,
            )
            with tempfile.TemporaryDirectory() as tmp:
                audio = Path(tmp) / "sample.wav"
                audio.write_bytes(b"RIFF")
                with self.assertRaisesRegex(ValueError, "exceeded 3 bytes"):
                    cli.transcribe_file(config, audio, response_format="text")
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_http_get_json_reports_invalid_json_success_response(self):
        class InvalidHealthHandler(MockCapsWriterHandler):
            def do_GET(self):
                if self.path == "/health":
                    self._text(200, f"<html>{'x' * 700}</html>", "text/html")
                    return
                super().do_GET()

        server = ThreadingHTTPServer(("127.0.0.1", 0), InvalidHealthHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            config = cli.ApiConfig(
                base_url=f"http://127.0.0.1:{server.server_port}",
                timeout=5,
            )
            with self.assertRaises(cli.ApiError) as ctx:
                cli.http_get_json(config, "/health")
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

        self.assertEqual(ctx.exception.status, 200)
        self.assertRegex(
            str(ctx.exception),
            r"^HTTP 200: Expected JSON response from /health: <html>x+.*\.\.\.$",
        )

    def test_render_transcription_reports_invalid_json_response(self):
        result = cli.HttpResult(
            status=200,
            content_type="text/html",
            body=f"<html>{'x' * 700}</html>".encode("utf-8"),
        )

        with self.assertRaises(cli.ApiError) as ctx:
            cli.render_transcription(result, "json")

        self.assertEqual(ctx.exception.status, 200)
        self.assertRegex(
            str(ctx.exception),
            r"^HTTP 200: Expected JSON response from /v1/audio/transcriptions: "
            r"<html>x+.*\.\.\.$",
        )

    def test_render_transcription_preserves_json_response(self):
        result = cli.HttpResult(
            status=200,
            content_type="application/json",
            body=json.dumps({"text": "mock cli transcript"}).encode("utf-8"),
        )

        rendered = cli.render_transcription(result, "json")

        self.assertEqual(json.loads(rendered), {"text": "mock cli transcript"})

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

    def test_main_writes_valid_json_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            output = Path(tmp) / "out.json"
            audio.write_bytes(b"RIFF")
            code = cli.main(
                [
                    "transcribe",
                    "--base-url",
                    self.base_url,
                    "--timeout",
                    "5",
                    "--format",
                    "json",
                    "--output",
                    str(output),
                    str(audio),
                ]
            )

            self.assertEqual(code, 0)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                {"text": "mock cli transcript"},
            )

    def test_write_output_replaces_existing_file_without_temp_residue(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out.txt"
            output.write_text("old transcript", encoding="utf-8")

            cli._write_output(output, "new transcript")

            self.assertEqual(output.read_text(encoding="utf-8"), "new transcript")
            self.assertEqual(list(output.parent.glob(f".{output.name}.*.tmp")), [])

    def test_write_output_cleans_temp_file_when_replace_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out.txt"
            output.write_text("old transcript", encoding="utf-8")

            with (
                patch.object(cli.Path, "replace", side_effect=OSError("replace failed")),
                self.assertRaisesRegex(OSError, "replace failed"),
            ):
                cli._write_output(output, "new transcript")

            self.assertEqual(output.read_text(encoding="utf-8"), "old transcript")
            self.assertEqual(list(output.parent.glob(f".{output.name}.*.tmp")), [])

    def test_output_dir_rejects_duplicate_generated_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "one" / "sample.wav"
            second = root / "two" / "sample.wav"
            output_dir = root / "out"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_bytes(b"RIFF")
            second.write_bytes(b"RIFF")

            with self.assertRaisesRegex(ValueError, "multiple inputs"):
                cli.output_targets_for([first, second], "text", output_dir)

    def test_output_dir_sanitizes_generated_filename_stems(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"

            self.assertEqual(
                cli.output_path_for(Path("CON.wav"), "text", output_dir).name,
                "CON_audio.txt",
            )
            self.assertEqual(
                cli.output_path_for(Path(".env.wav"), "json", output_dir).name,
                "env.json",
            )
            self.assertEqual(
                cli.output_path_for(Path('bad:name?.wav'), "vtt", output_dir).name,
                "bad_name_.vtt",
            )

    def test_output_dir_bounds_long_generated_filename_stems(self):
        stem = "a" * 200
        target = cli.output_path_for(Path(f"{stem}.wav"), "text", Path("out"))

        self.assertEqual(target.suffix, ".txt")
        self.assertLessEqual(len(target.stem), cli.MAX_OUTPUT_STEM_CHARS)
        self.assertRegex(target.stem, r"^a+-[0-9a-f]{8}$")

    def test_output_dir_rejects_duplicate_sanitized_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "bad:name.wav"
            second = root / "bad?name.wav"
            output_dir = root / "out"
            first.write_bytes(b"RIFF")
            second.write_bytes(b"RIFF")

            with self.assertRaisesRegex(ValueError, "multiple inputs"):
                cli.output_targets_for([first, second], "text", output_dir)

    def test_output_dir_rejects_case_only_generated_path_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "Sample.wav"
            second = root / "sample.wav"
            output_dir = root / "out"
            first.write_bytes(b"RIFF")
            second.write_bytes(b"RIFF")

            with self.assertRaisesRegex(ValueError, "multiple inputs"):
                cli.output_targets_for([first, second], "text", output_dir)

    def test_main_rejects_duplicate_output_dir_targets_before_transcribing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "one" / "sample.wav"
            second = root / "two" / "sample.wav"
            output_dir = root / "out"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_bytes(b"RIFF")
            second.write_bytes(b"RIFF")
            stderr = io.StringIO()

            with (
                patch.object(cli, "transcribe_file") as transcribe_file,
                redirect_stderr(stderr),
            ):
                code = cli.main(
                    [
                        "transcribe",
                        "--base-url",
                        self.base_url,
                        "--output-dir",
                        str(output_dir),
                        str(first),
                        str(second),
                    ]
                )

        self.assertEqual(code, 1)
        transcribe_file.assert_not_called()
        self.assertIn("--output-dir would write multiple inputs", stderr.getvalue())

    def test_output_dir_writes_each_success_before_later_batch_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.wav"
            second = root / "second.wav"
            output_dir = root / "out"
            first.write_bytes(b"RIFF")
            second.write_bytes(b"RIFF")
            stderr = io.StringIO()
            stdout = io.StringIO()

            with (
                patch.object(
                    cli,
                    "transcribe_file",
                    side_effect=[
                        cli.HttpResult(200, "text/plain", b"first transcript"),
                        cli.ApiError(500, "server failed"),
                    ],
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                code = cli.main(
                    [
                        "transcribe",
                        "--base-url",
                        self.base_url,
                        "--output-dir",
                        str(output_dir),
                        str(first),
                        str(second),
                    ]
                )

            self.assertEqual(code, 1)
            self.assertEqual(
                (output_dir / "first.txt").read_text(encoding="utf-8"),
                "first transcript",
            )
            self.assertIn("Wrote", stdout.getvalue())
            self.assertIn("HTTP 500: server failed", stderr.getvalue())

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

    def test_main_ready_reports_invalid_json_degraded_response(self):
        class InvalidDegradedReadyHandler(MockCapsWriterHandler):
            def do_GET(self):
                if self.path == "/ready":
                    self._text(503, f"<html>{'x' * 700}</html>", "text/html")
                    return
                super().do_GET()

        server = ThreadingHTTPServer(("127.0.0.1", 0), InvalidDegradedReadyHandler)
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
        self.assertRegex(
            stderr.getvalue(),
            r"capswriter-cli: HTTP 503: Expected JSON response from /ready: "
            r"<html>x+.*\.\.\.",
        )

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

    def test_peer_error_is_control_safe_bounded_and_secret_redacted(self):
        secret = "sk-memory-only"
        body = json.dumps(
            {"error": {"message": f"\x1b[31m{secret} " + "x" * 800}}
        ).encode("utf-8")
        result = cli.HttpResult(502, "application/json", body)

        with self.assertRaises(cli.ApiError) as ctx:
            cli._raise_result_api_error(result, secret=secret)

        message = str(ctx.exception)
        self.assertNotIn("\x1b", message)
        self.assertNotIn(secret, message)
        self.assertIn("[REDACTED]", message)
        self.assertLessEqual(len(ctx.exception.message), cli.MAX_ERROR_BODY_CHARS)

    def test_peer_error_redacts_secret_at_both_preview_boundaries(self):
        secret = "sk-" + ("s" * 64)
        exposed_prefix = secret[:-1]
        messages = (
            ("x" * 490) + "Bearer " + secret + ("y" * 700),
            "Denied" + (" " * 1995) + secret,
        )

        for reflected in messages:
            with self.subTest(reflected_length=len(reflected)):
                result = cli.HttpResult(
                    502,
                    "application/json",
                    json.dumps({"error": {"message": reflected}}).encode("utf-8"),
                )
                with self.assertRaises(cli.ApiError) as ctx:
                    cli._raise_result_api_error(result, secret=secret)

                self.assertNotIn(exposed_prefix, ctx.exception.message)
                self.assertNotIn(secret, ctx.exception.message)
                self.assertLessEqual(
                    len(ctx.exception.message),
                    cli.MAX_ERROR_BODY_CHARS,
                )

    def test_speak_reads_text_from_stdin(self):
        self.assertEqual(
            cli.read_text_argument(
                None,
                from_file=False,
                from_stdin=True,
                stdin=io.StringIO("hello from pipe"),
            ),
            "hello from pipe",
        )

    def test_main_speak_reads_text_from_stdin(self):
        stdout = io.StringIO()

        def command_for(text, **_kwargs):
            return ["tts", text]

        with (
            patch.object(sys, "stdin", io.StringIO("hello from pipeline")),
            patch.object(cli, "select_tts_command", side_effect=command_for),
            redirect_stdout(stdout),
        ):
            code = cli.main(["speak", "--stdin", "--dry-run"])

        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue(), "tts hello from pipeline\n")

    def test_speak_rejects_missing_text_without_stdin(self):
        with self.assertRaisesRegex(ValueError, "text is required"):
            cli.read_text_argument(None, from_file=False)

    def test_speak_rejects_missing_file_path(self):
        with self.assertRaisesRegex(ValueError, "file path is required"):
            cli.read_text_argument(None, from_file=True)

    def test_speak_rejects_stdin_and_text_argument(self):
        with self.assertRaisesRegex(ValueError, "--stdin cannot be combined"):
            cli.read_text_argument("ignored", from_file=False, from_stdin=True)

    def test_parser_rejects_speak_file_and_stdin_together(self):
        with redirect_stderr(io.StringIO()) as stderr, self.assertRaises(SystemExit) as ctx:
            cli.build_parser().parse_args(["speak", "--file", "--stdin", "transcript.txt"])

        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("not allowed with argument", stderr.getvalue())

    def test_parser_rejects_non_positive_tts_timeout(self):
        with redirect_stderr(io.StringIO()) as stderr, self.assertRaises(SystemExit) as ctx:
            cli.build_parser().parse_args(["speak", "--tts-timeout", "0", "hello"])

        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("must be > 0", stderr.getvalue())

    def test_speak_text_passes_timeout_to_tts_process(self):
        with (
            patch.object(cli, "select_tts_command", return_value=["tts", "hello"]) as select,
            patch.object(
                cli.subprocess,
                "run",
                return_value=cli.subprocess.CompletedProcess(["tts", "hello"], 7),
            ) as run,
        ):
            code = cli.speak_text("hello", timeout=3.25)

        self.assertEqual(code, 7)
        select.assert_called_once_with("hello", voice="", rate=None)
        run.assert_called_once_with(["tts", "hello"], check=False, timeout=3.25)

    def test_speak_text_dry_run_does_not_execute_tts_process(self):
        stdout = io.StringIO()
        with (
            patch.object(cli, "select_tts_command", return_value=["tts", "hello"]),
            patch.object(cli.subprocess, "run") as run,
            redirect_stdout(stdout),
        ):
            code = cli.speak_text("hello", dry_run=True, timeout=0.01)

        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue(), "tts hello\n")
        run.assert_not_called()

    def test_main_speak_reports_tts_timeout(self):
        stderr = io.StringIO()
        with (
            patch.object(cli, "select_tts_command", return_value=["tts", "hello"]),
            patch.object(
                cli.subprocess,
                "run",
                side_effect=cli.subprocess.TimeoutExpired(["tts", "hello"], timeout=2.5),
            ),
            redirect_stderr(stderr),
        ):
            code = cli.main(["speak", "--tts-timeout", "2.5", "hello"])

        self.assertEqual(code, 1)
        self.assertIn("capswriter-cli: TTS engine timed out after 2.5s", stderr.getvalue())

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
            voice="Reader's Voice",
            rate=2,
        )
        self.assertEqual(command[:3], ["powershell", "-NoProfile", "-Command"])
        self.assertIn("$s.SelectVoice('Reader''s Voice')", command[3])
        self.assertIn("$s.Rate = 2", command[3])
        self.assertIn("$s.Speak('hello')", command[3])


if __name__ == "__main__":
    unittest.main()
