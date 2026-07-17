from __future__ import annotations

import os
import unittest
from unittest import mock

from client.tui import __main__ as entrypoint


class EntrypointTest(unittest.TestCase):
    def test_parser_accepts_language_alias_and_bounds(self) -> None:
        args = entrypoint.build_parser().parse_args(
            [
                "--lang",
                "zh-TW",
                "--max-response-mb",
                "0.5",
                "--transcription-timeout",
                "30",
                "--max-recording-seconds",
                "45",
                "--recording-buffer-mb",
                "0.25",
            ]
        )
        self.assertEqual(args.lang, "zh-Hant")
        self.assertEqual(args.max_response_mb, 512 * 1024)
        self.assertEqual(args.transcription_timeout, 30)
        self.assertEqual(args.max_recording_seconds, 45)
        self.assertEqual(args.recording_buffer_mb, 256 * 1024)

    def test_parser_rejects_unsafe_url_and_excessive_limits(self) -> None:
        for arguments in (
            ["--base-url", "ftp://example.test"],
            ["--diagnostic-timeout", "901"],
            ["--max-response-mb", "65"],
            ["--max-recording-seconds", "1801"],
            ["--recording-buffer-mb", "65"],
        ):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                entrypoint.build_parser().parse_args(arguments)

    def test_main_uses_environment_key_without_cli_key_option(self) -> None:
        with mock.patch.dict(os.environ, {"CAPSWRITER_HTTP_API_KEY": "memory-secret"}, clear=False):
            with mock.patch.object(entrypoint.CapsWriterTui, "run") as run:
                with mock.patch.object(entrypoint, "CapsWriterTui", wraps=entrypoint.CapsWriterTui) as app_class:
                    self.assertEqual(entrypoint.main(["--lang", "en"]), 0)
        run.assert_called_once_with()
        self.assertEqual(app_class.call_args.kwargs["initial_api_key"], "memory-secret")
        self.assertNotIn("--api-key", entrypoint.build_parser().format_help())


if __name__ == "__main__":
    unittest.main()
