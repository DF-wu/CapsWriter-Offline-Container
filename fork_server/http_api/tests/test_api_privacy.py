# coding: utf-8

from __future__ import annotations

import unittest
from unittest.mock import Mock

from fork_server.http_api import privacy


def _mock_text(mock) -> str:
    return "\n".join(
        " ".join(str(arg) for arg in call.args)
        for call in mock.call_args_list
    )


class HttpApiPrivacyLoggingTest(unittest.TestCase):
    def test_transcription_result_redacts_text_by_default(self) -> None:
        logger = Mock()
        console = Mock()

        privacy.log_transcription_result(
            logger,
            console,
            "abcdef123456",
            1.25,
            "private\ntranscript",
            log_sensitive_text=False,
        )

        self.assertIn("text=<redacted>", _mock_text(logger.info))
        self.assertIn("text_chars=18", _mock_text(logger.info))
        self.assertNotIn("private", _mock_text(logger.info))
        self.assertIn("[redacted]", _mock_text(console.print))
        self.assertNotIn("private", _mock_text(console.print))

    def test_prompt_context_redacts_text_by_default(self) -> None:
        logger = Mock()

        privacy.log_prompt_context(
            logger,
            "abcdef123456",
            "secret customer term",
            log_sensitive_text=False,
        )

        self.assertIn("context=<redacted>", _mock_text(logger.debug))
        self.assertIn("context_chars=20", _mock_text(logger.debug))
        self.assertNotIn("secret", _mock_text(logger.debug))

    def test_log_transcripts_opt_in_preserves_existing_detail(self) -> None:
        logger = Mock()
        console = Mock()

        privacy.log_prompt_context(
            logger,
            "abcdef123456",
            "secret customer term",
            log_sensitive_text=True,
        )
        privacy.log_transcription_result(
            logger,
            console,
            "abcdef123456",
            1.25,
            "private\ntranscript",
            log_sensitive_text=True,
        )

        self.assertIn("secret customer term", _mock_text(logger.debug))
        self.assertIn("private transcript", _mock_text(logger.info))
        self.assertIn("private\ntranscript", _mock_text(console.print))


if __name__ == "__main__":
    unittest.main()
