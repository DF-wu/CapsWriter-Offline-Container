# coding: utf-8

from __future__ import annotations

import unittest
from unittest.mock import patch

from fork_server.http_api import auth


class HttpAuthTest(unittest.TestCase):
    def test_auth_enabled_ignores_empty_values(self) -> None:
        self.assertFalse(auth.auth_enabled(None))
        self.assertFalse(auth.auth_enabled(""))
        self.assertFalse(auth.auth_enabled("   "))
        self.assertTrue(auth.auth_enabled("sk-local"))

    def test_extract_bearer_token_accepts_http_scheme_case_insensitively(self) -> None:
        self.assertEqual(auth.extract_bearer_token("Bearer sk-local"), "sk-local")
        self.assertEqual(auth.extract_bearer_token("bearer sk-local"), "sk-local")
        self.assertEqual(auth.extract_bearer_token("BEARER   sk-local  "), "sk-local")

    def test_extract_bearer_token_rejects_malformed_headers(self) -> None:
        for value in (None, "", "Bearer", "Basic sk-local", "Bearer   "):
            with self.subTest(value=value):
                self.assertIsNone(auth.extract_bearer_token(value))

    def test_bearer_token_matches_when_auth_disabled(self) -> None:
        self.assertTrue(auth.bearer_token_matches(None, ""))
        self.assertTrue(auth.bearer_token_matches(None, "  "))

    def test_bearer_token_matches_expected_secret(self) -> None:
        self.assertTrue(auth.bearer_token_matches("Bearer sk-local", "sk-local"))
        self.assertFalse(auth.bearer_token_matches("Bearer wrong", "sk-local"))
        self.assertFalse(auth.bearer_token_matches(None, "sk-local"))

    def test_bearer_token_uses_constant_time_compare(self) -> None:
        with patch.object(auth.hmac, "compare_digest", return_value=True) as compare:
            self.assertTrue(auth.bearer_token_matches("Bearer sk-local", "sk-local"))

        compare.assert_called_once_with(b"sk-local", b"sk-local")


if __name__ == "__main__":
    unittest.main()
