# coding: utf-8

from __future__ import annotations

import unittest

from fork_server.http_api.runtime_config import (
    ConfigError,
    normalize_cors_origins,
    parse_http_api_env,
)


class HttpRuntimeConfigTest(unittest.TestCase):
    def test_defaults_match_documented_local_mode(self) -> None:
        settings = parse_http_api_env({})
        self.assertFalse(settings.enable)
        self.assertEqual(settings.bind, "127.0.0.1")
        self.assertEqual(settings.port, 6017)
        self.assertEqual(settings.max_upload_mb, 100)
        self.assertEqual(settings.task_timeout, 600.0)
        self.assertEqual(settings.max_concurrent_requests, 2)
        self.assertEqual(settings.cors_origins, ())

    def test_parses_valid_deploy_values(self) -> None:
        settings = parse_http_api_env(
            {
                "CAPSWRITER_HTTP_API_ENABLE": "yes",
                "CAPSWRITER_HTTP_API_BIND": "0.0.0.0",
                "CAPSWRITER_HTTP_API_PORT": "16017",
                "CAPSWRITER_HTTP_API_KEY": "sk-local-dev",
                "CAPSWRITER_HTTP_API_MAX_UPLOAD_MB": "16",
                "CAPSWRITER_HTTP_API_TASK_TIMEOUT": "30.5",
                "CAPSWRITER_HTTP_API_MAX_CONCURRENT_REQUESTS": "3",
                "CAPSWRITER_HTTP_API_CORS_ORIGINS": (
                    "http://localhost:5173/, https://example.test"
                ),
            }
        )
        self.assertTrue(settings.enable)
        self.assertEqual(settings.bind, "0.0.0.0")
        self.assertEqual(settings.port, 16017)
        self.assertEqual(settings.api_key, "sk-local-dev")
        self.assertEqual(settings.max_upload_mb, 16)
        self.assertEqual(settings.task_timeout, 30.5)
        self.assertEqual(settings.max_concurrent_requests, 3)
        self.assertEqual(
            settings.cors_origins,
            ("http://localhost:5173", "https://example.test"),
        )

    def test_rejects_invalid_boolean(self) -> None:
        with self.assertRaises(ConfigError):
            parse_http_api_env({"CAPSWRITER_HTTP_API_ENABLE": "maybe"})

    def test_rejects_invalid_port(self) -> None:
        for value in ("0", "65536", "abc"):
            with self.subTest(value=value):
                with self.assertRaises(ConfigError):
                    parse_http_api_env({"CAPSWRITER_HTTP_API_PORT": value})

    def test_rejects_non_positive_limits(self) -> None:
        for name in (
            "CAPSWRITER_HTTP_API_MAX_UPLOAD_MB",
            "CAPSWRITER_HTTP_API_TASK_TIMEOUT",
            "CAPSWRITER_HTTP_API_MAX_CONCURRENT_REQUESTS",
        ):
            with self.subTest(name=name):
                with self.assertRaises(ConfigError):
                    parse_http_api_env({name: "0"})

    def test_normalize_cors_origins_allows_star_and_rejects_paths(self) -> None:
        self.assertEqual(normalize_cors_origins("*"), ("*",))
        with self.assertRaises(ConfigError):
            normalize_cors_origins("http://localhost:5173/app")
        with self.assertRaises(ConfigError):
            normalize_cors_origins("file:///tmp/index.html")


if __name__ == "__main__":
    unittest.main()
