# coding: utf-8

from __future__ import annotations

import unittest

from fork_server.http_api.readiness import build_readiness


class ReadinessTest(unittest.TestCase):
    def test_ready_payload_is_ok_when_required_checks_pass(self) -> None:
        payload, status = build_readiness(
            model="qwen_asr",
            version="2.6",
            task_router_bound=True,
            ffmpeg_available=True,
            auth_enabled=True,
            max_upload_mb=100,
            task_timeout=600,
            max_concurrent_requests=2,
            cors_origins=["http://localhost:5173"],
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["checks"]["task_router_bound"])
        self.assertTrue(payload["config"]["auth_enabled"])
        self.assertEqual(payload["config"]["cors_origins_count"], 1)

    def test_ready_payload_is_degraded_when_required_check_fails(self) -> None:
        payload, status = build_readiness(
            model="qwen_asr",
            version="2.6",
            task_router_bound=False,
            ffmpeg_available=True,
            auth_enabled=False,
            max_upload_mb=100,
            task_timeout=600,
            max_concurrent_requests=2,
            cors_origins=[],
        )

        self.assertEqual(status, 503)
        self.assertEqual(payload["status"], "degraded")
        self.assertFalse(payload["checks"]["task_router_bound"])
        self.assertFalse(payload["config"]["cors_enabled"])


if __name__ == "__main__":
    unittest.main()
