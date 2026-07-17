# coding: utf-8

from __future__ import annotations

import unittest

from fork_server.http_api.readiness import build_readiness, readiness_auth_enabled


class ReadinessTest(unittest.TestCase):
    def test_ready_payload_is_ok_when_required_checks_pass(self) -> None:
        payload, status = build_readiness(
            model="qwen_asr",
            version="2.6",
            task_router_bound=True,
            recognizer_process_alive=True,
            ffmpeg_available=True,
            auth_enabled=True,
            max_upload_mb=100,
            max_audio_seconds=3600,
            task_timeout=600,
            max_concurrent_requests=2,
            max_pending_requests=4,
            max_websocket_connections=8,
            max_websocket_task_seconds=3600,
            cors_origins=["http://localhost:5173"],
            log_transcripts=True,
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["checks"]["task_router_bound"])
        self.assertTrue(payload["checks"]["recognizer_process_alive"])
        self.assertTrue(payload["config"]["auth_enabled"])
        self.assertEqual(payload["config"]["max_audio_seconds"], 3600.0)
        self.assertEqual(payload["config"]["max_pending_requests"], 4)
        self.assertEqual(payload["config"]["max_websocket_connections"], 8)
        self.assertEqual(payload["config"]["max_websocket_task_seconds"], 3600.0)
        self.assertEqual(payload["config"]["cors_origins_count"], 1)
        self.assertTrue(payload["config"]["log_transcripts"])

    def test_ready_payload_is_degraded_when_required_check_fails(self) -> None:
        payload, status = build_readiness(
            model="qwen_asr",
            version="2.6",
            task_router_bound=False,
            recognizer_process_alive=True,
            ffmpeg_available=True,
            auth_enabled=False,
            max_upload_mb=100,
            max_audio_seconds=3600,
            task_timeout=600,
            max_concurrent_requests=2,
            max_pending_requests=4,
            max_websocket_connections=8,
            max_websocket_task_seconds=3600,
            cors_origins=[],
        )

        self.assertEqual(status, 503)
        self.assertEqual(payload["status"], "degraded")
        self.assertFalse(payload["checks"]["task_router_bound"])
        self.assertFalse(payload["config"]["cors_enabled"])
        self.assertFalse(payload["config"]["log_transcripts"])

    def test_ready_payload_is_degraded_after_recognizer_process_dies(self) -> None:
        payload, status = build_readiness(
            model="qwen_asr",
            version="2.6",
            task_router_bound=True,
            recognizer_process_alive=False,
            ffmpeg_available=True,
            auth_enabled=False,
            max_upload_mb=100,
            max_audio_seconds=3600,
            task_timeout=600,
            max_concurrent_requests=2,
            max_pending_requests=4,
            max_websocket_connections=8,
            max_websocket_task_seconds=3600,
            cors_origins=[],
        )

        self.assertEqual(status, 503)
        self.assertEqual(payload["status"], "degraded")
        self.assertFalse(payload["checks"]["recognizer_process_alive"])

    def test_readiness_auth_enabled_matches_auth_policy(self) -> None:
        self.assertFalse(readiness_auth_enabled(None))
        self.assertFalse(readiness_auth_enabled(""))
        self.assertFalse(readiness_auth_enabled("   "))
        self.assertTrue(readiness_auth_enabled("sk-local-dev"))


if __name__ == "__main__":
    unittest.main()
