"""Exercise management routes without importing inference dependencies."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fork_server.settings import SettingsStore

API_DEPS_AVAILABLE = all(importlib.util.find_spec(name) for name in ("fastapi", "httpx"))


@unittest.skipUnless(API_DEPS_AVAILABLE, "FastAPI/httpx are required")
class SettingsRoutesTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from fastapi import FastAPI
        import httpx
        from fork_server.http_api.settings_routes import settings_router

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "server.json"
        self.store = SettingsStore({
            "CAPSWRITER_SETTINGS_ENABLE": "true",
            "CAPSWRITER_SETTINGS_PATH": str(self.path),
            "CAPSWRITER_HTTP_API_KEY": "test-secret",
        })
        self.config = SimpleNamespace(settings_store=self.store, http_api_key="test-secret")
        self.app = FastAPI()
        self.app.include_router(settings_router(self.config))
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test")
        self.addAsyncCleanup(self.client.aclose)
        self.auth = {"Authorization": "Bearer test-secret"}

    async def save(self, values, revision=None):
        return await self.client.patch("/v1/settings", headers=self.auth, json={
            "values": values,
            "revision": self.store.snapshot()["revision"] if revision is None else revision,
        })

    async def test_authentication_required_for_get_and_patch(self):
        for method in ("GET", "PATCH"):
            for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic test-secret"}):
                response = await self.client.request(method, "/v1/settings", headers=headers)
                self.assertEqual(response.status_code, 401)
                self.assertEqual(response.headers["www-authenticate"], "Bearer")
                self.assertNotIn("test-secret", response.text)
        self.assertFalse(self.path.exists())

    async def test_authentication_happens_before_request_body_is_read(self):
        async def receive():
            self.fail("Unauthenticated management request read its body")

        messages = []

        async def send(message):
            messages.append(message)

        await self.app({
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": "PATCH", "scheme": "http", "path": "/v1/settings",
            "raw_path": b"/v1/settings", "query_string": b"", "root_path": "",
            "headers": [], "client": ("127.0.0.1", 1234), "server": ("test", 80),
        }, receive, send)
        self.assertEqual(messages[0]["status"], 401)

    async def test_save_get_conflict_and_pending_restart_contract(self):
        initial = await self.client.get("/v1/settings", headers=self.auth)
        self.assertEqual(initial.status_code, 200)
        self.assertNotIn("test-secret", initial.text)
        self.assertNotIn(str(self.path), initial.text)
        revision = initial.json()["revision"]
        response = await self.save({"max_upload_mb": 25}, revision)
        self.assertEqual(response.status_code, 200)
        field = next(field for field in response.json()["fields"] if field["key"] == "max_upload_mb")
        self.assertEqual(field["value"], 100)
        self.assertEqual(field["saved_value"], 25)
        self.assertEqual(field["next_value"], 25)
        self.assertTrue(response.json()["restart_required"])
        stale = await self.save({"format_num": False}, revision)
        self.assertEqual(stale.status_code, 409)
        self.assertNotIn("format_num", json.loads(self.path.read_text())["values"])
        reloaded = await self.client.get("/v1/settings", headers=self.auth)
        self.assertEqual(reloaded.json(), response.json())

    async def test_invalid_payloads_and_nonfinite_or_deep_json_return_422(self):
        revision = self.store.snapshot()["revision"]
        for raw in (
            b"not-json", b"[]", b"{}", b'{"values": {}, "revision": null}',
            json.dumps({"values": {"max_upload_mb": True}, "revision": revision}),
            json.dumps({"values": {"max_audio_seconds": 10 ** 400}, "revision": revision}),
            json.dumps({"values": {"max_audio_seconds": float("nan")}, "revision": revision}),
            json.dumps({"values": {}, "revision": revision, "extra": "untrusted"}),
            b"[" * 2000 + b"]" * 2000,
        ):
            with self.subTest(raw=str(raw)[:80]):
                response = await self.client.patch("/v1/settings", headers=self.auth, content=raw)
                self.assertEqual(response.status_code, 422, response.text)
        field_error = await self.save({"max_upload_mb": 0})
        self.assertIn("max_upload_mb", field_error.json()["error"]["fields"])
        self.assertFalse(self.path.exists())

    async def test_streamed_oversized_payload_returns_413_without_writing(self):
        async def chunks():
            for _ in range(20):
                yield b" " * 1000

        response = await self.client.patch("/v1/settings", headers=self.auth, content=chunks())
        self.assertEqual(response.status_code, 413)
        self.assertFalse(self.path.exists())

    async def test_filesystem_failure_does_not_expose_internal_paths(self):
        with patch.object(self.store, "update", side_effect=PermissionError("private-path private-secret")):
            response = await self.save({"format_num": False})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private-", response.text)
        self.path.write_text("broken json")
        response = await self.client.get("/v1/settings", headers=self.auth)
        self.assertEqual(response.status_code, 503)
        self.assertNotIn(str(self.path), response.text)

    async def test_disabled_management_and_missing_key_fail_closed(self):
        self.store.enabled = False
        response = await self.client.get("/v1/settings")
        self.assertFalse(response.json()["enabled"])
        self.assertEqual(response.json()["fields"], [])
        response = await self.save({"format_num": False})
        self.assertEqual(response.status_code, 403)
        self.store.enabled = True
        self.config.http_api_key = ""
        response = await self.client.get("/v1/settings", headers=self.auth)
        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
