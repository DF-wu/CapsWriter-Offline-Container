# coding: utf-8

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BROWSER_SMOKE_SCRIPT = ROOT / "client" / "web" / "scripts" / "browser-smoke.mjs"


class WebBrowserSmokeScriptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = BROWSER_SMOKE_SCRIPT.read_text(encoding="utf-8")

    def test_agent_browser_subprocesses_are_bounded(self) -> None:
        self.assertIn("CAPSWRITER_WEB_BROWSER_AGENT_TIMEOUT_MS", self.source)
        self.assertIn("timeout: AGENT_BROWSER_TIMEOUT_MS", self.source)
        self.assertIn('result.error?.code === "ETIMEDOUT"', self.source)
        self.assertIn("timed out after", self.source)

    def test_child_cleanup_waits_are_bounded_and_escalate(self) -> None:
        self.assertIn("CAPSWRITER_WEB_BROWSER_CHILD_SHUTDOWN_TIMEOUT_MS", self.source)
        self.assertIn("CHILD_KILL_TIMEOUT_MS", self.source)
        self.assertIn("setTimeout(() => finish(false), timeoutMs)", self.source)
        self.assertIn('child.kill("SIGKILL")', self.source)
        self.assertIn("Promise.all(children.map(stopChild))", self.source)

    def test_http_readiness_probes_are_bounded(self) -> None:
        self.assertIn("CAPSWRITER_WEB_BROWSER_HTTP_PROBE_TIMEOUT_MS", self.source)
        self.assertIn("HTTP_PROBE_TIMEOUT_MS", self.source)
        self.assertIn("new AbortController()", self.source)
        self.assertIn("signal: controller.signal", self.source)
        self.assertIn("clearTimeout(timer)", self.source)
        self.assertIn("fetchWithTimeout(url", self.source)


if __name__ == "__main__":
    unittest.main()
