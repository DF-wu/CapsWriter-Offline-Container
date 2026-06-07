#!/usr/bin/env python3
"""Exercise chat and responses streaming against the mock provider."""

from __future__ import annotations

import os
import pathlib

from playwright.sync_api import expect, sync_playwright

from browser_utils import goto_with_retry


CLIENT_URL = os.environ.get("CLIENT_URL", "http://localhost:8081")
MOCK_BASE_URL = os.environ.get("MOCK_BASE_URL", "http://127.0.0.1:8099/v1")
SETTINGS_KEY = "capswriter-client.settings.v1"
ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "test-artifacts"


def settings(mode: str) -> dict:
    return {
        "asr": {
            "baseUrl": MOCK_BASE_URL,
            "apiKey": "",
            "model": "whisper-1",
            "responseFormat": "verbose_json",
            "language": "zh",
            "prompt": "",
            "temperature": 0,
            "timeoutSec": 30,
        },
        "conversation": {
            "baseUrl": MOCK_BASE_URL,
            "apiKey": "",
            "mode": mode,
            "model": "mock-model",
            "systemPrompt": "Reply tersely.",
            "temperature": 0,
            "topP": 1,
            "frequencyPenalty": 0,
            "presencePenalty": 0,
            "maxOutputTokens": 64,
            "stream": True,
            "timeoutSec": 30,
        },
        "tts": {
            "baseUrl": MOCK_BASE_URL,
            "apiKey": "",
            "model": "tts-1",
            "voice": "alloy",
            "responseFormat": "wav",
            "speed": 1,
            "instructions": "",
            "timeoutSec": 30,
        },
        "autoSpeak": False,
        "keepConversationHistory": False,
    }


def run_case(page, mode: str, expected: str) -> None:
    page.evaluate(
        """([key, value]) => localStorage.setItem(key, JSON.stringify(value))""",
        [SETTINGS_KEY, settings(mode)],
    )
    page.reload(wait_until="domcontentloaded")
    page.wait_for_selector("text=CapsWriter ASR Workbench")
    page.get_by_role("button", name="對話").click()
    page.get_by_placeholder("Type a message, or send the latest transcript.").fill("hello")
    page.get_by_role("button", name="Send").click()
    expect(page.get_by_text(expected)).to_be_visible(timeout=10000)


def run_diagnostics(page) -> None:
    page.evaluate(
        """([key, value]) => localStorage.setItem(key, JSON.stringify(value))""",
        [SETTINGS_KEY, settings("responses")],
    )
    page.reload(wait_until="domcontentloaded")
    page.wait_for_selector("text=CapsWriter ASR Workbench")
    page.get_by_role("button", name="設定").click()
    page.get_by_role("button", name="Check all").click()
    expect(page.get_by_text("All provider model endpoints are reachable.")).to_be_visible(
        timeout=10000
    )
    expect(page.get_by_text("2 models available").first).to_be_visible(timeout=10000)
    expect(page.get_by_text("mock-chat").first).to_be_visible(timeout=10000)
    page.get_by_label("Use mock-chat for Chat").click()
    expect(page.get_by_text("Conversation model set to mock-chat.")).to_be_visible(
        timeout=10000
    )
    model = page.evaluate(
        """(key) => JSON.parse(localStorage.getItem(key)).conversation.model""",
        SETTINGS_KEY,
    )
    assert model == "mock-chat", model
    page.screenshot(path=str(ARTIFACTS / "web-diagnostics-models.png"), full_page=True)


def main() -> int:
    ARTIFACTS.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1180, "height": 900})
        goto_with_retry(page, CLIENT_URL)
        run_diagnostics(page)
        run_case(page, "chat_completions", "Mock chat stream.")
        run_case(page, "responses", "Mock responses stream.")
        browser.close()
    print("mock streaming integration passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
