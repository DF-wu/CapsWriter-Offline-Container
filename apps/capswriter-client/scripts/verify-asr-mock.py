#!/usr/bin/env python3
"""Exercise ASR file upload against the mock provider."""

from __future__ import annotations

import json
import os
import struct
import tempfile
import wave

from playwright.sync_api import expect, sync_playwright

from browser_utils import goto_with_retry


CLIENT_URL = os.environ.get("CLIENT_URL", "http://localhost:8081")
MOCK_BASE_URL = os.environ.get("MOCK_BASE_URL", "http://127.0.0.1:8099/v1")
SETTINGS_KEY = "capswriter-client.settings.v1"


def settings() -> dict:
    return {
        "asr": {
            "baseUrl": MOCK_BASE_URL,
            "apiKey": "",
            "model": "whisper-1",
            "responseFormat": "verbose_json",
            "language": "zh",
            "prompt": "mock vocabulary",
            "temperature": 0,
            "timeoutSec": 30,
        },
        "conversation": {
            "baseUrl": MOCK_BASE_URL,
            "apiKey": "",
            "mode": "chat_completions",
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


def write_wav(path: str) -> None:
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        frames = [struct.pack("<h", 0) for _ in range(1600)]
        handle.writeframes(b"".join(frames))


def main() -> int:
    with tempfile.NamedTemporaryFile(suffix=".wav") as audio:
        write_wav(audio.name)
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1180, "height": 900})
            goto_with_retry(page, CLIENT_URL)
            page.evaluate(
                """([key, value]) => localStorage.setItem(key, JSON.stringify(value))""",
                [SETTINGS_KEY, settings()],
            )
            page.reload(wait_until="domcontentloaded")
            page.wait_for_selector("text=CapsWriter ASR Workbench")
            with page.expect_file_chooser() as chooser:
                page.get_by_role("button", name="Upload").click()
            chooser.value.set_files(audio.name)
            expect(page.get_by_text("Mock ASR transcript.").first).to_be_visible(timeout=10000)
            expect(page.get_by_text("Transcribed")).to_be_visible(timeout=10000)
            browser.close()
    print("mock ASR upload integration passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
