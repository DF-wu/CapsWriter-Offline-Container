#!/usr/bin/env python3
"""Smoke-test the Expo web client with Playwright."""

from __future__ import annotations

import pathlib

from playwright.sync_api import expect, sync_playwright


ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "test-artifacts"
URL = "http://localhost:8081"


def main() -> int:
    ARTIFACTS.mkdir(exist_ok=True)
    console_errors: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1366, "height": 900})
        page.on(
            "console",
            lambda msg: console_errors.append(msg.text) if msg.type == "error" else None,
        )
        page.goto(URL, wait_until="domcontentloaded")
        page.wait_for_selector("text=CapsWriter ASR Workbench")

        expect(page.get_by_text("CapsWriter ASR Workbench")).to_be_visible()
        expect(page.get_by_text("Record or upload audio")).to_be_visible()
        page.screenshot(path=str(ARTIFACTS / "web-home.png"), full_page=True)

        page.get_by_role("button", name="設定").click()
        expect(page.get_by_text("Transcription provider")).to_be_visible()
        expect(page.get_by_text("Chat provider")).to_be_visible()
        expect(page.get_by_text("Speech provider")).to_be_visible()

        page.get_by_role("button", name="範本").click()
        expect(page.get_by_text("CapsWriter 本機 ASR")).to_be_visible()
        expect(page.get_by_text("OpenAI / 相容雲端")).to_be_visible()

        page.set_viewport_size({"width": 390, "height": 844})
        page.get_by_role("button", name="語音").click()
        expect(page.get_by_text("CapsWriter ASR Workbench")).to_be_visible()
        page.screenshot(path=str(ARTIFACTS / "web-mobile.png"), full_page=True)

        browser.close()

    blocking_errors = [
        error
        for error in console_errors
        if "favicon" not in error.lower() and "source map" not in error.lower()
    ]
    if blocking_errors:
        print("\n".join(blocking_errors))
        return 1

    print(f"screenshots: {ARTIFACTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
