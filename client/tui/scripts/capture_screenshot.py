#!/usr/bin/env python3
"""Capture an accessible SVG from the real Textual app in headless Pilot mode."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import re
import sys
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client.tui.app import CapsWriterTui
from client.tui.i18n import normalize_locale
from client.tui.recorder import UnavailableRecorder


DEFAULT_OUTPUT = ROOT / "docs" / "assets" / "tui-workbench.svg"
DEFAULT_DESCRIPTION = (
    "CapsWriter TUI v2 running in Textual: server connection and diagnostics "
    "controls above a keyboard-first file transcription workspace and transcript panel."
)
RICH_REMOTE_FONT_SOURCE = re.compile(
    r'src:\s*local\("(?P<local>[^"]+)"\),\s*'
    r'url\("https://cdnjs\.cloudflare\.com/[^"]+"\)\s*format\("woff2"\),\s*'
    r'url\("https://cdnjs\.cloudflare\.com/[^"]+"\)\s*format\("woff"\);'
)
REMOTE_CSS_URL = re.compile(r"url\(\s*['\"]?https?://", re.IGNORECASE)


def add_accessibility_metadata(svg: str, description: str) -> str:
    """Give Rich's terminal SVG an explicit title/description relationship."""

    root = re.search(r"<svg\b[^>]*>", svg)
    if root is None:
        raise ValueError("Textual screenshot export did not contain an SVG root")
    opening = root.group(0)
    if "role=" not in opening:
        opening = opening[:-1] + ' role="img" aria-labelledby="tui-shot-title tui-shot-desc">'
        svg = svg[: root.start()] + opening + svg[root.end() :]

    title = re.search(r"<title(?:\s[^>]*)?>(?P<text>.*?)</title>", svg, re.DOTALL)
    if title is None:
        insert_at = svg.index(">", svg.index("<svg")) + 1
        metadata = (
            '<title id="tui-shot-title">CapsWriter TUI v2</title>'
            f'<desc id="tui-shot-desc">{escape(description)}</desc>'
        )
        return svg[:insert_at] + metadata + svg[insert_at:]

    title_text = title.group("text")
    replacement = (
        f'<title id="tui-shot-title">{title_text}</title>'
        f'<desc id="tui-shot-desc">{escape(description)}</desc>'
    )
    return svg[: title.start()] + replacement + svg[title.end() :]


def remove_remote_font_sources(svg: str) -> str:
    """Keep generated documentation screenshots self-contained and offline-safe."""

    offline = RICH_REMOTE_FONT_SOURCE.sub(
        lambda match: f'src: local("{match.group("local")}");',
        svg,
    )
    if REMOTE_CSS_URL.search(offline):
        raise ValueError("Textual screenshot export retained an external CSS URL")
    return offline


async def capture_svg(*, locale: str, width: int, height: int) -> str:
    """Mount the production app and export its rendered screen through Textual."""

    app = CapsWriterTui(
        locale=locale,
        show_clock=False,
        recorder=UnavailableRecorder("optional microphone dependency not installed"),
    )
    async with app.run_test(size=(width, height)) as pilot:
        await pilot.pause()
        svg = remove_remote_font_sources(
            app.export_screenshot(title="CapsWriter TUI v2 — Textual workbench")
        )
    accessible_svg = add_accessibility_metadata(svg, DEFAULT_DESCRIPTION)
    return "\n".join(line.rstrip() for line in accessible_svg.splitlines()) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lang", type=normalize_locale, default="en")
    parser.add_argument("--width", type=int, default=140)
    parser.add_argument("--height", type=int, default=46)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.width < 100 or args.height < 30:
        raise SystemExit("screenshot dimensions must be at least 100x30 cells")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        asyncio.run(
            capture_svg(
                locale=args.lang,
                width=args.width,
                height=args.height,
            )
        ),
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
