#!/usr/bin/env python3
# coding: utf-8
"""Validate local Markdown links and basic image accessibility.

The checker intentionally uses only the standard library.  It validates tracked
and untracked, non-ignored documentation so new pages are checked before their
first commit while model trees and dependency directories stay out of scope.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import html
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
GIT_TIMEOUT_SECONDS = 15.0
INLINE_LINK_RE = re.compile(r"(?P<image>!)?\[(?P<label>[^\]]*)\]\((?P<target>[^\n)]*)\)")
HTML_IMAGE_RE = re.compile(r"<img\b(?P<attrs>[^>]*)>", re.IGNORECASE)
HTML_ATTR_RE = re.compile(
    r"(?P<name>[A-Za-z_:][-A-Za-z0-9_:.]*)\s*=\s*(?P<quote>['\"])(?P<value>.*?)\2",
    re.DOTALL,
)
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(?P<label>.+?)\s*#*\s*$")
INLINE_CODE_RE = re.compile(r"(?P<ticks>`+)(?P<body>.*?)(?P=ticks)", re.DOTALL)
HTML_TAG_RE = re.compile(r"<[^>]+>")
EXTERNAL_SCHEMES = {"http", "https", "mailto", "tel", "data"}
IMAGE_SUFFIXES = {".avif", ".gif", ".jpeg", ".jpg", ".png", ".svg", ".webp"}


@dataclass(frozen=True, order=True)
class DocIssue:
    path: Path
    line: int
    message: str

    def render(self, root: Path = ROOT) -> str:
        try:
            display = self.path.relative_to(root)
        except ValueError:
            display = self.path
        return f"{display}:{self.line}: {self.message}"


def tracked_markdown_files(root: Path = ROOT) -> list[Path]:
    try:
        completed = subprocess.run(
            [
                "git",
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
                "--",
                "*.md",
            ],
            cwd=root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"git ls-files timed out after {GIT_TIMEOUT_SECONDS:g}s"
        ) from exc
    if completed.returncode != 0:
        detail = os.fsdecode(completed.stderr).strip() or "unknown git error"
        raise RuntimeError(f"git ls-files failed: {detail}")
    return sorted(
        root / os.fsdecode(item)
        for item in completed.stdout.split(b"\0")
        if item
    )


def _target_value(raw: str) -> str:
    value = html.unescape(raw.strip())
    if value.startswith("<") and ">" in value:
        return value[1 : value.index(">")]
    # Markdown permits an optional quoted title after a whitespace separator.
    return value.split(maxsplit=1)[0] if value else ""


def _github_slug(label: str) -> str:
    value = html.unescape(HTML_TAG_RE.sub("", label)).strip().casefold()
    value = re.sub(r"[^\w\- ]", "", value, flags=re.UNICODE)
    return re.sub(r"[ ]+", "-", value)


def heading_anchors(source: str) -> set[str]:
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    in_fence = False
    fence_marker = ""
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
            continue
        if in_fence:
            continue
        match = HEADING_RE.match(line)
        if not match:
            continue
        base = _github_slug(match.group("label"))
        occurrence = counts.get(base, 0)
        counts[base] = occurrence + 1
        anchors.add(base if occurrence == 0 else f"{base}-{occurrence}")
    return anchors


def _line_number(source: str, offset: int) -> int:
    return source.count("\n", 0, offset) + 1


def mask_markdown_code(source: str) -> str:
    """Replace fenced/inline code characters with spaces, retaining offsets."""

    masked = list(source)
    in_fence = False
    fence_marker = ""
    offset = 0
    for line in source.splitlines(keepends=True):
        stripped = line.lstrip()
        marker = stripped[:3] if stripped.startswith(("```", "~~~")) else ""
        should_mask = in_fence or bool(marker)
        if marker:
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
        if should_mask:
            for index in range(offset, offset + len(line)):
                if masked[index] not in {"\n", "\r"}:
                    masked[index] = " "
        offset += len(line)

    visible = "".join(masked)
    for match in INLINE_CODE_RE.finditer(visible):
        for index in range(match.start(), match.end()):
            if masked[index] not in {"\n", "\r"}:
                masked[index] = " "
    return "".join(masked)


def _looks_like_filename_alt(alt: str) -> bool:
    value = alt.strip()
    suffix = Path(value).suffix.casefold()
    return suffix in IMAGE_SUFFIXES and Path(value).name == value


def _resolve_local_target(document: Path, target: str, root: Path) -> tuple[Path, str]:
    parsed = urlsplit(target)
    path_text = unquote(parsed.path)
    if path_text:
        candidate = (document.parent / path_text).resolve()
    else:
        candidate = document.resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("local target escapes the repository") from exc
    return candidate, unquote(parsed.fragment).casefold()


def validate_markdown_file(path: Path, root: Path = ROOT) -> list[DocIssue]:
    source = path.read_text(encoding="utf-8")
    visible_source = mask_markdown_code(source)
    issues: list[DocIssue] = []
    anchor_cache: dict[Path, set[str]] = {}
    svg_accessibility_cache: dict[Path, tuple[bool, bool]] = {}

    for match in INLINE_LINK_RE.finditer(visible_source):
        line = _line_number(source, match.start())
        target = _target_value(match.group("target"))
        is_image = bool(match.group("image"))
        label = match.group("label").strip()

        if is_image:
            if not label:
                issues.append(DocIssue(path, line, "image is missing descriptive alt text"))
            elif _looks_like_filename_alt(label):
                issues.append(DocIssue(path, line, "image alt text is only a filename"))

        if not target:
            issues.append(DocIssue(path, line, "link target is empty"))
            continue
        parsed = urlsplit(target)
        if parsed.scheme.casefold() in EXTERNAL_SCHEMES or target.startswith("//"):
            continue
        if parsed.scheme:
            # Non-web schemes (for example vscode:) are not portable docs links.
            issues.append(DocIssue(path, line, f"unsupported link scheme: {parsed.scheme}"))
            continue

        try:
            candidate, fragment = _resolve_local_target(path, target, root)
        except ValueError as exc:
            issues.append(DocIssue(path, line, str(exc)))
            continue
        if not candidate.exists():
            issues.append(DocIssue(path, line, f"local target does not exist: {target}"))
            continue
        if is_image and candidate.suffix.casefold() == ".svg":
            accessibility = svg_accessibility_cache.get(candidate)
            if accessibility is None:
                svg_source = candidate.read_text(encoding="utf-8")
                accessibility = (
                    bool(re.search(r"<title(?:\s|>)", svg_source, re.IGNORECASE)),
                    bool(re.search(r"<desc(?:\s|>)", svg_source, re.IGNORECASE)),
                )
                svg_accessibility_cache[candidate] = accessibility
            if not accessibility[0]:
                issues.append(DocIssue(path, line, f"referenced SVG has no <title>: {target}"))
            if not accessibility[1]:
                issues.append(DocIssue(path, line, f"referenced SVG has no <desc>: {target}"))
        if fragment and candidate.suffix.casefold() == ".md":
            anchors = anchor_cache.get(candidate)
            if anchors is None:
                anchors = heading_anchors(candidate.read_text(encoding="utf-8"))
                anchor_cache[candidate] = anchors
            if fragment not in anchors:
                issues.append(DocIssue(path, line, f"Markdown anchor does not exist: #{fragment}"))

    for match in HTML_IMAGE_RE.finditer(visible_source):
        attrs = {
            attr.group("name").casefold(): attr.group("value")
            for attr in HTML_ATTR_RE.finditer(match.group("attrs"))
        }
        line = _line_number(source, match.start())
        alt = attrs.get("alt", "").strip()
        if not alt:
            issues.append(DocIssue(path, line, "HTML image is missing descriptive alt text"))
        elif _looks_like_filename_alt(alt):
            issues.append(DocIssue(path, line, "HTML image alt text is only a filename"))

    return issues


def validate_docs(paths: list[Path], root: Path = ROOT) -> list[DocIssue]:
    issues: list[DocIssue] = []
    for path in paths:
        issues.extend(validate_markdown_file(path, root))
    path_set = {path.resolve() for path in paths}
    language_roots = {
        "en": root / "docs" / "en",
        "zh-TW": root / "docs" / "zh-TW",
    }
    for language, language_root in language_roots.items():
        counterpart_language = "zh-TW" if language == "en" else "en"
        counterpart_root = language_roots[counterpart_language]
        for path in path_set:
            try:
                relative = path.relative_to(language_root.resolve())
            except ValueError:
                continue
            counterpart = (counterpart_root / relative).resolve()
            if counterpart not in path_set or not counterpart.exists():
                issues.append(
                    DocIssue(
                        path,
                        1,
                        f"missing {counterpart_language} counterpart: "
                        f"{counterpart.relative_to(root.resolve())}",
                    )
                )
    return sorted(issues)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate tracked Markdown links, anchors, and image alt text"
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Optional Markdown paths")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        paths = [path.resolve() for path in args.paths] or tracked_markdown_files()
        issues = validate_docs(paths)
    except (OSError, UnicodeError, RuntimeError) as exc:
        print(f"Documentation check failed: {exc}", file=sys.stderr)
        return 2
    if issues:
        print(f"Documentation check found {len(issues)} issue(s):", file=sys.stderr)
        for issue in issues:
            print(f"  {issue.render()}", file=sys.stderr)
        return 1
    print(f"Documentation check passed: {len(paths)} Markdown file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
