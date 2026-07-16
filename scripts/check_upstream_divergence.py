#!/usr/bin/env python3
# coding: utf-8
"""Guard the fork's low-drift contract against upstream-tracked file changes."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_REF = "origin/master"
BASE_ENV = "CAPSWRITER_UPSTREAM_BASE"
GIT_TIMEOUT_SECONDS = 15.0
ALLOWED_UPSTREAM_DIVERGENCE = frozenset(
    {
        ".gitignore",
        "CLAUDE.md",
        "LLM/default.py",
        "LLM/大助理.py",
        "assets/BUILD_GUIDE.md",
        "build.spec",
        "core/client/audio/file_manager.py",
        "core/client/audio/recorder.py",
        "core/client/audio/stream.py",
        "core/client/clipboard/clipboard.py",
        "core/client/connection/websocket_manager.py",
        "core/client/global_hotkey/__init__.py",
        "core/client/global_hotkey/global_hotkey.py",
        "core/client/hotword/hotword_standalone.ipynb",
        "core/client/hotword/hotword_standalone.py",
        "core/client/llm/llm_output_typing.py",
        "core/client/manager/file_runner.py",
        "core/client/manager/tray_manager.py",
        "core/client/output/result_processor.py",
        "core/client/output/text_output.py",
        "core/client/shortcut/emulator.py",
        "core/client/shortcut/key_mapper.py",
        "core/client/shortcut/shortcut_manager.py",
        "core/client/state.py",
        "core/client/transcribe/file_transcriber.py",
        "core/client/transcribe/media_tool.py",
        "core/client/transcribe/srt_adjuster.py",
        "core/protocol.py",
        "core/server/app.py",
        "core/server/connection/server_manager.py",
        "core/server/connection/ws_recv.py",
        "core/server/connection/ws_send.py",
        "core/server/engines/force_aligner_gguf/export/gguf/utility.py",
        "core/server/engines/force_aligner_gguf/inference/aligner.py",
        "core/server/engines/force_aligner_gguf/inference/audio.py",
        "core/server/engines/fun_asr_gguf/export/gguf/utility.py",
        "core/server/engines/fun_asr_gguf/inference/audio.py",
        "core/server/engines/fun_asr_gguf/inference/prompt_builder.py",
        "core/server/engines/qwen_asr_gguf/export/gguf/utility.py",
        "core/server/engines/qwen_asr_gguf/inference/aligner.py",
        "core/server/engines/qwen_asr_gguf/inference/audio.py",
        "core/server/engines/sensevoice_onnx/inference/audio.py",
        "core/server/schema.py",
        "core/server/state.py",
        "core/server/worker/__init__.py",
        "core/server/worker/gpu_boost.py",
        "core/server/worker/pipeline.py",
        "core/server/worker/process_manager.py",
        "core/server/worker/task_handler.py",
        "core/server/worker/worker.py",
        "core/tools/window_detector.py",
        "core/ui/tray.py",
        "docs/text_merge_algorithm.md",
        "docs/显卡加速的若干问题.md",
        "docs/角色功能如何使用.md",
        "readme.md",
        "requirements-server.txt",
        "start_client.py",
        "zip_release.py",
    }
)


def git_output(args: list[str], *, cwd: Path = ROOT) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"git {' '.join(args)} timed out after {GIT_TIMEOUT_SECONDS:g}s"
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout


def git_success(args: list[str], *, cwd: Path = ROOT) -> bool:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"git {' '.join(args)} timed out after {GIT_TIMEOUT_SECONDS:g}s"
        ) from exc
    return completed.returncode == 0


def ref_exists(ref: str, *, cwd: Path = ROOT) -> bool:
    return git_success(
        ["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        cwd=cwd,
    )


def path_exists_in_ref(ref: str, path: str, *, cwd: Path = ROOT) -> bool:
    return git_success(["cat-file", "-e", f"{ref}:{path}"], cwd=cwd)


def changed_paths(base_ref: str, *, cwd: Path = ROOT) -> set[str]:
    """Return tracked changes from the base commit through the working tree.

    Deliberately use ``git diff <base> --`` rather than ``<base>..HEAD`` so
    staged and unstaged tracked edits participate in the same guard.
    """
    output = git_output(
        [
            "diff",
            "--name-status",
            "-z",
            "--find-renames",
            "--find-copies",
            base_ref,
            "--",
        ],
        cwd=cwd,
    )
    fields = output.split("\0")
    if fields and fields[-1] == "":
        fields.pop()

    # With -z, Git emits status\0path\0, or status\0old\0new\0 for
    # rename/copy records. Paths remain unquoted and may contain tabs/newlines.
    paths: set[str] = set()
    index = 0
    while index < len(fields):
        status = fields[index]
        index += 1
        path_count = 2 if status.startswith(("R", "C")) else 1
        record_paths = fields[index : index + path_count]
        if not status or len(record_paths) != path_count or any(
            not path for path in record_paths
        ):
            raise RuntimeError(
                "git diff returned malformed NUL-delimited name-status output"
            )
        paths.update(record_paths)
        index += path_count
    return paths


def upstream_tracked_changes(base_ref: str, *, cwd: Path = ROOT) -> list[str]:
    upstream_paths = {
        path
        for path in changed_paths(base_ref, cwd=cwd)
        if path_exists_in_ref(base_ref, path, cwd=cwd)
    }
    return sorted(upstream_paths)


def unexpected_changes(paths: list[str], allowed: frozenset[str]) -> list[str]:
    return [path for path in paths if path not in allowed]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check that the fork working tree diverges only from known "
            "upstream files"
        ),
    )
    parser.add_argument(
        "--base",
        default=os.environ.get(BASE_ENV, DEFAULT_BASE_REF),
        help=(
            "Upstream base ref to compare against "
            f"(default: {BASE_ENV} or {DEFAULT_BASE_REF})"
        ),
    )
    parser.add_argument(
        "--require-base",
        action="store_true",
        help="Fail instead of skipping when --base is unavailable",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if not ref_exists(args.base):
            message = (
                "Upstream divergence guard skipped: "
                f"base ref {args.base!r} is not available"
            )
            if args.require_base:
                print(message, file=sys.stderr)
                return 1
            print(message)
            return 0

        tracked = upstream_tracked_changes(args.base)
    except RuntimeError as error:
        print(f"Upstream divergence guard failed: {error}", file=sys.stderr)
        return 1

    unexpected = unexpected_changes(tracked, ALLOWED_UPSTREAM_DIVERGENCE)
    if unexpected:
        print(
            "Unexpected upstream-tracked file divergence detected:",
            file=sys.stderr,
        )
        for path in unexpected:
            print(f"  {path}", file=sys.stderr)
        print(
            "Either move fork-only changes into fork-owned paths or document the "
            "new divergence in docs/architecture.md and docs/upstream-sync-guide.md.",
            file=sys.stderr,
        )
        return 1

    print(
        "Upstream divergence guard passed: "
        f"{len(tracked)} upstream-tracked file(s) changed"
    )
    for path in tracked:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
