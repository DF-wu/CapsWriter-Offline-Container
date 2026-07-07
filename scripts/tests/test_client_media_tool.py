# coding: utf-8

from __future__ import annotations

import ast
import asyncio
import math
import os
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
MEDIA_TOOL_PATH = ROOT / "core" / "client" / "transcribe" / "media_tool.py"
FILE_TRANSCRIBER_PATH = (
    ROOT / "core" / "client" / "transcribe" / "file_transcriber.py"
)


def load_media_tool_namespace() -> dict:
    source = MEDIA_TOOL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(MEDIA_TOOL_PATH))
    keep_names = {
        "CLIENT_MEDIA_TIMEOUT_ENV",
        "DEFAULT_CLIENT_MEDIA_TIMEOUT_SECONDS",
        "CLIENT_MEDIA_KILL_GRACE_SECONDS",
    }
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = {
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            if names & keep_names:
                body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "MediaTool":
            body.append(node)

    namespace = {
        "asyncio": asyncio,
        "console": SimpleNamespace(print=lambda *_args, **_kwargs: None),
        "List": List,
        "logger": SimpleNamespace(
            error=lambda *_args, **_kwargs: None,
            warning=lambda *_args, **_kwargs: None,
        ),
        "math": math,
        "Optional": Optional,
        "os": os,
        "Path": Path,
        "shutil": shutil,
    }
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(MEDIA_TOOL_PATH), "exec"), namespace)
    return namespace


def find_send_method() -> ast.AsyncFunctionDef:
    source = FILE_TRANSCRIBER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(FILE_TRANSCRIBER_PATH))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "FileTranscriber":
            for item in node.body:
                if isinstance(item, ast.AsyncFunctionDef) and item.name == "send":
                    return item
    raise AssertionError("FileTranscriber.send was not found")


def is_attr(node: ast.AST, *, value_name: str, attr: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == attr
        and isinstance(node.value, ast.Name)
        and node.value.id == value_name
    )


class FakeDurationProcess:
    def __init__(self, stdout: bytes = b"12.5\n") -> None:
        self.returncode = 0
        self.stdout = stdout
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        return self.stdout, b""

    async def wait(self) -> int:
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


class HangingDurationProcess:
    def __init__(self) -> None:
        self.returncode = None
        self.killed = False
        self.waited = False

    async def communicate(self) -> tuple[bytes, bytes]:
        await asyncio.sleep(60)
        return b"", b""

    async def wait(self) -> int:
        self.waited = True
        return self.returncode or -9

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


class ClientMediaToolTest(unittest.TestCase):
    def test_timeout_seconds_accepts_default_and_configured_values(self) -> None:
        namespace = load_media_tool_namespace()
        media_tool = namespace["MediaTool"]
        timeout_env = namespace["CLIENT_MEDIA_TIMEOUT_ENV"]

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                media_tool.timeout_seconds(),
                namespace["DEFAULT_CLIENT_MEDIA_TIMEOUT_SECONDS"],
            )

        with patch.dict(os.environ, {timeout_env: "2.5"}):
            self.assertEqual(media_tool.timeout_seconds(), 2.5)

    def test_timeout_seconds_rejects_invalid_values(self) -> None:
        namespace = load_media_tool_namespace()
        media_tool = namespace["MediaTool"]
        timeout_env = namespace["CLIENT_MEDIA_TIMEOUT_ENV"]

        for value in ("not-a-number", "0", "-1", "nan", "inf"):
            with self.subTest(value=value):
                with patch.dict(os.environ, {timeout_env: value}):
                    with self.assertRaises(ValueError):
                        media_tool.timeout_seconds()

    def test_get_audio_duration_uses_configured_timeout(self) -> None:
        namespace = load_media_tool_namespace()
        media_tool = namespace["MediaTool"]
        process = FakeDurationProcess()
        wait_for_timeouts = []

        async def fake_create_subprocess_exec(*_args, **_kwargs):
            return process

        original_wait_for = asyncio.wait_for

        async def recording_wait_for(awaitable, timeout=None):
            wait_for_timeouts.append(timeout)
            return await original_wait_for(awaitable, timeout=timeout)

        with (
            patch.dict(os.environ, {namespace["CLIENT_MEDIA_TIMEOUT_ENV"]: "1.25"}),
            patch.object(
                namespace["asyncio"],
                "create_subprocess_exec",
                side_effect=fake_create_subprocess_exec,
            ),
            patch.object(namespace["asyncio"], "wait_for", side_effect=recording_wait_for),
        ):
            duration = asyncio.run(media_tool.get_audio_duration(Path("input.wav")))

        self.assertEqual(duration, 12.5)
        self.assertEqual(wait_for_timeouts, [1.25])
        self.assertFalse(process.killed)

    def test_get_audio_duration_rejects_invalid_timeout_before_spawning(self) -> None:
        namespace = load_media_tool_namespace()
        media_tool = namespace["MediaTool"]

        with (
            patch.dict(os.environ, {namespace["CLIENT_MEDIA_TIMEOUT_ENV"]: "nan"}),
            patch.object(namespace["asyncio"], "create_subprocess_exec") as create,
        ):
            duration = asyncio.run(media_tool.get_audio_duration(Path("input.wav")))

        self.assertEqual(duration, 0.0)
        create.assert_not_called()

    def test_get_audio_duration_timeout_kills_process(self) -> None:
        namespace = load_media_tool_namespace()
        media_tool = namespace["MediaTool"]
        process = HangingDurationProcess()
        wait_for_timeouts = []

        async def fake_create_subprocess_exec(*_args, **_kwargs):
            return process

        original_wait_for = asyncio.wait_for

        async def recording_wait_for(awaitable, timeout=None):
            wait_for_timeouts.append(timeout)
            return await original_wait_for(awaitable, timeout=timeout)

        with (
            patch.dict(os.environ, {namespace["CLIENT_MEDIA_TIMEOUT_ENV"]: "0.01"}),
            patch.object(
                namespace["asyncio"],
                "create_subprocess_exec",
                side_effect=fake_create_subprocess_exec,
            ),
            patch.object(namespace["asyncio"], "wait_for", side_effect=recording_wait_for),
        ):
            duration = asyncio.run(media_tool.get_audio_duration(Path("input.wav")))

        self.assertEqual(duration, 0.0)
        self.assertTrue(process.killed)
        self.assertTrue(process.waited)
        self.assertEqual(
            wait_for_timeouts,
            [0.01, namespace["CLIENT_MEDIA_KILL_GRACE_SECONDS"]],
        )


class ClientFileTranscriberSourceGuardTest(unittest.TestCase):
    def test_send_bounds_ffmpeg_stdout_reads_and_final_wait(self) -> None:
        send = find_send_method()
        wait_for_calls = [
            node
            for node in ast.walk(send)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "wait_for"
        ]

        has_stdout_read_timeout = False
        has_process_wait_timeout = False
        for call in wait_for_calls:
            if not call.args or not any(kw.arg == "timeout" for kw in call.keywords):
                continue
            target = call.args[0]
            if (
                isinstance(target, ast.Call)
                and isinstance(target.func, ast.Attribute)
                and target.func.attr == "read"
                and isinstance(target.func.value, ast.Attribute)
                and is_attr(target.func.value, value_name="process", attr="stdout")
            ):
                has_stdout_read_timeout = True
            if (
                isinstance(target, ast.Call)
                and isinstance(target.func, ast.Attribute)
                and is_attr(target.func, value_name="process", attr="wait")
            ):
                has_process_wait_timeout = True

        self.assertTrue(has_stdout_read_timeout)
        self.assertTrue(has_process_wait_timeout)

    def test_send_uses_bounded_kill_cleanup_and_no_bare_terminate(self) -> None:
        send = find_send_method()
        kill_calls = [
            node
            for node in ast.walk(send)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "kill_process"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "MediaTool"
        ]
        terminate_calls = [
            node
            for node in ast.walk(send)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "terminate"
        ]

        self.assertGreaterEqual(len(kill_calls), 3)
        self.assertEqual(terminate_calls, [])

    def test_send_initializes_progress_before_stream_loop(self) -> None:
        send = find_send_method()
        has_zero_progress_assignment = any(
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "progress"
                for target in node.targets
            )
            and isinstance(node.value, ast.Constant)
            and node.value.value == 0.0
            for node in ast.walk(send)
        )

        self.assertTrue(has_zero_progress_assignment)


if __name__ == "__main__":
    unittest.main()
