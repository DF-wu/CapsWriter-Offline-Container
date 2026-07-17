# coding: utf-8

from __future__ import annotations

import ast
import math
import os
import re
import shutil
import tempfile
import time
import unittest
import wave
from pathlib import Path
from subprocess import DEVNULL, PIPE, TimeoutExpired
from types import SimpleNamespace
from typing import Optional, Tuple, Union
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
FILE_MANAGER_PATH = ROOT / "core" / "client" / "audio" / "file_manager.py"


class FakeStdin:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakePopen:
    def __init__(self, wait_responses=None, kill_error=None) -> None:
        self.stdin = FakeStdin()
        self.returncode = None
        self.killed = False
        self.wait_responses = list(wait_responses or [0])
        self.wait_timeouts = []
        self.kill_error = kill_error

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        response = self.wait_responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        self.returncode = response
        return response

    def kill(self) -> None:
        if self.kill_error is not None:
            raise self.kill_error
        self.killed = True


def load_file_manager_namespace() -> dict:
    source = FILE_MANAGER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(FILE_MANAGER_PATH))
    keep_names = {
        "AudioWriter",
        "CLIENT_AUDIO_FINISH_TIMEOUT_ENV",
        "DEFAULT_CLIENT_AUDIO_FINISH_TIMEOUT_SECONDS",
        "CLIENT_AUDIO_FINISH_KILL_GRACE_SECONDS",
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
        elif (
            isinstance(node, ast.FunctionDef)
            and node.name in {"audio_finish_timeout_seconds", "kill_process"}
        ):
            body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "AudioFileManager":
            body.append(node)

    namespace = {
        "Config": SimpleNamespace(audio_name_len=20),
        "DEVNULL": DEVNULL,
        "logger": SimpleNamespace(
            debug=lambda *_args, **_kwargs: None,
            error=lambda *_args, **_kwargs: None,
            warning=lambda *_args, **_kwargs: None,
        ),
        "math": math,
        "makedirs": lambda *_args, **_kwargs: None,
        "np": SimpleNamespace(ndarray=object, int16="int16"),
        "Optional": Optional,
        "os": os,
        "Path": Path,
        "PIPE": PIPE,
        "Popen": FakePopen,
        "re": re,
        "shutil": shutil,
        "tempfile": tempfile,
        "time": time,
        "TimeoutExpired": TimeoutExpired,
        "Tuple": Tuple,
        "Union": Union,
        "wave": wave,
    }
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(FILE_MANAGER_PATH), "exec"), namespace)
    return namespace


def build_manager(namespace: dict, process: FakePopen):
    manager = namespace["AudioFileManager"].__new__(namespace["AudioFileManager"])
    manager.file_path = Path("recording.mp3")
    manager.file_handle = process
    manager.channels = 1
    manager._has_ffmpeg = True
    return manager


class ClientAudioFileManagerTest(unittest.TestCase):
    def test_audio_finish_timeout_accepts_default_and_configured_values(self) -> None:
        namespace = load_file_manager_namespace()
        timeout_env = namespace["CLIENT_AUDIO_FINISH_TIMEOUT_ENV"]

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                namespace["audio_finish_timeout_seconds"](),
                namespace["DEFAULT_CLIENT_AUDIO_FINISH_TIMEOUT_SECONDS"],
            )

        with patch.dict(os.environ, {timeout_env: "4.5"}):
            self.assertEqual(namespace["audio_finish_timeout_seconds"](), 4.5)

    def test_audio_finish_timeout_rejects_invalid_values(self) -> None:
        namespace = load_file_manager_namespace()
        timeout_env = namespace["CLIENT_AUDIO_FINISH_TIMEOUT_ENV"]

        for value in ("bad", "0", "-2", "nan", "inf"):
            with self.subTest(value=value):
                with patch.dict(os.environ, {timeout_env: value}):
                    with self.assertRaises(ValueError):
                        namespace["audio_finish_timeout_seconds"]()

    def test_finish_waits_for_ffmpeg_with_configured_timeout(self) -> None:
        namespace = load_file_manager_namespace()
        process = FakePopen()
        manager = build_manager(namespace, process)

        with patch.dict(os.environ, {namespace["CLIENT_AUDIO_FINISH_TIMEOUT_ENV"]: "1.5"}):
            path = manager.finish()

        self.assertEqual(path, Path("recording.mp3"))
        self.assertTrue(process.stdin.closed)
        self.assertEqual(process.wait_timeouts, [1.5])
        self.assertFalse(process.killed)
        self.assertIsNone(manager.file_handle)

    def test_finish_timeout_kills_ffmpeg_with_bounded_wait(self) -> None:
        namespace = load_file_manager_namespace()
        process = FakePopen(
            [
                TimeoutExpired(["ffmpeg"], timeout=1.5),
                -9,
            ]
        )
        manager = build_manager(namespace, process)

        with patch.dict(os.environ, {namespace["CLIENT_AUDIO_FINISH_TIMEOUT_ENV"]: "1.5"}):
            path = manager.finish()

        self.assertEqual(path, Path("recording.mp3"))
        self.assertTrue(process.stdin.closed)
        self.assertTrue(process.killed)
        self.assertEqual(
            process.wait_timeouts,
            [1.5, namespace["CLIENT_AUDIO_FINISH_KILL_GRACE_SECONDS"]],
        )
        self.assertIsNone(manager.file_handle)

    def test_finish_timeout_preserves_bounded_wait_when_windows_denies_kill(self) -> None:
        namespace = load_file_manager_namespace()
        process = FakePopen(
            [
                TimeoutExpired(["ffmpeg"], timeout=1.5),
                0,
            ],
            kill_error=PermissionError("handle is closing"),
        )
        manager = build_manager(namespace, process)

        with patch.dict(os.environ, {namespace["CLIENT_AUDIO_FINISH_TIMEOUT_ENV"]: "1.5"}):
            path = manager.finish()

        self.assertEqual(path, Path("recording.mp3"))
        self.assertEqual(
            process.wait_timeouts,
            [1.5, namespace["CLIENT_AUDIO_FINISH_KILL_GRACE_SECONDS"]],
        )
        self.assertIsNone(manager.file_handle)

    def test_finish_invalid_timeout_uses_default_without_leaking_process(self) -> None:
        namespace = load_file_manager_namespace()
        process = FakePopen()
        manager = build_manager(namespace, process)

        with patch.dict(os.environ, {namespace["CLIENT_AUDIO_FINISH_TIMEOUT_ENV"]: "nan"}):
            path = manager.finish()

        self.assertEqual(path, Path("recording.mp3"))
        self.assertTrue(process.stdin.closed)
        self.assertEqual(
            process.wait_timeouts,
            [namespace["DEFAULT_CLIENT_AUDIO_FINISH_TIMEOUT_SECONDS"]],
        )
        self.assertFalse(process.killed)
        self.assertIsNone(manager.file_handle)


if __name__ == "__main__":
    unittest.main()
