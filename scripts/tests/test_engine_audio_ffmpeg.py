# coding: utf-8

from __future__ import annotations

import ast
import math
import os
import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
ENGINE_AUDIO_PATHS = [
    ROOT / "core" / "server" / "engines" / "qwen_asr_gguf" / "inference" / "audio.py",
    ROOT / "core" / "server" / "engines" / "force_aligner_gguf" / "inference" / "audio.py",
    ROOT / "core" / "server" / "engines" / "sensevoice_onnx" / "inference" / "audio.py",
    ROOT / "core" / "server" / "engines" / "fun_asr_gguf" / "inference" / "audio.py",
]


class FakeNumpy:
    float32 = "float32"

    def frombuffer(self, raw_bytes, dtype):
        return {"raw_bytes": raw_bytes, "dtype": dtype}


class FakeProcess:
    returncode = 0

    def __init__(self, responses):
        self.responses = list(responses)
        self.communicate_timeouts = []
        self.killed = False

    def communicate(self, timeout=None):
        self.communicate_timeouts.append(timeout)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def kill(self):
        self.killed = True


def load_audio_namespace(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    keep_names = {
        "ENGINE_FFMPEG_TIMEOUT_ENV",
        "DEFAULT_ENGINE_FFMPEG_TIMEOUT_SECONDS",
        "FFMPEG_KILL_GRACE_SECONDS",
        "MAX_FFMPEG_ERROR_CHARS",
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
            and node.name in {
                "_ffmpeg_timeout_seconds",
                "_stderr_preview",
                "check_ffmpeg",
                "load_audio",
                "load_audio_ffmpeg",
            }
        ):
            body.append(node)

    namespace = {
        "math": math,
        "np": FakeNumpy(),
        "os": os,
        "shutil": shutil,
        "subprocess": subprocess,
    }
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    namespace["check_ffmpeg"] = lambda: True
    if "load_audio_ffmpeg" not in namespace:
        namespace["load_audio_ffmpeg"] = namespace["load_audio"]
    return namespace


class EngineAudioFfmpegTest(unittest.TestCase):
    def test_ffmpeg_decode_uses_configured_timeout(self) -> None:
        for path in ENGINE_AUDIO_PATHS:
            with self.subTest(path=path):
                namespace = load_audio_namespace(path)
                process = FakeProcess([(b"\x00\x00\x00\x00", b"")])
                with (
                    patch.dict(
                        os.environ,
                        {namespace["ENGINE_FFMPEG_TIMEOUT_ENV"]: "3.5"},
                    ),
                    patch.object(namespace["subprocess"], "Popen", return_value=process) as popen,
                ):
                    audio = namespace["load_audio_ffmpeg"]("input.opus")

                self.assertEqual(audio["raw_bytes"], b"\x00\x00\x00\x00")
                self.assertEqual(audio["dtype"], "float32")
                self.assertEqual(process.communicate_timeouts, [3.5])
                self.assertFalse(process.killed)
                self.assertEqual(popen.call_args.args[0][0], "ffmpeg")

    def test_ffmpeg_decode_timeout_kills_process(self) -> None:
        for path in ENGINE_AUDIO_PATHS:
            with self.subTest(path=path):
                namespace = load_audio_namespace(path)
                process = FakeProcess(
                    [
                        subprocess.TimeoutExpired(["ffmpeg"], timeout=1.5),
                        (b"", b"still running"),
                    ]
                )
                with (
                    patch.dict(
                        os.environ,
                        {namespace["ENGINE_FFMPEG_TIMEOUT_ENV"]: "1.5"},
                    ),
                    patch.object(namespace["subprocess"], "Popen", return_value=process),
                ):
                    with self.assertRaisesRegex(RuntimeError, "ffmpeg .*1.5s"):
                        namespace["load_audio_ffmpeg"]("input.opus")

                self.assertTrue(process.killed)
                self.assertEqual(process.communicate_timeouts, [1.5, 2.0])

    def test_invalid_ffmpeg_timeout_rejects_before_spawning(self) -> None:
        for path in ENGINE_AUDIO_PATHS:
            with self.subTest(path=path):
                namespace = load_audio_namespace(path)
                with (
                    patch.dict(
                        os.environ,
                        {namespace["ENGINE_FFMPEG_TIMEOUT_ENV"]: "nan"},
                    ),
                    patch.object(namespace["subprocess"], "Popen") as popen,
                ):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "CAPSWRITER_ENGINE_FFMPEG_TIMEOUT must be a finite number",
                    ):
                        namespace["load_audio_ffmpeg"]("input.opus")

                popen.assert_not_called()

    def test_ffmpeg_error_preview_is_bounded(self) -> None:
        for path in ENGINE_AUDIO_PATHS:
            with self.subTest(path=path):
                namespace = load_audio_namespace(path)
                process = FakeProcess([(b"", b"x" * 1500)])
                process.returncode = 1
                with (
                    patch.dict(os.environ, {}, clear=False),
                    patch.object(namespace["subprocess"], "Popen", return_value=process),
                ):
                    os.environ.pop(namespace["ENGINE_FFMPEG_TIMEOUT_ENV"], None)
                    with self.assertRaises(RuntimeError) as ctx:
                        namespace["load_audio_ffmpeg"]("input.opus")

                message = str(ctx.exception)
                self.assertTrue(message.endswith("..."))
                self.assertLessEqual(
                    len(message),
                    len("ffmpeg 处理音频失败: ")
                    + namespace["MAX_FFMPEG_ERROR_CHARS"]
                    + 3,
                )


if __name__ == "__main__":
    unittest.main()
