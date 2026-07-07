# coding: utf-8

from __future__ import annotations

import asyncio
import importlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch


def load_audio_decoder_module():
    class AudioFormat:
        CHANNELS = 1
        SAMPLE_RATE = 16000

    core_module = types.ModuleType("core")
    constants_module = types.ModuleType("core.constants")
    constants_module.AudioFormat = AudioFormat
    server_module = types.ModuleType("core.server")
    server_module.logger = SimpleNamespace(warning=Mock())

    fake_modules = {
        "core": core_module,
        "core.constants": constants_module,
        "core.server": server_module,
    }

    module_name = "fork_server.http_api.audio_decoder"
    sys.modules.pop(module_name, None)
    with patch.dict(sys.modules, fake_modules):
        return importlib.import_module(module_name)


class CompletedProcess:
    def __init__(self, stderr: bytes, returncode: int = 1) -> None:
        self.stderr = stderr
        self.returncode = returncode

    async def communicate(self, input: bytes | None = None):
        del input
        return b"", self.stderr


class HangingProcess:
    returncode = None

    def __init__(self) -> None:
        self.killed = False
        self.waited = False

    async def communicate(self, input: bytes | None = None):
        del input
        await asyncio.sleep(10)
        return b"", b""

    def kill(self) -> None:
        self.killed = True

    async def wait(self):
        self.waited = True
        return -9


class StubbornProcess(HangingProcess):
    async def wait(self):
        self.waited = True
        await asyncio.sleep(10)
        return -9


class AudioDecoderTest(unittest.TestCase):
    def tearDown(self) -> None:
        sys.modules.pop("fork_server.http_api.audio_decoder", None)

    def test_ffmpeg_error_output_is_bounded(self) -> None:
        audio_decoder = load_audio_decoder_module()
        stderr = ("bad " * 400 + "secret-tail").encode("utf-8")
        proc = CompletedProcess(stderr)

        async def fake_exec(*args, **kwargs):
            del args, kwargs
            return proc

        with (
            patch.object(audio_decoder.shutil, "which", return_value="/usr/bin/ffmpeg"),
            patch.object(
                audio_decoder.asyncio,
                "create_subprocess_exec",
                new=fake_exec,
            ),
        ):
            with self.assertRaises(audio_decoder.AudioDecodeError) as ctx:
                asyncio.run(audio_decoder.decode_to_pcm(b"audio"))

        message = str(ctx.exception)
        self.assertLessEqual(
            len(message),
            len("ffmpeg failed: ") + audio_decoder.MAX_FFMPEG_ERROR_CHARS + 3,
        )
        self.assertTrue(message.endswith("..."))
        self.assertNotIn("secret-tail", message)

    def test_decode_timeout_kills_ffmpeg_process(self) -> None:
        audio_decoder = load_audio_decoder_module()
        proc = HangingProcess()

        async def fake_exec(*args, **kwargs):
            del args, kwargs
            return proc

        with (
            patch.object(audio_decoder.shutil, "which", return_value="/usr/bin/ffmpeg"),
            patch.object(
                audio_decoder.asyncio,
                "create_subprocess_exec",
                new=fake_exec,
            ),
        ):
            with self.assertRaises(audio_decoder.AudioDecodeError):
                asyncio.run(audio_decoder.decode_to_pcm(b"audio", timeout=0.01))

        self.assertTrue(proc.killed)
        self.assertTrue(proc.waited)

    def test_invalid_timeout_is_rejected_before_spawning_ffmpeg(self) -> None:
        audio_decoder = load_audio_decoder_module()
        spawn = AsyncMock()

        with (
            patch.object(audio_decoder.shutil, "which") as which,
            patch.object(
                audio_decoder.asyncio,
                "create_subprocess_exec",
                new=spawn,
            ),
        ):
            with self.assertRaisesRegex(
                audio_decoder.AudioDecodeError,
                "positive finite",
            ):
                asyncio.run(audio_decoder.decode_to_pcm(b"audio", timeout=0))

        which.assert_not_called()
        spawn.assert_not_awaited()

    def test_timeout_cleanup_wait_is_bounded(self) -> None:
        audio_decoder = load_audio_decoder_module()
        proc = StubbornProcess()

        async def fake_exec(*args, **kwargs):
            del args, kwargs
            return proc

        with (
            patch.object(audio_decoder.shutil, "which", return_value="/usr/bin/ffmpeg"),
            patch.object(
                audio_decoder.asyncio,
                "create_subprocess_exec",
                new=fake_exec,
            ),
            patch.object(audio_decoder, "FFMPEG_KILL_GRACE_SECONDS", 0.01),
        ):
            with self.assertRaisesRegex(
                audio_decoder.AudioDecodeError,
                "did not exit after kill",
            ):
                asyncio.run(audio_decoder.decode_to_pcm(b"audio", timeout=0.01))

        self.assertTrue(proc.killed)
        self.assertTrue(proc.waited)


if __name__ == "__main__":
    unittest.main()
