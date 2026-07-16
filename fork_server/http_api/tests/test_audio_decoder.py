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
        BYTES_PER_SAMPLE = 4
        BYTES_PER_SECOND = SAMPLE_RATE * BYTES_PER_SAMPLE * CHANNELS

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
    def __init__(
        self,
        stderr: bytes,
        returncode: int = 1,
        stdout: bytes = b"",
    ) -> None:
        self.stderr = stderr
        self.returncode = returncode
        self.stdout = stdout

    async def communicate(self, input: bytes | None = None):
        del input
        return self.stdout, self.stderr


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


class KillDeniedProcess(StubbornProcess):
    def kill(self) -> None:
        self.killed = True
        raise PermissionError("process handle is closing")


class ChunkStream:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = list(chunks)
        self.reads = 0

    async def read(self, size: int = -1) -> bytes:
        del size
        self.reads += 1
        if not self.chunks:
            return b""
        return self.chunks.pop(0)


class BlockingWriter:
    def __init__(self) -> None:
        self.closed = False
        self.drain_started = False

    def write(self, data) -> None:
        del data

    async def drain(self) -> None:
        self.drain_started = True
        await asyncio.Event().wait()

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


class BlockingCloseWriter(BlockingWriter):
    def __init__(self) -> None:
        super().__init__()
        self.wait_closed_started = False

    async def wait_closed(self) -> None:
        self.wait_closed_started = True
        await asyncio.Event().wait()


class AbortedCloseWriter(BlockingWriter):
    async def drain(self) -> None:
        self.drain_started = True

    async def wait_closed(self) -> None:
        raise ConnectionAbortedError("Windows pipe closed")


class StreamingProcess:
    def __init__(self, writer=None) -> None:
        self.stdin = writer or BlockingWriter()
        self.stdout = ChunkStream([b"x" * 11])
        self.stderr = ChunkStream([])

    async def wait(self):
        return 0


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
        warning = audio_decoder.logger.warning.call_args.args[0]
        self.assertIn("details=<redacted>", warning)
        self.assertNotIn("bad", warning)
        self.assertNotIn("secret-tail", warning)

    def test_ffmpeg_error_preview_removes_terminal_controls(self) -> None:
        audio_decoder = load_audio_decoder_module()

        preview = audio_decoder._stderr_preview(
            b"first\n\x1b[31mforged\x00\rsecond"
        )

        self.assertNotIn("\x1b", preview)
        self.assertNotIn("\x00", preview)
        self.assertEqual(preview, "first [31mforged second")

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

    def test_windows_kill_permission_error_preserves_timeout_contract(self) -> None:
        audio_decoder = load_audio_decoder_module()
        proc = KillDeniedProcess()

        async def fake_exec(*args, **kwargs):
            del args, kwargs
            return proc

        with (
            patch.object(audio_decoder.shutil, "which", return_value="ffmpeg.exe"),
            patch.object(audio_decoder.asyncio, "create_subprocess_exec", new=fake_exec),
            patch.object(audio_decoder, "FFMPEG_KILL_GRACE_SECONDS", 0.01),
        ):
            with self.assertRaisesRegex(
                audio_decoder.AudioDecodeTimeoutError,
                "did not exit after kill",
            ):
                asyncio.run(audio_decoder.decode_to_pcm(b"audio", timeout=0.01))

        self.assertTrue(proc.killed)
        self.assertTrue(proc.waited)

    def test_decoded_output_limit_is_passed_to_ffmpeg_and_enforced(self) -> None:
        audio_decoder = load_audio_decoder_module()
        proc = CompletedProcess(b"", returncode=0, stdout=b"x" * 11)
        command: list[str] = []

        async def fake_exec(*args, **kwargs):
            del kwargs
            command.extend(args)
            return proc

        with (
            patch.object(audio_decoder.shutil, "which", return_value="/usr/bin/ffmpeg"),
            patch.object(audio_decoder.asyncio, "create_subprocess_exec", new=fake_exec),
        ):
            with self.assertRaises(audio_decoder.AudioTooLongError):
                asyncio.run(
                    audio_decoder.decode_to_pcm(
                        b"audio",
                        max_output_bytes=10,
                    )
                )

        self.assertIn("-t", command)

    def test_outer_cancellation_kills_ffmpeg_process(self) -> None:
        audio_decoder = load_audio_decoder_module()
        proc = HangingProcess()

        async def fake_exec(*args, **kwargs):
            del args, kwargs
            return proc

        async def scenario() -> None:
            with (
                patch.object(
                    audio_decoder.shutil,
                    "which",
                    return_value="/usr/bin/ffmpeg",
                ),
                patch.object(
                    audio_decoder.asyncio,
                    "create_subprocess_exec",
                    new=fake_exec,
                ),
            ):
                task = asyncio.create_task(
                    audio_decoder.decode_to_pcm(b"audio", timeout=60)
                )
                await asyncio.sleep(0)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

        asyncio.run(scenario())
        self.assertTrue(proc.killed)
        self.assertTrue(proc.waited)

    def test_repeated_outer_cancellation_still_reaps_ffmpeg_process(self) -> None:
        audio_decoder = load_audio_decoder_module()

        async def scenario() -> None:
            reap_started = asyncio.Event()
            release_reap = asyncio.Event()

            class DelayedReapProcess(HangingProcess):
                def __init__(self) -> None:
                    super().__init__()
                    self.reaped = False
                    self.wait_cancelled = False

                async def wait(self):
                    self.waited = True
                    reap_started.set()
                    try:
                        await release_reap.wait()
                    except asyncio.CancelledError:
                        self.wait_cancelled = True
                        raise
                    self.reaped = True
                    return -9

            proc = DelayedReapProcess()

            async def fake_exec(*args, **kwargs):
                del args, kwargs
                return proc

            with (
                patch.object(
                    audio_decoder.shutil,
                    "which",
                    return_value="/usr/bin/ffmpeg",
                ),
                patch.object(
                    audio_decoder.asyncio,
                    "create_subprocess_exec",
                    new=fake_exec,
                ),
            ):
                task = asyncio.create_task(
                    audio_decoder.decode_to_pcm(b"audio", timeout=60)
                )
                await asyncio.sleep(0)
                task.cancel()
                await asyncio.wait_for(reap_started.wait(), timeout=0.5)

                try:
                    for _ in range(3):
                        task.cancel()
                        await asyncio.sleep(0)
                        self.assertFalse(task.done())
                finally:
                    release_reap.set()

                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=0.5)

            self.assertTrue(proc.killed)
            self.assertTrue(proc.waited)
            self.assertTrue(proc.reaped)
            self.assertFalse(proc.wait_cancelled)

        asyncio.run(scenario())

    def test_stderr_is_fully_drained_but_only_bounded_prefix_is_retained(self) -> None:
        audio_decoder = load_audio_decoder_module()
        stream = ChunkStream([b"a" * 8, b"b" * 8, b"c" * 8])

        retained = asyncio.run(audio_decoder._drain_stderr(stream, retain_bytes=10))

        self.assertEqual(retained, b"a" * 8 + b"b" * 2)
        self.assertEqual(stream.reads, 4)

    def test_stdout_limit_interrupts_a_blocked_stdin_writer(self) -> None:
        audio_decoder = load_audio_decoder_module()
        proc = StreamingProcess()

        async def scenario() -> None:
            with self.assertRaises(audio_decoder.AudioTooLongError):
                await asyncio.wait_for(
                    audio_decoder._communicate_bounded(
                        proc,
                        b"input",
                        output_limit=10,
                    ),
                    timeout=0.1,
                )

        asyncio.run(scenario())
        self.assertTrue(proc.stdin.drain_started)
        self.assertTrue(proc.stdin.closed)

    def test_stdout_limit_does_not_wait_for_cancelled_stdin_close(self) -> None:
        audio_decoder = load_audio_decoder_module()
        writer = BlockingCloseWriter()
        proc = StreamingProcess(writer)

        async def scenario() -> None:
            with self.assertRaises(audio_decoder.AudioTooLongError):
                await asyncio.wait_for(
                    audio_decoder._communicate_bounded(
                        proc,
                        b"input",
                        output_limit=10,
                    ),
                    timeout=0.1,
                )

        asyncio.run(scenario())
        self.assertTrue(writer.drain_started)
        self.assertTrue(writer.closed)
        self.assertFalse(writer.wait_closed_started)

    def test_windows_aborted_pipe_close_is_a_normal_input_shutdown(self) -> None:
        audio_decoder = load_audio_decoder_module()
        writer = AbortedCloseWriter()

        asyncio.run(audio_decoder._write_process_input(writer, b"input"))

        self.assertTrue(writer.drain_started)
        self.assertTrue(writer.closed)


if __name__ == "__main__":
    unittest.main()
