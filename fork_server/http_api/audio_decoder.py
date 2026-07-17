# coding: utf-8
"""
HTTP API 專用的音訊解碼器

透過 FFmpeg subprocess 把任意格式的音訊位元流轉為 CapsWriter
識別管線接受的 PCM 格式 (16 kHz / float32 / mono)。
"""

from __future__ import annotations
import asyncio
from contextlib import suppress
import math
import shutil

from core.constants import AudioFormat
from core.server import logger


MAX_FFMPEG_ERROR_CHARS = 1000
MAX_FFMPEG_ERROR_BYTES = 4096
FFMPEG_KILL_GRACE_SECONDS = 2.0
PROCESS_IO_CHUNK_BYTES = 64 * 1024
PROCESS_INPUT_CHUNK_BYTES = 1024 * 1024


class AudioDecodeError(Exception):
    """FFmpeg 解碼失敗。HTTP 層轉為 400。"""


class FFmpegNotFoundError(AudioDecodeError):
    """ffmpeg 不在 PATH 中。HTTP 層應視為 500 (server config 問題)。"""


class AudioDecodeTimeoutError(AudioDecodeError):
    """The decoder exceeded the request deadline."""


class AudioTooLongError(AudioDecodeError):
    """Decoded PCM exceeded the configured duration limit."""


def _stderr_preview(stderr: bytes) -> str:
    text = stderr.decode("utf-8", errors="replace")
    # Collapse line breaks and remove terminal/control characters before this
    # bounded preview can reach the opt-in diagnostic log path.
    printable = "".join(char if char.isprintable() else " " for char in text)
    preview = " ".join(printable.split())
    if not preview:
        return "unknown error"
    if len(preview) > MAX_FFMPEG_ERROR_CHARS:
        return f"{preview[:MAX_FFMPEG_ERROR_CHARS].rstrip()}..."
    return preview


def _positive_timeout_seconds(value: float) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise AudioDecodeError("ffmpeg timeout must be a positive finite number") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise AudioDecodeError("ffmpeg timeout must be a positive finite number")
    return timeout


async def _kill_timed_out_process(proc) -> bool:
    try:
        proc.kill()
    except ProcessLookupError:
        return True
    except OSError:
        # Windows can deny kill() while the process handle is already closing.
        # Still perform one bounded reap attempt and preserve the caller's
        # timeout/cancellation semantics instead of replacing them with OSError.
        pass
    try:
        await asyncio.wait_for(proc.wait(), timeout=FFMPEG_KILL_GRACE_SECONDS)
    except (asyncio.TimeoutError, ChildProcessError, OSError):
        return False
    return True


async def _await_cleanup_task(cleanup_task: asyncio.Task):
    """Join an independent cleanup task despite repeated caller cancellation."""

    cancellation = None
    while True:
        try:
            result = await asyncio.shield(cleanup_task)
            break
        except asyncio.CancelledError as exc:
            cancellation = cancellation or exc
            if cleanup_task.cancelled():
                raise
            if cleanup_task.done():
                result = cleanup_task.result()
                break

    if cancellation is not None:
        raise cancellation
    return result


async def _kill_and_reap(proc) -> bool:
    """Finish child cleanup even when an outer deadline cancels this task."""
    cleanup_task = asyncio.create_task(_kill_timed_out_process(proc))
    return await _await_cleanup_task(cleanup_task)


async def _write_process_input(writer, data: bytes) -> None:
    if writer is None:
        return
    cancelled = False
    try:
        view = memoryview(data)
        for offset in range(0, len(view), PROCESS_INPUT_CHUNK_BYTES):
            writer.write(view[offset : offset + PROCESS_INPUT_CHUNK_BYTES])
            await writer.drain()
    except asyncio.CancelledError:
        # Output overflow and request cancellation both cancel this producer.
        # Closing the pipe is enough here: waiting for close can depend on the
        # child consuming buffered stdin while the child is itself blocked on
        # stdout, preventing the caller from reaching kill/reap cleanup.
        cancelled = True
        raise
    except ConnectionError:
        pass
    finally:
        writer.close()
        wait_closed = getattr(writer, "wait_closed", None)
        if wait_closed is not None and not cancelled:
            with suppress(ConnectionError):
                await wait_closed()


async def _read_stdout(stream, output_limit: int | None) -> bytes:
    data = bytearray()
    while True:
        chunk = await stream.read(PROCESS_IO_CHUNK_BYTES)
        if not chunk:
            break
        if output_limit is not None and len(data) + len(chunk) > output_limit:
            raise AudioTooLongError(
                "decoded audio exceeds the configured duration limit"
            )
        data.extend(chunk)
    return bytes(data)


async def _drain_stderr(stream, retain_bytes: int = MAX_FFMPEG_ERROR_BYTES) -> bytes:
    """Drain all stderr to avoid deadlock while retaining only a bounded prefix."""
    retained = bytearray()
    while True:
        chunk = await stream.read(PROCESS_IO_CHUNK_BYTES)
        if not chunk:
            break
        remaining = max(0, retain_bytes - len(retained))
        if remaining:
            retained.extend(chunk[:remaining])
    return bytes(retained)


async def _communicate_bounded(proc, audio_bytes: bytes, output_limit: int | None):
    # Lightweight unit-test fakes from older tests expose communicate() only.
    # Real asyncio subprocesses always take the bounded streaming path below.
    if any(getattr(proc, name, None) is None for name in ("stdin", "stdout", "stderr")):
        return await proc.communicate(input=audio_bytes)

    input_task = asyncio.create_task(_write_process_input(proc.stdin, audio_bytes))
    stdout_task = asyncio.create_task(_read_stdout(proc.stdout, output_limit))
    stderr_task = asyncio.create_task(_drain_stderr(proc.stderr))
    tasks = (input_task, stdout_task, stderr_task)
    try:
        _ignored, stdout, stderr = await asyncio.gather(*tasks)
        await proc.wait()
        return stdout, stderr
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def decode_to_pcm(
    audio_bytes: bytes,
    timeout: float = 120.0,
    *,
    max_output_bytes: int | None = None,
) -> bytes:
    """
    把任意格式音訊轉為 16 kHz / float32 / mono PCM。

    Returns:
        raw float32 PCM bytes, 可直接送入 Task.data。

    Raises:
        FFmpegNotFoundError / AudioDecodeError
    """
    timeout_seconds = _positive_timeout_seconds(timeout)
    if shutil.which("ffmpeg") is None:
        raise FFmpegNotFoundError("ffmpeg not found in PATH")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", "pipe:0",
    ]
    if max_output_bytes is not None:
        try:
            output_limit = int(max_output_bytes)
        except (TypeError, ValueError) as exc:
            raise AudioDecodeError("PCM output limit must be a positive integer") from exc
        if output_limit <= 0:
            raise AudioDecodeError("PCM output limit must be a positive integer")
        # Ask ffmpeg for only a tiny amount beyond the limit.  This keeps stdout
        # memory bounded while still letting us distinguish an exact-limit file
        # from one that must be rejected rather than silently truncated.
        max_seconds = output_limit / AudioFormat.BYTES_PER_SECOND
        cmd.extend(["-t", f"{max_seconds + 0.1:.6f}"])
    else:
        output_limit = None
    cmd.extend([
        "-f", "f32le",
        "-ac", str(AudioFormat.CHANNELS),
        "-ar", str(AudioFormat.SAMPLE_RATE),
        "pipe:1",
    ])

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            _communicate_bounded(proc, audio_bytes, output_limit),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        killed = await _kill_and_reap(proc)
        detail = f"ffmpeg timeout after {timeout_seconds:g}s"
        if not killed:
            detail += "; process did not exit after kill"
        raise AudioDecodeTimeoutError(detail)
    except asyncio.CancelledError:
        # An outer end-to-end request deadline or client disconnect must not
        # leave ffmpeg running after the coroutine has gone away.
        await _kill_and_reap(proc)
        raise
    except AudioTooLongError:
        await _kill_and_reap(proc)
        raise
    except Exception:
        await _kill_and_reap(proc)
        raise

    if proc.returncode != 0:
        err = _stderr_preview(stderr)
        # FFmpeg diagnostics are derived from an untrusted upload and may
        # include embedded metadata or terminal control sequences.  Keep the
        # always-on decoder log privacy-safe; the bounded exception is handled
        # by the API layer, which emits details only when sensitive logging is
        # explicitly enabled.
        logger.warning(
            f"ffmpeg 解碼失敗 (exit={proc.returncode}); details=<redacted>"
        )
        raise AudioDecodeError(f"ffmpeg failed: {err}")

    if not stdout:
        raise AudioDecodeError("ffmpeg produced empty PCM output")

    # The streaming reader enforces this before retaining an oversized chunk;
    # keep the postcondition as defense-in-depth and for lightweight fakes.
    if output_limit is not None and len(stdout) > output_limit:
        raise AudioTooLongError("decoded audio exceeds the configured duration limit")

    return stdout
