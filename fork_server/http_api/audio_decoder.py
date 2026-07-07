# coding: utf-8
"""
HTTP API 專用的音訊解碼器

透過 FFmpeg subprocess 把任意格式的音訊位元流轉為 CapsWriter
識別管線接受的 PCM 格式 (16 kHz / float32 / mono)。
"""

from __future__ import annotations
import asyncio
import math
import shutil

from core.constants import AudioFormat
from core.server import logger


MAX_FFMPEG_ERROR_CHARS = 1000
FFMPEG_KILL_GRACE_SECONDS = 2.0


class AudioDecodeError(Exception):
    """FFmpeg 解碼失敗。HTTP 層轉為 400。"""


class FFmpegNotFoundError(AudioDecodeError):
    """ffmpeg 不在 PATH 中。HTTP 層應視為 500 (server config 問題)。"""


def _stderr_preview(stderr: bytes) -> str:
    text = stderr.decode("utf-8", errors="replace").strip()
    preview = " ".join(text.split())
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
    try:
        await asyncio.wait_for(proc.wait(), timeout=FFMPEG_KILL_GRACE_SECONDS)
    except asyncio.TimeoutError:
        return False
    return True


async def decode_to_pcm(audio_bytes: bytes, timeout: float = 120.0) -> bytes:
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
        "-f", "f32le",
        "-ac", str(AudioFormat.CHANNELS),
        "-ar", str(AudioFormat.SAMPLE_RATE),
        "pipe:1",
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=audio_bytes),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        killed = await _kill_timed_out_process(proc)
        detail = f"ffmpeg timeout after {timeout_seconds:g}s"
        if not killed:
            detail += "; process did not exit after kill"
        raise AudioDecodeError(detail)

    if proc.returncode != 0:
        err = _stderr_preview(stderr)
        logger.warning(f"ffmpeg 解碼失敗 (exit={proc.returncode}): {err}")
        raise AudioDecodeError(f"ffmpeg failed: {err}")

    if not stdout:
        raise AudioDecodeError("ffmpeg produced empty PCM output")

    return stdout
