# coding: utf-8
"""
HTTP API 专用的音频解码器

透过 FFmpeg subprocess 将任意格式的音频字节流转为 CapsWriter
识别管线接受的 PCM 格式 (16 kHz / float32 / mono)。

设计与 util/client/transcribe/media_tool.py 中既有的 FFmpeg 调用一致,
保持单一格式约定与依赖。
"""

import asyncio
import shutil

from util.constants import AudioFormat

from . import logger


class AudioDecodeError(Exception):
    """FFmpeg 解码失败时抛出。HTTP 层会转为 400 响应。"""


class FFmpegNotFoundError(AudioDecodeError):
    """ffmpeg 不在 PATH 中。HTTP 层应视为 500 (服务器配置问题), 非 400。"""


async def decode_to_pcm(audio_bytes: bytes, timeout: float = 120.0) -> bytes:
    """
    将任意格式的音频文件字节流转为 16 kHz / float32 / mono PCM。

    Args:
        audio_bytes: 上传文件的原始字节 (mp3/wav/m4a/flac/ogg/webm/...)。
        timeout:     FFmpeg 子进程最大执行时间 (秒)。

    Returns:
        raw float32 PCM bytes, 可直接送入 Task.data。

    Raises:
        AudioDecodeError: ffmpeg 不存在 / 解码失败 / 超时 / 输出为空。
    """
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
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise AudioDecodeError(f"ffmpeg timeout after {timeout:.0f}s")

    if proc.returncode != 0:
        err = stderr.decode("utf-8", errors="replace").strip()
        logger.warning(f"ffmpeg 解码失败 (exit={proc.returncode}): {err}")
        raise AudioDecodeError(f"ffmpeg failed: {err or 'unknown error'}")

    if not stdout:
        raise AudioDecodeError("ffmpeg produced empty PCM output")

    return stdout
