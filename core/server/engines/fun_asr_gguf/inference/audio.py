import numpy as np
import math
import os
import shutil
import subprocess

ENGINE_FFMPEG_TIMEOUT_ENV = "CAPSWRITER_ENGINE_FFMPEG_TIMEOUT"
DEFAULT_ENGINE_FFMPEG_TIMEOUT_SECONDS = 120.0
FFMPEG_KILL_GRACE_SECONDS = 2.0
MAX_FFMPEG_ERROR_CHARS = 1000


def _ffmpeg_timeout_seconds():
    raw_timeout = os.environ.get(ENGINE_FFMPEG_TIMEOUT_ENV)
    if raw_timeout is None or raw_timeout == "":
        return DEFAULT_ENGINE_FFMPEG_TIMEOUT_SECONDS
    try:
        timeout = float(raw_timeout)
    except ValueError as exc:
        raise RuntimeError(f"{ENGINE_FFMPEG_TIMEOUT_ENV} must be a number") from exc
    if not math.isfinite(timeout):
        raise RuntimeError(f"{ENGINE_FFMPEG_TIMEOUT_ENV} must be a finite number")
    if timeout <= 0:
        raise RuntimeError(f"{ENGINE_FFMPEG_TIMEOUT_ENV} must be > 0")
    return timeout


def _stderr_preview(stderr):
    text = stderr.decode('utf-8', errors='ignore').strip()
    preview = " ".join(text.split())
    if len(preview) > MAX_FFMPEG_ERROR_CHARS:
        return f"{preview[:MAX_FFMPEG_ERROR_CHARS].rstrip()}..."
    return preview


def check_ffmpeg():
    return shutil.which('ffmpeg') is not None


def load_audio(audio_path, sample_rate=16000, use_normalizer=True, start_second=None, duration=None):
    """加载音频文件并转换为 16kHz PCM，支持按需加载指定片段"""
    if not check_ffmpeg():
        raise RuntimeError("系统未发现 ffmpeg。请先安装 ffmpeg 并将其添加到系统环境变量 PATH 中。")

    cmd = ['ffmpeg', '-y', '-i', str(audio_path)]

    if start_second is not None:
        cmd.extend(['-ss', str(start_second)])
    if duration is not None and duration > 0:
        cmd.extend(['-t', str(duration)])

    cmd.extend([
        '-ar', str(sample_rate),
        '-ac', '1',
        '-f', 'f32le',
        'pipe:1'
    ])

    timeout = _ffmpeg_timeout_seconds()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0
    )

    try:
        raw_bytes, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        try:
            _, stderr = process.communicate(timeout=FFMPEG_KILL_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"ffmpeg 处理音频超时: {timeout:g}s; 进程未在 kill 后退出"
            ) from exc
        detail = _stderr_preview(stderr)
        raise RuntimeError(f"ffmpeg 处理音频超时: {timeout:g}s: {detail}") from exc

    if process.returncode != 0:
        error_msg = _stderr_preview(stderr)
        raise RuntimeError(f"ffmpeg 处理音频失败: {error_msg}")

    return np.frombuffer(raw_bytes, dtype=np.float32)
