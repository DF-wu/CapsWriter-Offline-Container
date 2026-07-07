# coding: utf-8
import asyncio
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from core.client.state import console
from . import logger

CLIENT_MEDIA_TIMEOUT_ENV = "CAPSWRITER_CLIENT_MEDIA_TIMEOUT"
DEFAULT_CLIENT_MEDIA_TIMEOUT_SECONDS = 120.0
CLIENT_MEDIA_KILL_GRACE_SECONDS = 2.0


class MediaTool:
    """媒体工具类：负责 FFmpeg 相关操作"""

    @staticmethod
    def timeout_seconds() -> float:
        raw_timeout = os.environ.get(CLIENT_MEDIA_TIMEOUT_ENV)
        if raw_timeout is None or raw_timeout == "":
            return DEFAULT_CLIENT_MEDIA_TIMEOUT_SECONDS
        try:
            timeout = float(raw_timeout)
        except ValueError as exc:
            raise ValueError(f"{CLIENT_MEDIA_TIMEOUT_ENV} must be a number") from exc
        if not math.isfinite(timeout):
            raise ValueError(f"{CLIENT_MEDIA_TIMEOUT_ENV} must be a finite number")
        if timeout <= 0:
            raise ValueError(f"{CLIENT_MEDIA_TIMEOUT_ENV} must be > 0")
        return timeout

    @staticmethod
    async def kill_process(process) -> bool:
        if process.returncode is not None:
            return True
        try:
            process.kill()
        except ProcessLookupError:
            return True
        try:
            await asyncio.wait_for(
                process.wait(),
                timeout=CLIENT_MEDIA_KILL_GRACE_SECONDS,
            )
        except asyncio.TimeoutError:
            return False
        return True

    @staticmethod
    def check_environment() -> bool:
        """检查 FFmpeg 和 ffprobe 环境"""
        ffmpeg_path = shutil.which('ffmpeg')
        ffprobe_path = shutil.which('ffprobe')
        
        if ffmpeg_path is None:
            console.print('\n[bold red]错误：未检测到 FFmpeg 环境[/bold red]')
            console.print('    文件转录功能依赖 FFmpeg 来提取音视频中的音频。')
            console.print('    [cyan]建议处理方案：[/cyan]')
            console.print('    1. 请确保已安装 FFmpeg 并将其 [bold]bin[/bold] 目录添加到系统环境变量 [bold]Path[/bold] 中。')
            console.print('    2. 或者将 [bold]ffmpeg.exe[/bold] 放置在程序根目录下。')
            console.print('    3. 也可以前往官方下载：[u]https://ffmpeg.org/download.html[/u]\n')
            logger.error("未检测到 FFmpeg 环境，无法进行文件转录")
            return False
            
        if ffprobe_path is None:
            console.print('\n[bold yellow]提示：未检测到 ffprobe 环境[/bold yellow]')
            console.print('    程序将无法预先获取文件时长，进度条将只显示当前已发送时长。')
            console.print('    [cyan]建议：[/cyan]若需完整进度条，请在安装 FFmpeg 时确保 bin 目录下包含 ffprobe.exe。\n')
            logger.warning("未检测到 ffprobe 环境，进度显示将受到限制")
            
        return True

    @staticmethod
    async def get_audio_duration(file: Path) -> float:
        """获取音视频文件时长"""
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(file)
        ]
        try:
            timeout = MediaTool.timeout_seconds()
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
            if process.returncode == 0:
                return float(stdout.decode().strip())
        except asyncio.TimeoutError:
            logger.warning("ffprobe 获取时长超时")
            if 'process' in locals():
                await MediaTool.kill_process(process)
        except Exception as e:
            logger.warning(f"无法通过 ffprobe 获取时长: {e}")
        return 0.0

    @staticmethod
    def build_ffmpeg_cmd(file: Path) -> List[str]:
        """构建提取音频的 FFmpeg 命令"""
        return [
            "ffmpeg", "-i", str(file),
            "-f", "f32le", "-ac", "1", "-ar", "16000", "-"
        ]
