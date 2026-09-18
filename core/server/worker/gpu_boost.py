"""
GPU 加速管理模块

封装 GPU 显存频率锁定/解锁逻辑，用于减少冷启动延迟。
"""

import math
import os
import signal
import subprocess
import time
import ctypes
from config_server import ServerConfig as Config
from . import logger


GPU_BOOST_TIMEOUT_ENV = "CAPSWRITER_GPU_BOOST_TIMEOUT"
DEFAULT_GPU_BOOST_TIMEOUT_SECONDS = 5.0
GPU_COMMAND_CLEANUP_TIMEOUT_SECONDS = 2.0


def _gpu_boost_timeout_seconds() -> float:
    raw_timeout = os.environ.get(GPU_BOOST_TIMEOUT_ENV)
    if raw_timeout is None or raw_timeout == "":
        return DEFAULT_GPU_BOOST_TIMEOUT_SECONDS
    try:
        timeout = float(raw_timeout)
    except ValueError as exc:
        raise ValueError(f"{GPU_BOOST_TIMEOUT_ENV} must be a number") from exc
    if not math.isfinite(timeout):
        raise ValueError(f"{GPU_BOOST_TIMEOUT_ENV} must be a finite number")
    if timeout <= 0:
        raise ValueError(f"{GPU_BOOST_TIMEOUT_ENV} must be > 0")
    return timeout


def _gpu_command_popen_kwargs() -> dict:
    kwargs = {
        "shell": True,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "posix":
        kwargs["start_new_session"] = True
    elif os.name == "nt":
        kwargs["creationflags"] = getattr(
            subprocess,
            "CREATE_NEW_PROCESS_GROUP",
            0,
        )
    return kwargs


def _kill_gpu_command_tree(process) -> None:
    """Force-stop the timed-out shell and descendants, then reap the shell."""

    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as exc:
            logger.warning(f"GPU 命令 process group 清理失败: {exc}")
    elif os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=GPU_COMMAND_CLEANUP_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning(f"GPU 命令 process tree 清理失败: {exc}")

    if process.poll() is None:
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.wait(timeout=GPU_COMMAND_CLEANUP_TIMEOUT_SECONDS)
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(f"GPU 命令 shell 回收失败: {exc}")


def _run_gpu_command(command: str) -> bool:
    try:
        timeout = _gpu_boost_timeout_seconds()
    except ValueError as exc:
        logger.warning(f"GPU 命令 timeout 配置无效: {exc}")
        return False

    try:
        process = subprocess.Popen(command, **_gpu_command_popen_kwargs())
    except (OSError, ValueError) as exc:
        logger.warning(f"GPU 命令启动失败: {exc}")
        return False

    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_gpu_command_tree(process)
        logger.warning(f"GPU 命令超时 ({timeout:g}s): {command}")
        return False
    if returncode != 0:
        logger.warning(f"GPU 命令失败 (exit {returncode}): {command}")
        return False
    return True


class GpuBoostManager:
    """
    GPU 加速管理器。

    负责检测管理员权限、执行加速/取消加速命令、检查闲置超时。
    """

    def __init__(self, state):
        self.state = state

    # ── 公开方法 ──────────────────────────────────

    def handle_command(self, task):
        """处理 GPU 加速命令任务。"""
        if task.command != 'gpu_boost':
            return
        if not self._check_admin():
            logger.warning("非管理员权限，无法执行 GPU 加速命令")
            return
        if self.state.gpu_boosted:
            self.state.gpu_last_active = 0
            return

        logger.info(f"GPU 加速命令: {Config.gpu_boost_cmd}")
        if not _run_gpu_command(Config.gpu_boost_cmd):
            return
        self.state.gpu_boosted = True
        self.state.gpu_last_active = 0  # 0 表示已加速但尚未有实际音频任务使用过

    def check_idle(self):
        """GPU 闲置超时检查，超时则取消加速。"""
        if not Config.gpu_boost_enabled or not self.state.gpu_boosted:
            return
        # gpu_last_active = 0 表示刚加速但尚未被实际音频任务使用，不取消
        if self.state.gpu_last_active <= 0:
            return

        idle_time = time.time() - self.state.gpu_last_active
        if idle_time <= Config.gpu_unboost_timeout:
            return

        if not self._check_admin():
            logger.warning("非管理员权限，无法执行 GPU 取消加速命令")
            return

        logger.info(f"GPU 闲置 {idle_time:.0f}s，取消加速: {Config.gpu_unboost_cmd}")
        if not _run_gpu_command(Config.gpu_unboost_cmd):
            return
        self.state.gpu_boosted = False
        self.state.gpu_last_active = 0.0

    # ── 内部方法 ──────────────────────────────────

    @staticmethod
    def _check_admin() -> bool:
        """检测是否以管理员权限运行。"""
        try:
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            return False
