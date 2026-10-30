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
GPU_COMMAND_RETRY_COOLDOWN_SECONDS = 30.0


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
    if not 0.1 <= timeout <= 60:
        raise ValueError(f"{GPU_BOOST_TIMEOUT_ENV} must be >= 0.1 and <= 60")
    return timeout


def _gpu_command_popen_kwargs(command: str | list[str]) -> dict:
    kwargs = {
        "shell": isinstance(command, str),
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


def _run_gpu_command(command: str | list[str]) -> bool:
    try:
        timeout = _gpu_boost_timeout_seconds()
    except ValueError as exc:
        logger.warning(f"GPU 命令 timeout 配置无效: {exc}")
        return False

    try:
        process = subprocess.Popen(command, **_gpu_command_popen_kwargs(command))
    except (OSError, ValueError) as exc:
        logger.warning(f"GPU 命令启动失败: {exc}")
        return False

    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_gpu_command_tree(process)
        logger.warning(f"GPU 命令超时 ({timeout:g}s): {command}")
        return False
    except OSError as exc:
        _kill_gpu_command_tree(process)
        logger.warning(f"GPU 命令等待失败: {exc}")
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
        self._boost_retry_after = 0.0
        self._unboost_retry_after = 0.0
        # A command can apply clocks before failing or timing out. Keep that
        # uncertainty separate from the confirmed-success gpu_boosted flag.
        self._cleanup_needed = False
        self._cleanup_last_active = 0.0

    # ── 公开方法 ──────────────────────────────────

    def handle_command(self, task):
        """Warm up on an explicit mic command or before an audio task."""
        if not Config.gpu_boost_enabled:
            return
        if (getattr(task, "command", None) != 'gpu_boost'
                and getattr(task, "type", "cmd") == 'cmd'):
            return
        self.ensure_boosted()

    def ensure_boosted(self):
        """Best-effort boost before inference, with bounded failure retries."""
        if (not Config.gpu_boost_enabled
                or not getattr(Config, "gpu_boost_backend_active", True)):
            return
        if self.state.gpu_boosted:
            self.record_activity()
            return
        if time.monotonic() < self._boost_retry_after:
            return

        if not self._check_admin():
            logger.warning("非管理员权限，无法执行 GPU 加速命令")
            self._boost_retry_after = time.monotonic() + GPU_COMMAND_RETRY_COOLDOWN_SECONDS
            return
        logger.info(f"GPU 加速命令: {Config.gpu_boost_cmd}")
        self._cleanup_needed = True
        if not _run_gpu_command(Config.gpu_boost_cmd):
            self._cleanup_last_active = time.monotonic()
            self._boost_retry_after = time.monotonic() + GPU_COMMAND_RETRY_COOLDOWN_SECONDS
            return
        self.state.gpu_boosted = True
        # Even an abandoned mic warmup must eventually expire.
        self.record_activity()
        self._boost_retry_after = 0.0

    def record_activity(self):
        """Record monotonic completion time for any boost we still own."""
        if self.state.gpu_boosted or self._cleanup_needed:
            self.state.gpu_last_active = time.monotonic()
            self._cleanup_last_active = self.state.gpu_last_active

    def check_idle(self):
        """GPU 闲置超时检查，超时则取消加速。"""
        # Disabling future boosts does not release clocks already owned here.
        if not (self.state.gpu_boosted or self._cleanup_needed):
            return
        idle_time = time.monotonic() - max(self.state.gpu_last_active, self._cleanup_last_active)
        if idle_time <= Config.gpu_unboost_timeout:
            return
        if time.monotonic() < self._unboost_retry_after:
            return

        self._unboost()

    def shutdown(self):
        """Attempt one reset on normal shutdown, including after an idle failure.

        The parent must allow the bounded command and process-tree cleanup to
        complete before escalating to terminate/kill.
        """
        if self.state.gpu_boosted or self._cleanup_needed:
            self._unboost()

    def _unboost(self):
        if not self._check_admin():
            logger.warning("非管理员权限，无法执行 GPU 取消加速命令")
            self._unboost_retry_after = time.monotonic() + GPU_COMMAND_RETRY_COOLDOWN_SECONDS
            return

        logger.info(f"GPU 取消加速: {Config.gpu_unboost_cmd}")
        if not _run_gpu_command(Config.gpu_unboost_cmd):
            self._unboost_retry_after = time.monotonic() + GPU_COMMAND_RETRY_COOLDOWN_SECONDS
            return
        self.state.gpu_boosted = False
        self.state.gpu_last_active = 0.0
        self._cleanup_needed = False
        self._cleanup_last_active = 0.0
        self._unboost_retry_after = 0.0

    # ── 内部方法 ──────────────────────────────────

    @staticmethod
    def _check_admin() -> bool:
        """Preserve the Windows guard; other systems rely on command status."""
        if os.name != "nt":
            return True
        try:
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            return False
