# coding: utf-8
"""
识别子进程管理器 (ProcessManager)

负责维护单机识别进程的生命周期，包括启动、模型加载监控、异常退出捕获。
"""
from __future__ import annotations
import sys
import os
import math
import queue
import threading
import time
from multiprocessing import Process, Manager
from typing import TYPE_CHECKING
from ..state import console
from ..queue_limits import QUEUE_PUT_RETRY_SECONDS
from . import start_worker
from .check_model import check_model
from . import logger
if TYPE_CHECKING:
    from ..app import CapsWriterServer


SERVER_WORKER_STOP_TIMEOUT_ENV = "CAPSWRITER_SERVER_WORKER_STOP_TIMEOUT"
DEFAULT_SERVER_WORKER_STOP_TIMEOUT_SECONDS = 2.0
SERVER_WORKER_KILL_GRACE_SECONDS = 2.0
SERVER_WORKER_STALL_TIMEOUT_ENV = "CAPSWRITER_SERVER_WORKER_STALL_TIMEOUT"
DEFAULT_SERVER_WORKER_STALL_TIMEOUT_SECONDS = 900.0
SERVER_MODEL_LOAD_TIMEOUT_ENV = "CAPSWRITER_SERVER_MODEL_LOAD_TIMEOUT"
DEFAULT_SERVER_MODEL_LOAD_TIMEOUT_SECONDS = 600.0
SERVER_WORKER_WATCHDOG_HTTP_GRACE_SECONDS = 2.0
SERVER_WORKER_WATCHDOG_POLL_SECONDS = 0.1


def _worker_stop_timeout_seconds() -> float:
    raw_timeout = os.environ.get(SERVER_WORKER_STOP_TIMEOUT_ENV)
    if raw_timeout is None or raw_timeout == "":
        return DEFAULT_SERVER_WORKER_STOP_TIMEOUT_SECONDS
    try:
        timeout = float(raw_timeout)
    except ValueError as exc:
        raise ValueError(f"{SERVER_WORKER_STOP_TIMEOUT_ENV} must be a number") from exc
    if not math.isfinite(timeout):
        raise ValueError(f"{SERVER_WORKER_STOP_TIMEOUT_ENV} must be a finite number")
    if timeout <= 0:
        raise ValueError(f"{SERVER_WORKER_STOP_TIMEOUT_ENV} must be > 0")
    return timeout


def _worker_stall_timeout_seconds() -> float:
    """Maximum time any one synchronous inference call may hold the worker."""
    raw_timeout = os.environ.get(SERVER_WORKER_STALL_TIMEOUT_ENV)
    if raw_timeout is None or raw_timeout == "":
        return DEFAULT_SERVER_WORKER_STALL_TIMEOUT_SECONDS
    try:
        timeout = float(raw_timeout)
    except ValueError as exc:
        raise ValueError(f"{SERVER_WORKER_STALL_TIMEOUT_ENV} must be a number") from exc
    if not math.isfinite(timeout):
        raise ValueError(f"{SERVER_WORKER_STALL_TIMEOUT_ENV} must be a finite number")
    if timeout <= 0:
        raise ValueError(f"{SERVER_WORKER_STALL_TIMEOUT_ENV} must be > 0")
    return timeout


def _model_load_timeout_seconds() -> float:
    """Maximum startup time allowed for the child to report model readiness."""
    raw_timeout = os.environ.get(SERVER_MODEL_LOAD_TIMEOUT_ENV)
    if raw_timeout is None or raw_timeout == "":
        return DEFAULT_SERVER_MODEL_LOAD_TIMEOUT_SECONDS
    try:
        timeout = float(raw_timeout)
    except ValueError as exc:
        raise ValueError(f"{SERVER_MODEL_LOAD_TIMEOUT_ENV} must be a number") from exc
    if not math.isfinite(timeout):
        raise ValueError(f"{SERVER_MODEL_LOAD_TIMEOUT_ENV} must be a finite number")
    if timeout <= 0:
        raise ValueError(f"{SERVER_MODEL_LOAD_TIMEOUT_ENV} must be > 0")
    return timeout


class ProcessManager:
    """
    识别子进程管理器
    
    由 CapsWriterServer 调用，专注于进程层级的控制。
    """
    def __init__(self, app: CapsWriterServer):
        self._process = None
        self.app = app
        self.is_alive = False
        self._watchdog_stop = threading.Event()
        self._watchdog_thread = None
        self._fail_stop_lock = threading.Lock()
        self._fail_stop_requested = False

    def start(self):
        """
        启动识别子进程并等待模型加载完成
        
        Returns:
            Process: 启动成功的子进程对象
        """
        # 防连续触发
        if self.is_alive: return
        # Parse explicit startup bounds before creating a Manager or child.
        # Invalid configuration must never leave a spawned recognizer behind.
        model_load_timeout = _model_load_timeout_seconds()
        self.is_alive = True
        self._watchdog_stop.clear()
        self._fail_stop_requested = False

        # 1. 前置检查
        check_model()

        # 2. 初始化共享资源
        # 使用 Manager 管理共享列表，用于追踪活动连接
        state = self.app.state
        state.sockets_id = Manager().list()
        state.recognizer_watchdog_failed = False
        self._reset_active_inference()
        
        # 获取标准输入文件描述符，用于 Windows 下的信号传递补丁
        stdin_fn = sys.stdin.fileno()
        
        # 3. 创建并启动进程
        self._process = Process(
            target=start_worker,
            args=(state.queue_in,
                  state.queue_out,
                  state.sockets_id, 
                  stdin_fn,
                  state.recognizer_active_inference),
            daemon=True
        )
        model_load_deadline = time.monotonic() + model_load_timeout
        self._process.start()
        
        # 存入状态以便其他模块引用
        state.recognize_process = self._process
        logger.info(f"识别子进程已拉起 (PID: {self._process.pid})")

        # 4. 等待模型加载完成 (轮询方式)
        self._wait_for_models(
            deadline=model_load_deadline,
            timeout_seconds=model_load_timeout,
        )
        if self.is_alive and self._process and self._process.is_alive():
            self._start_watchdog()
        
        return self._process

    def _wait_for_models(self, *, deadline: float, timeout_seconds: float):
        """轮询队列直到收到模型加载成功 (True) 或发生错误"""
        logger.info("正在等待子进程加载模型...")
        
        while self.is_alive:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._handle_model_load_timeout(timeout_seconds)
            try:
                # Never block beyond the single startup deadline.
                status = self.app.state.queue_out.get(timeout=min(0.1, remaining))
                if status is True:
                    if time.monotonic() >= deadline:
                        self._handle_model_load_timeout(timeout_seconds)
                    # 收到 True 说明模型加载成功
                    break
            except (queue.Empty, OSError):
                if self._process and not self._process.is_alive():
                    self._handle_unexpected_exit()
                    return
                continue
            
        if not self.is_alive: return
        logger.info("模型加载完成，ASR 服务就绪")
        console.rule('[green3]开始服务')
        console.line()

    def _handle_model_load_timeout(self, timeout_seconds: float) -> None:
        """Fail startup only after terminating and reaping the hung live child."""
        self.is_alive = False
        self._watchdog_stop.set()
        self.app.state.recognizer_watchdog_failed = True
        logger.error(
            f"识别子进程模型加载超过 {timeout_seconds:g}s；"
            "正在回收子进程并中止启动"
        )
        self._force_stop_process()
        raise TimeoutError(
            f"recognizer model load exceeded {timeout_seconds:g} seconds"
        )

    def _handle_unexpected_exit(self):
        """处理子进程加载模型时的意外退出"""
        exit_code = self._process.exitcode
        if exit_code != 0:
            logger.error(f"识别子进程意外退出! ExitCode: {exit_code}")
            logger.error("这通常是由于模型损坏、底层库冲突或系统资源不足导致的。")
        
        # 请求主系统同步退出
        self.app.stop()

    def _start_watchdog(self) -> None:
        """Monitor child liveness and parent-visible synchronous inference leases."""
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            return
        try:
            stall_timeout = _worker_stall_timeout_seconds()
        except ValueError as exc:
            logger.error(f"识别子进程 watchdog timeout 配置无效: {exc}")
            stall_timeout = DEFAULT_SERVER_WORKER_STALL_TIMEOUT_SECONDS
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            args=(stall_timeout,),
            name="capswriter-recognizer-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

    def _watchdog_loop(self, stall_timeout: float) -> None:
        while not self._watchdog_stop.wait(SERVER_WORKER_WATCHDOG_POLL_SECONDS):
            if not self.is_alive:
                return
            process = self._process
            try:
                process_alive = bool(process and process.is_alive())
            except (AssertionError, OSError, ValueError):
                process_alive = False
            if not process_alive:
                self._fail_stop("识别子进程在服务期间意外退出")
                return

            started, http_deadline = self._active_inference_snapshot()
            if started <= 0:
                continue
            now = time.monotonic()
            hard_deadline = started + stall_timeout
            if http_deadline > 0:
                hard_deadline = min(
                    hard_deadline,
                    http_deadline + SERVER_WORKER_WATCHDOG_HTTP_GRACE_SECONDS,
                )
            if now < hard_deadline:
                continue

            # Confirm the same call is still active after crossing the lease.
            current_started, current_http_deadline = self._active_inference_snapshot()
            if (
                current_started != started
                or current_http_deadline != http_deadline
                or current_started <= 0
            ):
                continue
            self._fail_stop("识别子进程同步推理超过 watchdog 上限")
            return

    def _active_inference_snapshot(self) -> tuple[float, float]:
        active = getattr(self.app.state, "recognizer_active_inference", None)
        if active is None:
            return 0.0, 0.0
        lock = active.get_lock()
        acquired = False
        try:
            acquired = lock.acquire(timeout=SERVER_WORKER_WATCHDOG_POLL_SECONDS)
            if not acquired:
                return 0.0, 0.0
            return float(active[0]), float(active[1])
        except (OSError, TypeError, ValueError):
            logger.error("无法读取识别子进程 watchdog 状态")
            return 0.0, 0.0
        finally:
            if acquired:
                lock.release()

    def _reset_active_inference(self) -> None:
        active = getattr(self.app.state, "recognizer_active_inference", None)
        if active is None:
            return
        with active.get_lock():
            active[0] = 0.0
            active[1] = 0.0

    def _fail_stop(self, reason: str) -> None:
        """Reap the unsafe child, then stop the whole server on its loop thread."""
        with self._fail_stop_lock:
            if self._fail_stop_requested:
                return
            self._fail_stop_requested = True
        self.app.state.recognizer_watchdog_failed = True
        logger.error(f"{reason}；为避免继续接受无法完成的任务，服务器将停止")
        self._watchdog_stop.set()
        try:
            self._force_stop_process()
        finally:
            self._schedule_app_stop()

    def _force_stop_process(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            alive = process.is_alive()
        except (AssertionError, OSError, ValueError):
            alive = False
        if alive:
            try:
                process.terminate()
                process.join(timeout=SERVER_WORKER_KILL_GRACE_SECONDS)
            except (AssertionError, OSError, ValueError):
                logger.error("watchdog terminate 识别子进程失败")
        try:
            alive = process.is_alive()
        except (AssertionError, OSError, ValueError):
            alive = False
        if alive:
            kill = getattr(process, "kill", None)
            if kill is not None:
                try:
                    kill()
                    process.join(timeout=SERVER_WORKER_KILL_GRACE_SECONDS)
                except (AssertionError, OSError, ValueError):
                    logger.error("watchdog kill 识别子进程失败")
        try:
            alive = process.is_alive()
        except (AssertionError, OSError, ValueError):
            alive = False
        if alive:
            logger.error(f"watchdog 无法回收识别子进程 (PID: {process.pid})")

    def _schedule_app_stop(self) -> None:
        loop = getattr(self.app, "loop", None)
        if loop is not None:
            try:
                loop.call_soon_threadsafe(self.app.stop)
                return
            except RuntimeError:
                pass
        self.app.stop()

    def _stop_watchdog(self) -> None:
        stop_event = getattr(self, "_watchdog_stop", None)
        if stop_event is not None:
            stop_event.set()
        thread = getattr(self, "_watchdog_thread", None)
        if (
            thread is not None
            and thread is not threading.current_thread()
            and thread.is_alive()
        ):
            thread.join(timeout=1.0)

    def stop(self):
        """停止子进程"""

        self._stop_watchdog()

        # 防连续触发
        if not self.is_alive: return
        self.is_alive = False

        if self._process and self._process.is_alive():
            logger.info(f"正在终止识别子进程 (PID: {self._process.pid})...")
            # 发送 None 任务通知优雅退出 (作为兜底)
            try:
                timeout = _worker_stop_timeout_seconds()
            except ValueError as e:
                logger.error(f"识别子进程停止 timeout 配置无效: {e}")
                timeout = DEFAULT_SERVER_WORKER_STOP_TIMEOUT_SECONDS

            try:
                self.app.state.queue_in.put(
                    None,
                    timeout=QUEUE_PUT_RETRY_SECONDS,
                )
            except Exception as e:
                logger.warning(f"发送识别子进程停止信号失败: {e}")

            # 如果配置时间内没退，则逐级强制结束并等待回收。
            self._process.join(timeout=timeout)
            if self._process.is_alive():
                logger.debug("子进程未响应优雅退出，执行强制终止")
                self._process.terminate()
                self._process.join(timeout=SERVER_WORKER_KILL_GRACE_SECONDS)

            if self._process.is_alive():
                kill = getattr(self._process, "kill", None)
                if kill is not None:
                    logger.error("子进程 terminate 后仍未退出，执行 kill")
                    kill()
                    self._process.join(timeout=SERVER_WORKER_KILL_GRACE_SECONDS)

            if self._process.is_alive():
                logger.error(f"识别子进程仍未退出 (PID: {self._process.pid})")
