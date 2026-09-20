# coding: utf-8
"""
识别任务处理器

负责监听任务队列、执行识别流水线并将结果返回主进程。

公平调度：从不同客户端（socket）轮转取任务处理，防止文件转录淹没队列。
同 socket 内保持 FIFO 顺序，跨 socket 间轮转调度。
"""

from collections import OrderedDict, deque
from multiprocessing import Queue
from multiprocessing.managers import ListProxy
import queue
import time
from .pipeline import TaskPipeline
from ..schema import Result
from ..state import WorkerState
from ..queue_limits import WORKER_BUFFER_MAX_TASKS
from ..task_control import (
    WEBSOCKET_TASK_LIMIT_COMMAND,
    WEBSOCKET_TASK_LIMIT_ERROR_CODE,
    WEBSOCKET_TASK_LIMIT_ERROR_MESSAGE,
)
from .gpu_boost import GpuBoostManager
from . import logger


MAX_QUEUE_DRAIN_PER_CYCLE = 32
IDLE_QUEUE_TIMEOUT_SECONDS = 1.0
WORKER_TASK_ERROR_CODE = "worker_processing_failed"
WORKER_TASK_ERROR_MESSAGE = "Recognition failed while processing the audio."


class TaskBuffer:
    """按 (socket_id, task_id) 分组缓冲，支持跨 session 轮转出队。"""
    def __init__(
        self,
        state: WorkerState,
        max_tasks: int = WORKER_BUFFER_MAX_TASKS,
    ):
        self.state = state
        self.max_tasks = max(1, int(max_tasks))
        self._size = 0
        self._buffers: OrderedDict[tuple[str, str], deque] = OrderedDict()

    def enqueue(self, task):
        """将任务放入 composite session 的缓冲尾部（同 session 内 FIFO）。"""
        if self._size >= self.max_tasks:
            raise queue.Full
        key = (task.socket_id, task.task_id)
        if key not in self._buffers:
            self._buffers[key] = deque()
            self.state.get_session(task.task_id, task.socket_id, task.type)
        self._buffers[key].append(task)
        self._size += 1

    def pop(self):
        """轮转取出最早等待 session 的下一个任务；空时返回 None。"""
        if not self._buffers:
            return None

        key, buf = next(iter(self._buffers.items()))
        task = buf.popleft()
        self._size -= 1

        if buf:
            self._buffers.move_to_end(key)
        else:
            del self._buffers[key]

        return task

    def cleanup_tasks(self):
        """清理已断开连接的 session 的缓冲任务。"""
        for key in list(self._buffers):
            if key not in self.state.sessions:
                logger.debug(f"清理断开连接的 session: {key[1][:8]}")
                self._size -= len(self._buffers[key])
                del self._buffers[key]

    def discard_session(self, socket_id: str, task_id: str) -> int:
        """Drop all queued segments for one composite stream identity."""
        key = (socket_id, task_id)
        tasks = self._buffers.pop(key, None)
        if tasks is None:
            return 0
        removed = len(tasks)
        self._size -= removed
        return removed

    @property
    def is_empty(self) -> bool:
        return self._size == 0

    @property
    def remaining_capacity(self) -> int:
        return self.max_tasks - self._size

    def __len__(self) -> int:
        return self._size


class TaskHandler:
    """
    任务处理器

    协调输入输出队列与识别引擎之间的任务流。
    支持跨 socket 公平轮转调度。
    """
    def __init__(
        self,
        queue_in: Queue,
        queue_out: Queue,
        sockets_id: ListProxy,
        state: WorkerState,
        *,
        active_inference=None,
    ):
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.sockets_id = sockets_id
        self.state = state
        self.active_inference = active_inference

        self.recognizer = None
        self.punc_model = None
        self.aligner = None
        self.pipeline = None

        self.buffer = TaskBuffer(state)
        self.gpu_boost = GpuBoostManager(state)

    def set_engine(self, recognizer, punc_model=None, aligner=None):
        """注入识别引擎实例并初始化管线"""
        self.recognizer = recognizer
        self.punc_model = punc_model
        self.aligner = aligner
        self.pipeline = TaskPipeline(recognizer, punc_model, aligner, self.state)

    def drain_queue(self) -> bool:
        """有界地搬运输入任务。Returns: False = 收到退出信号。"""
        drain_limit = min(
            MAX_QUEUE_DRAIN_PER_CYCLE,
            self.buffer.remaining_capacity,
        )
        for index in range(drain_limit):
            try:
                if index == 0 and self.buffer.is_empty:
                    task = self.queue_in.get(timeout=IDLE_QUEUE_TIMEOUT_SECONDS)
                else:
                    task = self.queue_in.get_nowait()
            except queue.Empty:
                if self.buffer.is_empty:
                    # A client may disconnect after receiving a non-final
                    # result and never submit another segment. Reap its
                    # session even when no later task arrives to trigger the
                    # normal post-dispatch cleanup path.
                    self.cleanup()
                    self.cleanup_engines()
                return True
            except InterruptedError:
                continue

            # 判断退出信号
            if task is None:
                return False

            # 跳过已断开连接客户端的任务
            if task.socket_id not in self.sockets_id:
                logger.debug(f"跳过断连客户端任务: {task.task_id[:8]}")
                continue
            if self.task_deadline_expired(task):
                logger.info(f"跳过已超过 HTTP deadline 的任务: {task.task_id[:8]}")
                self.state.sessions.pop((task.socket_id, task.task_id), None)
                continue

            # 任务进入缓冲区
            self.buffer.enqueue(task)

        return True

    def cleanup(self):
        """清理断连 socket 的缓冲任务和 session。"""
        self.state.cleanup_sessions(self.sockets_id)
        self.buffer.cleanup_tasks()

    def cleanup_engines(self):
        """闲置资源清理：对齐器卸载 + GPU 加速取消。"""
        if self.pipeline and self.pipeline.aligner:
            self.pipeline.aligner.check_idle()
        self.gpu_boost.check_idle()

    def handle_command_task(self, task):
        """处理命令任务。"""
        if task.command == WEBSOCKET_TASK_LIMIT_COMMAND:
            self.state.sessions.pop((task.socket_id, task.task_id), None)
            self.buffer.discard_session(task.socket_id, task.task_id)
            self.queue_out.put(
                Result(
                    task_id=task.task_id,
                    socket_id=task.socket_id,
                    type="cmd",
                    time_start=task.time_start,
                    time_submit=task.time_submit,
                    time_complete=time.time(),
                    is_final=True,
                    error_code=WEBSOCKET_TASK_LIMIT_ERROR_CODE,
                    error_message=WEBSOCKET_TASK_LIMIT_ERROR_MESSAGE,
                )
            )
            return
        self.gpu_boost.handle_command(task)

    @staticmethod
    def task_deadline_expired(task) -> bool:
        """Return whether an optional cross-process deadline has passed."""
        deadline = getattr(task, "deadline_monotonic", None)
        return deadline is not None and time.monotonic() >= deadline

    def dispatch_task(self, task) -> None:
        """Recheck liveness and expiry immediately before synchronous work."""
        if task.socket_id not in self.sockets_id:
            logger.debug(f"跳过断连客户端任务: {task.task_id[:8]}")
            self.state.sessions.pop((task.socket_id, task.task_id), None)
            self.buffer.cleanup_tasks()
            return
        if self.task_deadline_expired(task):
            logger.info(f"跳过已超过 HTTP deadline 的任务: {task.task_id[:8]}")
            self.state.sessions.pop((task.socket_id, task.task_id), None)
            self.buffer.cleanup_tasks()
            return

        if task.type == 'cmd':
            self.handle_command_task(task)
        else:
            self.handle_audio_task(task)

    def handle_audio_task(self, task):
        """处理音频识别任务。"""
        self._mark_inference_active(task)
        try:
            result = self.pipeline.process(task)
        except Exception:
            logger.error(
                f"任务 {task.task_id[:8]} 推理管线执行失败",
                exc_info=getattr(task, "log_transcript", True),
            )
            result = Result(
                task_id=task.task_id,
                socket_id=task.socket_id,
                type=task.type,
                time_start=task.time_start,
                time_submit=task.time_submit,
                time_complete=time.time(),
                is_final=True,
                error_code=WORKER_TASK_ERROR_CODE,
                error_message=WORKER_TASK_ERROR_MESSAGE,
            )
            try:
                self.queue_out.put(result)
            finally:
                self.state.sessions.pop((task.socket_id, task.task_id), None)
                self.buffer.discard_session(task.socket_id, task.task_id)
            return
        finally:
            self._clear_inference_active()

        self.queue_out.put(result)
        if result.is_final:
            self.state.sessions.pop((task.socket_id, task.task_id), None)

    def _mark_inference_active(self, task) -> None:
        """Publish an atomic parent-visible lease before native inference."""
        if self.active_inference is None:
            return
        deadline = getattr(task, "deadline_monotonic", None)
        with self.active_inference.get_lock():
            self.active_inference[0] = time.monotonic()
            self.active_inference[1] = float(deadline or 0.0)

    def _clear_inference_active(self) -> None:
        if self.active_inference is None:
            return
        with self.active_inference.get_lock():
            self.active_inference[0] = 0.0
            self.active_inference[1] = 0.0

    def loop(self):
        """核心任务循环：drain 队列 → 清理断连 → 轮转执行一个。"""
        logger.info("TaskHandler 开始工作循环 (公平调度)")

        while True:
            try:
                if not self.drain_queue():
                    break

                task = self.buffer.pop()
                if task is None:
                    continue

                self.dispatch_task(task)

                self.cleanup()
            except InterruptedError:
                continue
            except Exception as e:
                logger.error(f"任务执行出错: {str(e)}", exc_info=True)

        logger.info("TaskHandler 工作循环结束")
