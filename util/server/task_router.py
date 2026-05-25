# coding: utf-8
"""
HTTP 任务路由器

为 HTTP API 提供 task_id ↔ asyncio.Future 的对应关系：

- HTTP handler 调用 register(task_id) 取得一个 Future。
- 同时在 Cosmic.sockets_id 跨进程清单注册一个合成的 socket_id
  (recognizer 子进程会用 socket_id 判断连接是否仍存活,
   见 server_init_recognizer.py 中的 sockets_id 检查)。
- ws_send 从 queue_out 取到 Result 后调用 try_resolve;
  若 task_id 在 pending 中, 由本模块处理 (resolve future), ws_send 跳过 WebSocket 派发。
- 任何 cleanup 路径 (超时/异常/完成) 都会移除合成的 socket_id。
"""

import asyncio
from typing import Dict, Optional

from util.server.server_cosmic import Cosmic
from util.server.server_classes import Result


def _synthetic_socket_id(task_id: str) -> str:
    """HTTP 任务专用的合成 socket_id。"""
    return f"http:{task_id}"


class TaskRouter:
    """HTTP 任务 future 路由器 (单例)。"""

    def __init__(self) -> None:
        self._pending: Dict[str, asyncio.Future] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """由 HTTP server 启动时绑定运行中的事件循环。"""
        self._loop = loop

    def register(self, task_id: str) -> asyncio.Future:
        """
        注册一个 HTTP 任务, 返回可 await 的 Future。
        同时将合成 socket_id 加入 Cosmic.sockets_id 跨进程清单, 让识别子进程不会丢弃任务。
        """
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        fut = self._loop.create_future()
        self._pending[task_id] = fut

        if Cosmic.sockets_id is not None:
            Cosmic.sockets_id.append(_synthetic_socket_id(task_id))
        return fut

    def cancel(self, task_id: str) -> None:
        """超时或异常路径调用; 清理 future 与跨进程清单条目。"""
        fut = self._pending.pop(task_id, None)
        if fut is not None and not fut.done():
            fut.cancel()
        self._remove_synthetic(task_id)

    def synthetic_socket_id(self, task_id: str) -> str:
        """供 HTTP handler 在构造 Task 时使用。"""
        return _synthetic_socket_id(task_id)

    def try_resolve(self, result: Result) -> bool:
        """
        ws_send 取到结果时调用。

        - 若 result.task_id 不在 pending: 返回 False, ws_send 走原 WebSocket 派发。
        - 若在 pending 但非 final: 返回 True (HTTP 不关心中间结果), ws_send 不再派发。
        - 若在 pending 且 is_final=True: resolve future, 移除条目, 返回 True。
        """
        fut = self._pending.get(result.task_id)
        if fut is None:
            return False

        if result.is_final:
            self._pending.pop(result.task_id, None)
            self._remove_synthetic(result.task_id)
            if not fut.done():
                if self._loop is not None and self._loop.is_running():
                    self._loop.call_soon_threadsafe(fut.set_result, result)
                else:
                    fut.set_result(result)
        return True

    @staticmethod
    def _remove_synthetic(task_id: str) -> None:
        if Cosmic.sockets_id is None:
            return
        try:
            Cosmic.sockets_id.remove(_synthetic_socket_id(task_id))
        except ValueError:
            pass


# 全局单例 (与 Cosmic 同模式)
router = TaskRouter()
