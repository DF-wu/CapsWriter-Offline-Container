# coding: utf-8
"""
HTTP 任務路由器

為 HTTP API 提供 task_id ↔ asyncio.Future 的對應關係:

- HTTP handler 呼叫 register(task_id) 取得 Future, 同時把合成的
  socket_id 加入 state.sockets_id (跨進程 ListProxy), 否則
  TaskHandler 會把 task 當作斷線客戶端而 skip。
- ws_send_with_http 從 queue_out 取到 Result 後呼叫 try_resolve:
  - task_id 在 pending → resolve future, 返回 True (跳過 ws 廣播)
  - 不在 pending → 返回 False, 走上游原 ws 廣播流程
- cleanup 路徑 (timeout / cancel / 完成) 都會移除 sockets_id 條目。
"""

from __future__ import annotations
import asyncio
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from core.server.schema import Result


def _synthetic_socket_id(task_id: str) -> str:
    """HTTP 任務專用的合成 socket_id。"""
    return f"http:{task_id}"


class TaskRouter:
    """
    HTTP 任務 future 路由器 (單例)。

    與上游耦合面:
    - 讀寫 state.sockets_id (ListProxy)
    - resolve future 跨 thread (queue_out.get 在執行緒池)
    """

    def __init__(self) -> None:
        self._pending: Dict[str, asyncio.Future] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._state = None  # core.server.state.ServerState

    def bind(self, state, loop: asyncio.AbstractEventLoop) -> None:
        """在 server 啟動時呼叫一次, 注入 state 與 event loop。"""
        self._state = state
        self._loop = loop

    def is_bound(self) -> bool:
        return self._state is not None and self._loop is not None

    def synthetic_socket_id(self, task_id: str) -> str:
        return _synthetic_socket_id(task_id)

    def register(self, task_id: str) -> asyncio.Future:
        """
        註冊 HTTP 任務, 返回可 await 的 Future。
        同時把合成 socket_id 加入 state.sockets_id, 讓識別子進程不會丟棄任務。
        """
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        fut = self._loop.create_future()
        self._pending[task_id] = fut

        if self._state is not None and self._state.sockets_id is not None:
            self._state.sockets_id.append(_synthetic_socket_id(task_id))
        return fut

    def cancel(self, task_id: str) -> None:
        """timeout/異常時清理 future 與 sockets_id。"""
        fut = self._pending.pop(task_id, None)
        if fut is not None and not fut.done():
            fut.cancel()
        self._remove_synthetic(task_id)

    def try_resolve(self, result: "Result") -> bool:
        """
        ws_send_with_http 取到結果時呼叫。

        Returns:
            False — task_id 不在 pending, 不是 HTTP 任務, 走原 ws 派發
            True  — 已處理 (resolve 或中間結果), ws_send 不再派發
        """
        fut = self._pending.get(result.task_id)
        if fut is None:
            return False

        if result.is_final:
            self._pending.pop(result.task_id, None)
            self._remove_synthetic(result.task_id)
            if not fut.done():
                if self._loop is not None and self._loop.is_running():
                    self._loop.call_soon_threadsafe(self._set_result, fut, result)
                else:
                    self._set_result(fut, result)
        return True

    @staticmethod
    def _set_result(fut: asyncio.Future, result: Any) -> None:
        if not fut.done():
            fut.set_result(result)

    def _remove_synthetic(self, task_id: str) -> None:
        if self._state is None or self._state.sockets_id is None:
            return
        try:
            self._state.sockets_id.remove(_synthetic_socket_id(task_id))
        except ValueError:
            pass

    @property
    def queue_in(self):
        """便捷存取 state.queue_in (HTTP API 推 Task 用)。"""
        if self._state is None:
            raise RuntimeError("TaskRouter not bound; call bind(state, loop) first")
        return self._state.queue_in


# 全局單例 (與上游 state 透過 bind() 注入耦合)
router = TaskRouter()
