# coding: utf-8
"""Bounded, isolated delivery of recognition results to WebSocket peers."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
import queue
import threading
from typing import Any

from ..queue_limits import (
    OUTPUT_QUEUE_MAX_RESULTS,
    WEBSOCKET_CLOSE_TIMEOUT_SECONDS,
    WEBSOCKET_SEND_TIMEOUT_SECONDS,
    MAX_PENDING_RESULTS_PER_WEBSOCKET,
)


def _abort_websocket_transport(websocket) -> None:
    """Hard-close a peer that ignores the bounded close handshake."""

    transport = getattr(websocket, "transport", None)
    if transport is None:
        transport = getattr(getattr(websocket, "protocol", None), "transport", None)
    abort = getattr(transport, "abort", None)
    if abort is None:
        return
    try:
        abort()
    except Exception:
        pass


class AsyncResultQueueReader:
    """One persistent daemon bridge from multiprocessing.Queue to asyncio."""

    _POLL_SECONDS = 0.5

    def __init__(self, queue_out) -> None:
        self._queue_out = queue_out
        self._loop = asyncio.get_running_loop()
        self._items: asyncio.Queue = asyncio.Queue(
            maxsize=OUTPUT_QUEUE_MAX_RESULTS
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="capswriter-result-queue-reader",
            daemon=True,
        )
        self._thread.start()

    async def get(self):
        return await self._items.get()

    def close(self) -> None:
        self._stop.set()

    async def aclose(self) -> None:
        """Stop the bridge and give its bounded poll loop time to exit."""
        self.close()
        deadline = self._loop.time() + self._POLL_SECONDS + 0.1
        while self._thread.is_alive() and self._loop.time() < deadline:
            await asyncio.sleep(0.01)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue_out.get(timeout=self._POLL_SECONDS)
            except queue.Empty:
                continue
            except (EOFError, OSError, ValueError):
                return
            put_item = self._items.put(item)
            try:
                delivery = asyncio.run_coroutine_threadsafe(
                    put_item,
                    self._loop,
                )
            except RuntimeError:
                put_item.close()
                return
            while not self._stop.is_set():
                try:
                    delivery.result(timeout=self._POLL_SECONDS)
                    break
                except FutureTimeoutError:
                    continue
                except Exception:
                    return
            else:
                delivery.cancel()
                return
            if item is None:
                return


@dataclass
class _SendItem:
    websocket: Any
    payload: str
    task_id: str
    is_final: bool


@dataclass
class _PeerChannel:
    task: asyncio.Task
    pending: OrderedDict[str, _SendItem]


class WebSocketResultDispatcher:
    """Keep at most one active and one latest pending payload per peer."""

    def __init__(self, state, logger) -> None:
        self._state = state
        self._logger = logger
        self._channels: dict[str, _PeerChannel] = {}
        self._closing = False

    @property
    def active_peer_count(self) -> int:
        return len(self._channels)

    def submit(
        self,
        websocket,
        payload: str,
        *,
        socket_id: str,
        task_id: str,
        is_final: bool,
    ) -> None:
        if self._closing:
            return
        item = _SendItem(websocket, payload, task_id, is_final)
        channel = self._channels.get(socket_id)
        if channel is not None and channel.task.done():
            self._consume_task(channel.task)
            self._channels.pop(socket_id, None)
            channel = None
        if channel is None:
            task = asyncio.create_task(
                self._run_peer(socket_id, item),
                name=f"capswriter-ws-send-{socket_id[:16]}",
            )
            self._channels[socket_id] = _PeerChannel(
                task=task,
                pending=OrderedDict(),
            )
            return

        existing = channel.pending.get(task_id)
        if existing is not None:
            if existing.is_final and not item.is_final:
                return
            self._logger.warning(
                f"客户端 {socket_id} 消费结果过慢；合并任务 {task_id[:8]} 的结果"
            )
            channel.pending[task_id] = item
            return

        if len(channel.pending) >= MAX_PENDING_RESULTS_PER_WEBSOCKET:
            self._logger.warning(
                f"客户端 {socket_id} 待发送任务过多；隔离慢速连接"
            )
            channel.pending.clear()
            channel.task.cancel()
            close_task = asyncio.create_task(
                self._close_overloaded_peer(socket_id, websocket),
                name=f"capswriter-ws-close-{socket_id[:16]}",
            )
            channel.task = close_task
            return

        # Keep one cumulative latest snapshot per task. This preserves final
        # results when composite worker sessions interleave on one socket.
        channel.pending[task_id] = item

    async def _run_peer(self, socket_id: str, item: _SendItem) -> None:
        current = asyncio.current_task()
        try:
            while True:
                if not await self._send_one(socket_id, item):
                    return
                # Let the shared result loop submit a just-arrived successor
                # before deciding that this short-lived channel is idle.
                await asyncio.sleep(0)
                channel = self._channels.get(socket_id)
                if channel is None or channel.task is not current:
                    return
                if not channel.pending:
                    return
                _task_id, item = channel.pending.popitem(last=False)
        finally:
            channel = self._channels.get(socket_id)
            if channel is not None and channel.task is current:
                self._channels.pop(socket_id, None)

    async def _send_one(self, socket_id: str, item: _SendItem) -> bool:
        try:
            await asyncio.wait_for(
                item.websocket.send(item.payload),
                timeout=WEBSOCKET_SEND_TIMEOUT_SECONDS,
            )
            return True
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            self._logger.warning(
                f"客户端 {socket_id} 发送超时；隔离慢速连接"
            )
        except Exception:
            self._logger.warning(
                f"客户端 {socket_id} 发送失败；移除连接",
                exc_info=False,
            )
        await self._evict_and_close(socket_id, item.websocket)
        return False

    async def _close_overloaded_peer(self, socket_id: str, websocket) -> None:
        current = asyncio.current_task()
        try:
            await self._evict_and_close(socket_id, websocket)
        finally:
            channel = self._channels.get(socket_id)
            if channel is not None and channel.task is current:
                self._channels.pop(socket_id, None)

    async def _evict_and_close(self, socket_id: str, websocket) -> None:
        sockets = getattr(self._state, "sockets", None)
        if sockets is not None and sockets.get(socket_id) is websocket:
            sockets.pop(socket_id, None)
        sockets_id = getattr(self._state, "sockets_id", None)
        if sockets_id is not None:
            try:
                sockets_id.remove(socket_id)
            except (BrokenPipeError, EOFError, OSError, TypeError, ValueError):
                pass
        try:
            await asyncio.wait_for(
                websocket.close(code=1013, reason="Slow result consumer"),
                timeout=WEBSOCKET_CLOSE_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            _abort_websocket_transport(websocket)
            raise
        except Exception:
            _abort_websocket_transport(websocket)

    async def aclose(self) -> None:
        self._closing = True
        tasks = [channel.task for channel in self._channels.values()]
        if not tasks:
            self._channels.clear()
            return
        timeout = (
            WEBSOCKET_SEND_TIMEOUT_SECONDS
            + WEBSOCKET_CLOSE_TIMEOUT_SECONDS
            + 0.5
        )
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            self._channels.clear()

    @staticmethod
    def _consume_task(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        task.exception()
