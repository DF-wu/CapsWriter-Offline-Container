# coding: utf-8
"""Bounded, cancellation-safe admission control for transcription requests."""

from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager
from tempfile import SpooledTemporaryFile
from typing import Any, AsyncIterator, Awaitable, Callable


class AdmissionQueueFullError(Exception):
    """Raised before reading a request body when the bounded wait queue is full."""


class ReplayableReceive:
    """Serialize ASGI receive probes and replay request events they consume.

    ASGI exposes disconnect notification through the same ``receive`` callable
    that carries request-body events.  A queued admission waiter therefore
    cannot probe the socket with ``Request.is_disconnected()``: that helper may
    consume a body event which the later multipart parser would never see.

    This wrapper stores probed request bodies in a small spooled file and keeps
    their ASGI metadata in FIFO order.  Multipart parsing receives the exact
    event sequence later, while large pending uploads spill to disk instead of
    accumulating in Python memory.  Once observed, ``http.disconnect`` remains
    sticky for every subsequent receiver.
    """

    _SPOOL_MEMORY_BYTES = 1024 * 1024

    def __init__(self, receive: Callable[[], Awaitable[dict[str, Any]]]) -> None:
        self._receive = receive
        self._lock = asyncio.Lock()
        self._buffer: deque[tuple[dict[str, Any], int, int]] = deque()
        self._spool = SpooledTemporaryFile(max_size=self._SPOOL_MEMORY_BYTES)
        self._disconnected = False
        self._closed = False

    @property
    def disconnected(self) -> bool:
        return self._disconnected

    @property
    def buffered_event_count(self) -> int:
        """Number of replay records (adjacent request events are coalesced)."""
        return len(self._buffer)

    def _remember_request(self, message: dict[str, Any]) -> None:
        body = bytes(message.get("body", b""))
        offset = self._spool.seek(0, 2)
        self._spool.write(body)
        metadata = dict(message)
        metadata.pop("body", None)
        if self._buffer:
            previous, previous_offset, previous_length = self._buffer[-1]
            if (
                previous.get("type") == "http.request"
                and previous.get("more_body", False)
            ):
                # ASGI consumers may receive different chunk boundaries.  One
                # aggregate record prevents a peer from growing a deque of
                # empty/tiny events while it waits for admission.
                self._buffer[-1] = (
                    metadata,
                    previous_offset,
                    previous_length + len(body),
                )
                return
        self._buffer.append((metadata, offset, len(body)))

    def _replay_request(self) -> dict[str, Any]:
        metadata, offset, length = self._buffer.popleft()
        self._spool.seek(offset)
        body = self._spool.read(length)
        return {**metadata, "body": body}

    async def probe_disconnect(self) -> bool:
        """Read one raw ASGI event, preserving it unless it is a disconnect."""
        async with self._lock:
            if self._disconnected:
                return True
            message = await self._receive()
            if message.get("type") == "http.disconnect":
                self._disconnected = True
                return True
            if message.get("type") == "http.request":
                self._remember_request(message)
            return False

    async def wait_for_disconnect(self) -> None:
        while not await self.probe_disconnect():
            pass

    async def __call__(self) -> dict[str, Any]:
        async with self._lock:
            if self._buffer:
                return self._replay_request()
            if self._disconnected:
                return {"type": "http.disconnect"}
            message = await self._receive()
            if message.get("type") == "http.disconnect":
                self._disconnected = True
            return message

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._buffer.clear()
        self._spool.close()


class AdmissionController:
    """Limit active work and the number of callers allowed to wait for a slot."""

    def __init__(self, max_active: int, max_pending: int) -> None:
        self.max_active = max(1, int(max_active))
        self.max_pending = max(0, int(max_pending))
        self._active = 0
        self._waiters: deque[asyncio.Future[None]] = deque()
        self._lock = asyncio.Lock()

    @property
    def active(self) -> int:
        return self._active

    @property
    def waiting(self) -> int:
        return len(self._waiters)

    def _wake_next_locked(self) -> None:
        while self._waiters and self._active < self.max_active:
            waiter = self._waiters.popleft()
            if waiter.cancelled():
                continue
            self._active += 1
            waiter.set_result(None)
            return

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        acquired = False
        waiter: asyncio.Future[None] | None = None
        async with self._lock:
            if self._active < self.max_active and not self._waiters:
                self._active += 1
                acquired = True
            else:
                if len(self._waiters) >= self.max_pending:
                    raise AdmissionQueueFullError
                waiter = asyncio.get_running_loop().create_future()
                self._waiters.append(waiter)

        if waiter is not None:
            try:
                await waiter
                acquired = True
            except BaseException:
                # Cancellation may race with a release that already reserved an
                # active slot for this waiter.  Reclaim either queued or reserved
                # capacity before propagating the cancellation.
                async with self._lock:
                    try:
                        self._waiters.remove(waiter)
                    except ValueError:
                        if waiter.done() and not waiter.cancelled():
                            self._active -= 1
                            self._wake_next_locked()
                raise

        try:
            yield
        finally:
            if acquired:
                async with self._lock:
                    self._active -= 1
                    self._wake_next_locked()
