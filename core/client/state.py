# coding: utf-8
"""
客户端状态管理模块

提供 ClientState 类用于管理客户端的全局状态。
使用 dataclass 提供类型安全和清晰的状态定义。
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Dict, Any

if TYPE_CHECKING:
    import sounddevice as sd
    from websockets.legacy.client import WebSocketClientProtocol
    from .app import CapsWriterClient

from rich.console import Console
from rich.theme import Theme

from . import logger


# 配置 Rich console
_theme = Theme({
    'markdown.code': 'cyan',
    'markdown.item.number': 'yellow'
})
console = Console(highlight=False, soft_wrap=True, theme=_theme)


# The PortAudio callback runs every 50 ms. Reserve at most 128 copied blocks
# (about 6.4 seconds) across the callback thread, loop callbacks, and recorder
# queue. A small separate headroom keeps begin/finish control messages able to
# enter the queue when audio is saturated.
CLIENT_AUDIO_QUEUE_MAX_CHUNKS = 128
CLIENT_AUDIO_QUEUE_CONTROL_HEADROOM = 8


def websocket_state_name(websocket: object | None) -> str | None:
    """Return a portable WebSocket lifecycle state without importing its API.

    ``websockets`` 14+ connections expose a ``state`` enum but no ``closed``
    attribute.  Legacy protocols expose ``closed`` (and may also expose
    ``state``).  Normalizing both shapes here keeps the desktop client usable
    with either dependency generation and keeps dependency-light tests simple.
    """

    if websocket is None:
        return "CLOSED"

    state = getattr(websocket, "state", None)
    if state is not None:
        name = getattr(state, "name", None)
        if isinstance(name, str):
            return name.upper()
        if isinstance(state, str):
            return state.upper()
        try:
            # websockets.protocol.State is an IntEnum:
            # CONNECTING=0, OPEN=1, CLOSING=2, CLOSED=3.
            return {
                0: "CONNECTING",
                1: "OPEN",
                2: "CLOSING",
                3: "CLOSED",
            }.get(int(state))
        except (TypeError, ValueError):
            return None

    closed = getattr(websocket, "closed", None)
    if closed is not None:
        return "CLOSED" if bool(closed) else "OPEN"
    return None


def websocket_is_open(websocket: object | None) -> bool:
    """Return whether a modern or legacy WebSocket is fully open."""

    return websocket_state_name(websocket) == "OPEN"


def websocket_is_closed(websocket: object | None) -> bool:
    """Return whether a modern or legacy WebSocket is fully closed."""

    return websocket_state_name(websocket) == "CLOSED"


@dataclass
class ClientState:
    """
    客户端运行状态

    管理客户端运行过程中的所有共享状态，包括事件循环、消息队列、
    WebSocket 连接、音频流和录音状态等。

    Attributes:
        loop: asyncio 事件循环
        queue_in: 音频数据输入队列
        queue_out: 处理结果输出队列（保留）
        websocket: WebSocket 客户端连接
        stream: 音频输入流
        recording: 是否正在录音
        recording_start_time: 录音开始时间戳
        audio_files: 任务ID到音频文件路径的映射
        last_recognition_text: 最近一次识别的最终文本（热词替换后），供"添加纠错记录"使用
    """

    queue_in: asyncio.Queue = field(
        default_factory=lambda: asyncio.Queue(
            maxsize=(
                CLIENT_AUDIO_QUEUE_MAX_CHUNKS
                + CLIENT_AUDIO_QUEUE_CONTROL_HEADROOM
            )
        )
    )
    queue_out: asyncio.Queue = field(
        default_factory=lambda: asyncio.Queue(maxsize=4)
    )
    websocket: Optional[WebSocketClientProtocol] = None
    stream: Optional[sd.InputStream] = None
    app: Optional[CapsWriterClient] = None

    recording: bool = False
    recording_start_time: float = 0.0
    audio_files: Dict[str, Path] = field(default_factory=dict)

    # 最近一次识别结果（用于手动添加纠错记录）
    last_recognition_text: Optional[str] = None
    
    # 最近一次输出内容（如果是 LLM 润色，则是润色结果；否则是原始识别结果）
    last_output_text: Optional[str] = None

    _audio_ingress_lock: Any = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )
    _audio_reserved_chunks: int = field(default=0, init=False, repr=False)
    _audio_input_overflow: bool = field(default=False, init=False, repr=False)
    

    
    def reset(self) -> None:
        """
        重置状态
        
        清理所有状态，关闭连接和流。用于重新初始化或退出时清理。
        """
        logger.debug("正在重置客户端状态...")
        
        # 关闭 WebSocket 连接
        ws = self.websocket
        if ws is not None:
            try:
                if (
                    not websocket_is_closed(ws)
                    and self.app
                    and self.app.loop
                    and self.app.loop.is_running()
                ):
                    asyncio.run_coroutine_threadsafe(ws.close(), self.app.loop)
            except Exception:
                pass
            self.websocket = None
        
        # 关闭音频流
        if self.stream is not None:
            try:
                self.stream.close()
                logger.debug("音频流已关闭")
            except Exception:
                pass
            self.stream = None
        
        # 重置其他状态
        self.recording = False
        self.recording_start_time = 0.0
        self.audio_files.clear()
        
        logger.debug("客户端状态重置完成")

    def reserve_audio_chunk(self) -> bool:
        """Reserve one callback-owned audio block without blocking PortAudio."""

        with self._audio_ingress_lock:
            if self._audio_reserved_chunks >= CLIENT_AUDIO_QUEUE_MAX_CHUNKS:
                self._audio_input_overflow = True
                return False
            self._audio_reserved_chunks += 1
            return True

    def release_audio_chunk(self) -> None:
        """Release a reservation after a data event leaves the queue."""

        with self._audio_ingress_lock:
            if self._audio_reserved_chunks > 0:
                self._audio_reserved_chunks -= 1

    def mark_audio_input_overflow(self) -> None:
        with self._audio_ingress_lock:
            self._audio_input_overflow = True

    def consume_audio_input_overflow(self) -> bool:
        """Return and clear the current recording's overflow signal."""

        with self._audio_ingress_lock:
            overflowed = self._audio_input_overflow
            self._audio_input_overflow = False
            return overflowed

    @property
    def pending_audio_chunks(self) -> int:
        with self._audio_ingress_lock:
            return self._audio_reserved_chunks
    
    def start_recording(self, start_time: float) -> None:
        """
        开始录音
        
        Args:
            start_time: 录音开始的时间戳
        """
        self.recording = True
        self.recording_start_time = start_time
        self.consume_audio_input_overflow()
        logger.debug(f"录音状态已更新: recording=True, start_time={start_time:.2f}")
    
    def stop_recording(self) -> float:
        """
        停止录音
        
        Returns:
            录音持续时间（秒）
        """
        duration = 0.0
        if self.recording_start_time > 0:
            duration = time.time() - self.recording_start_time
        
        self.recording = False
        self.recording_start_time = 0.0
        logger.debug(f"录音状态已更新: recording=False, duration={duration:.2f}s")
        return duration
    
    @property
    def is_connected(self) -> bool:
        """检查 WebSocket 是否已连接"""
        if self.websocket is None:
            return False
        return websocket_is_open(self.websocket)
    
    def register_audio_file(self, task_id: str, file_path: Path) -> None:
        """
        注册音频文件
        
        Args:
            task_id: 任务ID
            file_path: 音频文件路径
        """
        self.audio_files[task_id] = file_path
        logger.debug(f"注册音频文件: task_id={task_id}, path={file_path}")
    
    def pop_audio_file(self, task_id: str) -> Optional[Path]:
        """
        获取并移除音频文件路径
        
        Args:
            task_id: 任务ID
            
        Returns:
            音频文件路径，如果不存在则返回 None
        """
        file_path = self.audio_files.pop(task_id, None)
        if file_path:
            logger.debug(f"获取音频文件: task_id={task_id}, path={file_path}")
        return file_path

    def set_output_text(self, text: str) -> None:
        """
        设置最近一次输出文本
        
        Args:
            text: 输出文本内容
        """
        self.last_output_text = text

