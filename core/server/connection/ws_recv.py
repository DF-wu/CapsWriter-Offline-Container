# coding: utf-8
"""
WebSocket 接收处理模块

处理客户端发送的音频数据，进行分段和缓冲，提交到识别队列。
"""

import asyncio
import json
import math
import queue
import time
import unicodedata
from base64 import b64decode
from binascii import Error as Base64Error
from typing import Optional

import websockets

from ..state import console
from ..schema import Task
from config_server import ServerConfig as Config
from core.protocol import AudioMessage
from core.constants import AudioFormat
from core.tools.my_status import Status
from .. import logger
from ..queue_limits import (
    MAX_TASK_AUDIO_BYTES,
    MAX_TASK_AUDIO_SECONDS,
    QUEUE_PUT_RETRY_SECONDS,
    WEBSOCKET_CLOSE_TIMEOUT_SECONDS,
)
from ..task_control import WEBSOCKET_TASK_LIMIT_COMMAND
from fork_server.runtime_limits import (
    DEFAULT_SERVER_MAX_WEBSOCKET_TASK_SECONDS,
)


# 麦克风接收状态指示器
status_mic = Status('正在接收音频', spinner='point')

MAX_TASK_ID_CHARS = 128
MAX_CONTEXT_CHARS = 8192
MAX_LANGUAGE_CHARS = 32
MIN_SEGMENT_DURATION_SECONDS = 1.0
MAX_SEGMENT_OVERLAP_SECONDS = 8.0
MAX_SEGMENTS_PER_MESSAGE = 66


class InvalidAudioMessage(ValueError):
    """A WebSocket audio message violates the bounded wire contract."""


class WebSocketConnectionLimiter:
    """Immediate event-loop-local admission for aggregate WS state."""

    def __init__(self, maximum: int) -> None:
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
            raise ValueError("maximum WebSocket connections must be a positive integer")
        self.maximum = maximum
        self.active = 0

    def try_acquire(self) -> bool:
        if self.active >= self.maximum:
            return False
        self.active += 1
        return True

    def release(self) -> None:
        if self.active <= 0:
            raise RuntimeError("WebSocket connection limiter released without a lease")
        self.active -= 1


def _abort_websocket_transport(websocket) -> None:
    """Best-effort hard close after a peer ignores the close handshake."""
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


async def _close_websocket_bounded(websocket, *, code: int, reason: str) -> None:
    """Send a policy close without letting an unresponsive peer retain state."""
    try:
        await asyncio.wait_for(
            websocket.close(code=code, reason=reason),
            timeout=WEBSOCKET_CLOSE_TIMEOUT_SECONDS,
        )
    except asyncio.CancelledError:
        _abort_websocket_transport(websocket)
        raise
    except websockets.ConnectionClosed:
        return
    except Exception:
        _abort_websocket_transport(websocket)


def _websocket_is_open(websocket) -> bool:
    """Best-effort liveness check compatible with old and new websockets APIs."""
    if getattr(websocket, "close_code", None) is not None:
        return False
    closed = getattr(websocket, "closed", None)
    if closed is not None:
        return not bool(closed)
    state = getattr(websocket, "state", None)
    return getattr(state, "name", "OPEN") == "OPEN"


async def _submit_task(queue_in, task, websocket) -> None:
    """Cooperatively wait for bounded queue capacity without spawning threads."""
    while _websocket_is_open(websocket):
        try:
            queue_in.put_nowait(task)
            return
        except queue.Full:
            await asyncio.sleep(QUEUE_PUT_RETRY_SECONDS)
    raise ConnectionError("WebSocket closed while waiting for recognizer capacity")


def _max_websocket_task_bytes() -> int:
    seconds = getattr(
        Config,
        "max_websocket_task_seconds",
        DEFAULT_SERVER_MAX_WEBSOCKET_TASK_SECONDS,
    )
    return AudioFormat.seconds_to_bytes(seconds)


def _finite_number(value, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidAudioMessage(f"{field_name} must be a finite number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise InvalidAudioMessage(f"{field_name} must be a finite number") from exc
    if not math.isfinite(value):
        raise InvalidAudioMessage(f"{field_name} must be a finite number")
    return value


def _bounded_control_free_text(
    value,
    field_name: str,
    *,
    max_chars: int,
    allow_empty: bool,
) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise InvalidAudioMessage(f"{field_name} must be a bounded string")
    if len(value) > max_chars:
        raise InvalidAudioMessage(f"{field_name} must be a bounded string")
    if any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in value):
        raise InvalidAudioMessage(f"{field_name} must not contain control characters")
    return value


def validate_audio_message(message: dict) -> tuple[AudioMessage, bytes]:
    """Validate metadata and strictly decode one bounded float32 PCM chunk."""
    if not isinstance(message, dict):
        raise InvalidAudioMessage("message must be a JSON object")

    task_id = _bounded_control_free_text(
        message.get("task_id"),
        "task_id",
        max_chars=MAX_TASK_ID_CHARS,
        allow_empty=False,
    )
    source = message.get("source")
    if not isinstance(source, str) or source not in {"mic", "file"}:
        raise InvalidAudioMessage("source must be 'mic' or 'file'")
    is_final = message.get("is_final")
    if not isinstance(is_final, bool):
        raise InvalidAudioMessage("is_final must be a boolean")

    seg_duration = _finite_number(
        message.get("seg_duration", 15.0),
        "seg_duration",
    )
    seg_overlap = _finite_number(
        message.get("seg_overlap", 2.0),
        "seg_overlap",
    )
    time_start = _finite_number(message.get("time_start"), "time_start")
    if seg_duration < MIN_SEGMENT_DURATION_SECONDS:
        raise InvalidAudioMessage(
            f"seg_duration must be >= {MIN_SEGMENT_DURATION_SECONDS:g}s"
        )
    if seg_overlap < 0:
        raise InvalidAudioMessage("seg_overlap must be >= 0")
    if seg_overlap > MAX_SEGMENT_OVERLAP_SECONDS:
        raise InvalidAudioMessage(
            f"seg_overlap must be <= {MAX_SEGMENT_OVERLAP_SECONDS:g}s"
        )
    if seg_overlap > seg_duration:
        raise InvalidAudioMessage("seg_overlap must not exceed seg_duration")
    if seg_duration + seg_overlap > MAX_TASK_AUDIO_SECONDS:
        raise InvalidAudioMessage(
            f"seg_duration plus seg_overlap must be <= "
            f"{MAX_TASK_AUDIO_SECONDS:g}s"
        )

    stride_bytes = AudioFormat.seconds_to_bytes(seg_duration)
    segment_bytes = AudioFormat.seconds_to_bytes(seg_duration + seg_overlap)
    if stride_bytes < AudioFormat.BYTES_PER_SAMPLE:
        raise InvalidAudioMessage("seg_duration is shorter than one audio sample")
    if (
        stride_bytes % AudioFormat.BYTES_PER_SAMPLE
        or segment_bytes % AudioFormat.BYTES_PER_SAMPLE
    ):
        raise InvalidAudioMessage("segment geometry must be float32 sample-aligned")

    context = _bounded_control_free_text(
        message.get("context", ""),
        "context",
        max_chars=MAX_CONTEXT_CHARS,
        allow_empty=True,
    )
    language = _bounded_control_free_text(
        message.get("language", "auto"),
        "language",
        max_chars=MAX_LANGUAGE_CHARS,
        allow_empty=True,
    )
    encoded = message.get("data")
    if not isinstance(encoded, str):
        raise InvalidAudioMessage("data must be a Base64 string")
    try:
        decoded = b64decode(encoded, validate=True)
    except (Base64Error, ValueError) as exc:
        raise InvalidAudioMessage("data is not valid Base64") from exc
    if len(decoded) > MAX_TASK_AUDIO_BYTES:
        raise InvalidAudioMessage(
            f"decoded audio chunk exceeds {MAX_TASK_AUDIO_BYTES} bytes"
        )
    if len(decoded) % AudioFormat.BYTES_PER_SAMPLE:
        raise InvalidAudioMessage("decoded float32 audio is not sample-aligned")

    return (
        AudioMessage(
            task_id=task_id,
            source=source,
            data=encoded,
            is_final=is_final,
            time_start=time_start,
            seg_duration=seg_duration,
            seg_overlap=seg_overlap,
            context=context,
            language=language,
        ),
        decoded,
    )


class AudioCache:
    """
    音频缓冲区

    用于缓存接收到的音频数据，直到达到分段阈值后提交处理。
    """
    def __init__(self):
        self.chunks: bytes = b''    # 音频数据缓冲
        self.offset: float = 0.0    # 当前偏移时间（秒）
        self.byte_count: int = 0    # 累计接收字节数
        self.task_id: Optional[str] = None
        self.source: Optional[str] = None
        self.rejected: bool = False

    @property
    def duration(self) -> float:
        """缓冲区音频时长（秒）"""
        return AudioFormat.bytes_to_seconds(len(self.chunks))

    @property
    def total_duration(self) -> float:
        """累计接收的音频总时长（秒）"""
        return AudioFormat.bytes_to_seconds(self.byte_count)

    def reset(self) -> None:
        """重置缓冲区"""
        self.chunks = b''
        self.offset = 0.0
        self.byte_count = 0
        self.task_id = None
        self.source = None
        self.rejected = False

    def bind_stream(self, task_id: str, source: str) -> None:
        """Allow only one sequential task/source stream per connection."""
        if self.task_id is None:
            self.task_id = task_id
            self.source = source
            return
        if task_id != self.task_id or source != self.source:
            raise InvalidAudioMessage(
                "task_id/source changed before the active stream was finalized"
            )


async def message_handler(websocket, message: dict, cache: AudioCache, app) -> None:
    """
    处理客户端发送的音频消息

    根据消息中的分段参数，将音频数据分段后提交到识别队列。
    """
    msg, data = validate_audio_message(message)
    queue_in = app.state.queue_in

    global status_mic
    is_start = cache.task_id is None
    socket_id = str(websocket.id)
    cache.bind_stream(msg.task_id, msg.source)

    if cache.rejected:
        # Keep the rejected identity bound until its final marker. Otherwise a
        # sender could continue the same task after the worker-side purge, or
        # splice a new identity into the rejected stream. No rejected audio is
        # retained or submitted.
        if msg.is_final:
            cache.reset()
        return

    total_stream_bytes = cache.byte_count + len(data)
    if total_stream_bytes > _max_websocket_task_bytes():
        # The decoded bytes were already validated. Reject this chunk, reset
        # ingress state, and enqueue a composite-session cancellation/error;
        # other tasks on this socket and colliding task IDs remain live.
        cache.chunks = b""
        cache.offset = 0.0
        cache.byte_count = 0
        cache.rejected = True
        await _submit_task(
            queue_in,
            Task(
                type="cmd",
                task_id=msg.task_id,
                data=b"",
                offset=0,
                overlap=0,
                socket_id=socket_id,
                is_final=True,
                time_start=msg.time_start,
                time_submit=time.time(),
                command=WEBSOCKET_TASK_LIMIT_COMMAND,
            ),
            websocket,
        )
        logger.warning(
            f"拒绝超过 WebSocket task 音讯上限的 stream，"
            f"客户端ID {socket_id}，任务ID {msg.task_id[:8]}，"
            f"累计 bytes={total_stream_bytes}"
        )
        if msg.is_final:
            cache.reset()
        return

    # 麦克风首次消息 → GPU 加速
    if is_start and msg.source == 'mic' and Config.gpu_boost_enabled:
        await _submit_task(queue_in, Task(
            type='cmd',
            task_id='gpu_boost',
            data=b'', offset=0, overlap=0,
            socket_id=socket_id, is_final=False,
            time_start=0, time_submit=0,
            command='gpu_boost'
        ), websocket)

    # 从消息中获取分段参数
    seg_threshold = msg.seg_duration + msg.seg_overlap

    try:
        # Base64 was already decoded and float32-alignment checked above.
        cache.chunks += data
        cache.byte_count += len(data)

        segment_bytes = AudioFormat.seconds_to_bytes(
            msg.seg_duration + msg.seg_overlap
        )
        stride_bytes = AudioFormat.seconds_to_bytes(msg.seg_duration)
        emitted_segments = 0

        if not msg.is_final:
            # 打印状态消息
            if msg.source == 'mic':
                status_mic.start()
            if msg.source == 'file' and is_start:
                console.print('正在接收音频文件...')
                logger.info(f"开始接收音频文件，任务ID: {msg.task_id}")

            # 若缓冲已达到分段阈值，将片段作为任务提交
            while cache.duration >= seg_threshold:
                emitted_segments += 1
                if emitted_segments > MAX_SEGMENTS_PER_MESSAGE:
                    raise InvalidAudioMessage(
                        "one WebSocket message expands to too many segments"
                    )
                segment_data = cache.chunks[:segment_bytes]
                cache.chunks = cache.chunks[stride_bytes:]

                task = Task(
                    type=msg.source,
                    data=segment_data,
                    offset=cache.offset,
                    task_id=msg.task_id,
                    socket_id=socket_id,
                    overlap=msg.seg_overlap,
                    is_final=False,
                    time_start=msg.time_start,
                    time_submit=time.time(),
                    context=msg.context,
                    language=msg.language,
                )
                cache.offset += msg.seg_duration
                await _submit_task(queue_in, task, websocket)
                logger.debug(
                    f"提交音频片段，任务ID: {msg.task_id}, "
                    f"偏移: {cache.offset}s, 缓冲区: {len(cache.chunks)} bytes"
                )

        else:  # is_final
            # 打印状态消息
            if msg.source == 'mic':
                status_mic.stop()
            elif msg.source == 'file':
                print(f'音频文件接收完毕，时长 {cache.total_duration:.2f}s')
                logger.info(f"音频文件接收完毕，任务ID: {msg.task_id}, 时长: {cache.total_duration:.2f}s")

            # Keep every task at or below the 64-second cross-process ceiling,
            # including a final cache assembled from unusually sized messages.
            while len(cache.chunks) > segment_bytes:
                emitted_segments += 1
                if emitted_segments > MAX_SEGMENTS_PER_MESSAGE:
                    raise InvalidAudioMessage(
                        "one WebSocket message expands to too many segments"
                    )
                segment_data = cache.chunks[:segment_bytes]
                cache.chunks = cache.chunks[stride_bytes:]
                task = Task(
                    type=msg.source,
                    data=segment_data,
                    offset=cache.offset,
                    task_id=msg.task_id,
                    socket_id=socket_id,
                    overlap=msg.seg_overlap,
                    is_final=False,
                    time_start=msg.time_start,
                    time_submit=time.time(),
                    context=msg.context,
                    language=msg.language,
                )
                cache.offset += msg.seg_duration
                await _submit_task(queue_in, task, websocket)

            # 提交最终片段
            task = Task(
                type=msg.source,
                data=cache.chunks,
                offset=cache.offset,
                task_id=msg.task_id,
                socket_id=socket_id,
                overlap=msg.seg_overlap,
                is_final=True,
                time_start=msg.time_start,
                time_submit=time.time(),
                context=msg.context,
                language=msg.language,
            )
            await _submit_task(queue_in, task, websocket)
            logger.debug(f"提交最终片段，任务ID: {msg.task_id}, 数据大小: {len(cache.chunks)} bytes")

            # 重置缓冲区
            cache.reset()

    except Exception as e:
        logger.error(f"音频数据处理错误，任务ID: {msg.task_id}: {e}", exc_info=True)
        raise


async def ws_recv(websocket, app, admission=None) -> None:
    """Admit one bounded connection immediately, then own its lease."""
    if admission is not None and not admission.try_acquire():
        await _close_websocket_bounded(
            websocket,
            code=1013,
            reason="Server connection limit reached",
        )
        return
    try:
        await _ws_recv_admitted(websocket, app)
    finally:
        if admission is not None:
            admission.release()


async def _ws_recv_admitted(websocket, app) -> None:
    """
    WebSocket 接收主函数

    处理单个客户端连接，接收音频数据并分发处理。
    """
    global status_mic

    # 登记 socket 到连接池
    state = app.state
    sockets = state.sockets
    sockets_id = state.sockets_id
    socket_id = str(websocket.id)
    sockets[socket_id] = websocket
    sockets_id.append(socket_id)
    remote = websocket.remote_address
    console.print(f'[bold green]客户端已连接: {remote[0]}:{remote[1]}[/bold green]\n')
    logger.info(f"新客户端连接: {websocket}, ID: {socket_id}")

    # 创建音频缓冲区
    cache = AudioCache()

    # 接收并处理消息
    try:
        async for raw_message in websocket:
            data = json.loads(raw_message)
            await message_handler(websocket, data, cache, app)

        logger.info(f"客户端正常关闭连接: {socket_id}")

    except (json.JSONDecodeError, UnicodeDecodeError, InvalidAudioMessage) as e:
        logger.warning(
            f"拒绝无效 WebSocket 音訊訊息，客户端ID {socket_id}: {e}"
        )
        await _close_websocket_bounded(
            websocket,
            code=1008,
            reason="Invalid audio message",
        )
    except websockets.ConnectionClosed:
        console.print("ConnectionClosed...")
        logger.warning(f"客户端连接已关闭: {socket_id}")
    except websockets.InvalidState:
        console.print("InvalidState...")
        logger.error(f"WebSocket 状态异常: {socket_id}")
    except Exception as e:
        console.print("Exception:", e)
        logger.error(f"WebSocket 接收异常，客户端ID {socket_id}: {e}", exc_info=True)
    finally:
        # 清理资源
        status_mic.stop()
        status_mic.on = False
        sockets.pop(socket_id, None)
        if socket_id in sockets_id:
            sockets_id.remove(socket_id)

        console.print(f'[bold red]客户端已断开: {remote[0]}:{remote[1]}[/bold red]\n')

        # 注意：session 清理由 TaskHandler 在子进程中定期执行
        # （通过检查 sockets_id 判断客户端是否已断开）
        logger.debug(f"客户端资源已清理: {socket_id}")
