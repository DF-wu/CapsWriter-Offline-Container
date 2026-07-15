# coding: utf-8
"""
WebSocket 接收处理模块

处理客户端发送的音频数据，进行分段和缓冲，提交到识别队列。
"""

import json
import math
import time
from base64 import b64decode
from binascii import Error as Base64Error
from typing import Optional

import websockets

from util.server.server_cosmic import console, Cosmic
from util.server.server_classes import Task
from util.constants import AudioFormat
from util.tools.my_status import Status
from . import logger



# 麦克风接收状态指示器
status_mic = Status('正在接收音频', spinner='point')


# 这些上限涵盖官方客户端的 15/2 秒麦克风终止包与 60/4 秒文件包，
# 同时避免客户端控制的分段参数造成无限循环或无界缓存。
MAX_AUDIO_CHUNK_BYTES = 4 * 1024 * 1024
MAX_SEGMENT_DURATION_SECONDS = 300.0
MAX_SEGMENT_OVERLAP_SECONDS = 30.0
MAX_TASK_ID_CHARS = 128
MAX_CONTEXT_CHARS = 8192


class InvalidAudioMessage(ValueError):
    """客户端音频消息不符合受支持协议。"""


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

    def bind_stream(self, task_id: str, source: str) -> None:
        """一个连接同一时间只允许传送一个顺序音频流。"""
        if self.task_id is None:
            self.task_id = task_id
            self.source = source
            return
        if task_id != self.task_id or source != self.source:
            raise InvalidAudioMessage(
                "task_id/source changed before the active stream was finalized"
            )


def _finite_number(value, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidAudioMessage(f"{field_name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise InvalidAudioMessage(f"{field_name} must be a finite number")
    return value


def validate_audio_message(message: dict) -> bytes:
    """验证消息元数据并严格解码 Base64 音频块。"""
    if not isinstance(message, dict):
        raise InvalidAudioMessage("message must be a JSON object")

    task_id = message.get("task_id")
    if not isinstance(task_id, str) or not task_id or len(task_id) > MAX_TASK_ID_CHARS:
        raise InvalidAudioMessage("task_id must be a non-empty bounded string")
    if any(ord(char) < 32 or ord(char) == 127 for char in task_id):
        raise InvalidAudioMessage("task_id must not contain control characters")

    if message.get("source") not in {"mic", "file"}:
        raise InvalidAudioMessage("source must be 'mic' or 'file'")
    if not isinstance(message.get("is_final"), bool):
        raise InvalidAudioMessage("is_final must be a boolean")

    seg_duration = _finite_number(message.get("seg_duration"), "seg_duration")
    seg_overlap = _finite_number(message.get("seg_overlap"), "seg_overlap")
    _finite_number(message.get("time_start"), "time_start")
    if not 0 < seg_duration <= MAX_SEGMENT_DURATION_SECONDS:
        raise InvalidAudioMessage(
            f"seg_duration must be > 0 and <= {MAX_SEGMENT_DURATION_SECONDS:g}"
        )
    if not 0 <= seg_overlap <= MAX_SEGMENT_OVERLAP_SECONDS:
        raise InvalidAudioMessage(
            f"seg_overlap must be >= 0 and <= {MAX_SEGMENT_OVERLAP_SECONDS:g}"
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

    context = message.get("context", "")
    if not isinstance(context, str) or len(context) > MAX_CONTEXT_CHARS:
        raise InvalidAudioMessage("context must be a bounded string")

    encoded = message.get("data")
    if not isinstance(encoded, str):
        raise InvalidAudioMessage("data must be a Base64 string")
    try:
        decoded = b64decode(encoded, validate=True)
    except (Base64Error, ValueError) as exc:
        raise InvalidAudioMessage("data is not valid Base64") from exc
    if len(decoded) > MAX_AUDIO_CHUNK_BYTES:
        raise InvalidAudioMessage(
            f"decoded audio chunk exceeds {MAX_AUDIO_CHUNK_BYTES} bytes"
        )
    if len(decoded) % AudioFormat.BYTES_PER_SAMPLE:
        raise InvalidAudioMessage("decoded float32 audio is not sample-aligned")
    return decoded


async def message_handler(websocket, message: dict, cache: AudioCache) -> None:
    """
    处理客户端发送的音频消息
    
    根据消息中的分段参数，将音频数据分段后提交到识别队列。
    """
    data = validate_audio_message(message)
    queue_in = Cosmic.queue_in

    global status_mic
    source = message['source']
    is_final = message['is_final']
    is_start = cache.task_id is None

    # 获取 id
    task_id = message['task_id']
    socket_id = str(websocket.id)
    context = message.get('context', '')
    cache.bind_stream(task_id, source)

    # 从消息中获取分段参数（由客户端决定）
    seg_duration = message['seg_duration']
    seg_overlap = message['seg_overlap']
    seg_threshold = seg_duration + seg_overlap * 2

    try:
        # Base64 已由 validate_audio_message 严格解码（float32, 16kHz, mono）
        cache.chunks += data
        cache.byte_count += len(data)

        if not is_final:
            # 打印状态消息
            if source == 'mic':
                status_mic.start()
            if source == 'file' and is_start:
                console.print('正在接收音频文件...')
                logger.info(f"开始接收音频文件，任务ID: {task_id}")

            # 若缓冲已达到分段阈值，将片段作为任务提交
            segment_bytes = AudioFormat.seconds_to_bytes(seg_duration + seg_overlap)
            stride_bytes = AudioFormat.seconds_to_bytes(seg_duration)
            
            while cache.duration >= seg_threshold:
                segment_data = cache.chunks[:segment_bytes]
                cache.chunks = cache.chunks[stride_bytes:]
                
                task = Task(
                    source=source,
                    data=segment_data,
                    offset=cache.offset,
                    task_id=task_id,
                    socket_id=socket_id,
                    overlap=seg_overlap,
                    is_final=False,
                    time_start=message['time_start'],
                    time_submit=time.time(),
                    context=context
                )
                cache.offset += seg_duration
                queue_in.put(task)
                logger.debug(
                    f"提交音频片段，任务ID: {task_id}, "
                    f"偏移: {cache.offset}s, 缓冲区: {len(cache.chunks)} bytes"
                )

        else:  # is_final
            # 打印状态消息
            if source == 'mic':
                status_mic.stop()
            elif source == 'file':
                print(f'音频文件接收完毕，时长 {cache.total_duration:.2f}s')
                logger.info(f"音频文件接收完毕，任务ID: {task_id}, 时长: {cache.total_duration:.2f}s")

            # 提交最终片段
            task = Task(
                source=source,
                data=cache.chunks,
                offset=cache.offset,
                task_id=task_id,
                socket_id=socket_id,
                overlap=seg_overlap,
                is_final=True,
                time_start=message['time_start'],
                time_submit=time.time(),
                context=context
            )
            queue_in.put(task)
            logger.debug(f"提交最终片段，任务ID: {task_id}, 数据大小: {len(cache.chunks)} bytes")

            # 重置缓冲区
            cache.reset()

    except Exception as e:
        logger.error(f"音频数据处理错误，任务ID: {task_id}: {e}", exc_info=True)
        raise


async def ws_recv(websocket) -> None:
    """
    WebSocket 接收主函数
    
    处理单个客户端连接，接收音频数据并分发处理。
    """
    global status_mic

    # 登记 socket 到连接池
    sockets = Cosmic.sockets
    sockets_id = Cosmic.sockets_id
    socket_id = str(websocket.id)
    sockets[socket_id] = websocket
    sockets_id.append(socket_id)
    console.print(f'接客了：{websocket}\n', style='yellow')
    logger.info(f"新客户端连接: {websocket}, ID: {socket_id}")

    # 创建音频缓冲区
    cache = AudioCache()

    # 接收并处理消息
    try:
        async for message in websocket:
            # 解析 JSON 消息
            message = json.loads(message)
            # 处理音频数据
            await message_handler(websocket, message, cache)

        console.print("ConnectionClosed...")
        logger.info(f"客户端正常关闭连接: {socket_id}")
        
    except (json.JSONDecodeError, InvalidAudioMessage) as e:
        logger.warning(f"拒绝无效 WebSocket 音频消息，客户端ID {socket_id}: {e}")
        try:
            await websocket.close(code=1008, reason="Invalid audio message")
        except websockets.ConnectionClosed:
            pass
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
        
        # 清理识别结果缓存，防止内存泄漏
        from util.server.server_recognize import clear_results_by_socket_id
        clear_results_by_socket_id(socket_id)
        
        logger.debug(f"客户端资源已清理: {socket_id}")
