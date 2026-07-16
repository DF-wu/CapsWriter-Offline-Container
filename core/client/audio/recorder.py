# coding: utf-8
"""
音频录制模块

提供 AudioRecorder 类用于管理录音会话，包括开始录音、
发送音频数据到服务端、结束录音等功能。
"""

from __future__ import annotations

import asyncio
import base64
import uuid
from typing import TYPE_CHECKING, Optional

import numpy as np
import websockets

from config_client import ClientConfig as Config
from core.client.state import console
from core.client.audio.file_manager import AudioFileManager
from core.client.connection import WebSocketManager
from core.protocol import AudioMessage
from . import logger

if TYPE_CHECKING:
    from core.client.state import ClientState
    from core.client.app import CapsWriterClient

# 日志记录器
CLIENT_RECORDER_FAILURE_DRAIN_SECONDS = 30.0


class AudioRecorder:
    """
    音频录制器
    
    管理一次完整的录音会话，包括：
    - 从音频流接收数据
    - 可选地保存到本地文件
    - 将音频数据发送到识别服务端
    """
    
    def __init__(self, app: CapsWriterClient):
        """
        初始化录制器
        
        Args:
            app: 客户端 App 实例
        """
        self.app = app
        self.task_id: Optional[str] = None
        self._file_manager: Optional[AudioFileManager] = None
        self._file_manager_finished = False
        self._start_time: float = 0.0
        self._duration: float = 0.0
        self._cache: list = []
        self._stream_started = False

    @property
    def state(self) -> ClientState:
        """快捷访问状态单例"""
        return self.app.state

    @property
    def _ws_manager(self) -> WebSocketManager:
        """快捷访问桥接到 app.ws"""
        return self.app.ws
    
    async def _send_message(self, message: AudioMessage) -> bool:
        """发送消息到服务端"""
        if not self._ws_manager.is_connected:
            if message.is_final:
                self.state.pop_audio_file(message.task_id)
                console.print('    服务端未连接，无法发送\n')
                logger.warning("服务端未连接，无法发送音频数据")
            return False
        
        try:
            # Await every send in stream order. Backpressure is intentional:
            # callback ingress has its own hard cap and reports overflow rather
            # than retaining an unbounded set of base64-bearing Tasks.
            success = await self._ws_manager.send(message)
        except Exception as error:
            logger.error(f"发送录音片段失败: {error}")
            success = False
        if not success and message.is_final:
            self.state.pop_audio_file(message.task_id)
            # 具体错误日志由 WebSocketManager 记录
        return success

    async def _drain_until_finish(self) -> None:
        """Consume bounded ingress after a terminal recorder error.

        ShortcutTask.finish enqueues the finish marker asynchronously. Keeping
        one consumer alive for a bounded interval prevents that control put
        from becoming an orphan when a strict server stream is aborted.
        """

        loop = asyncio.get_running_loop()
        deadline = loop.time() + CLIENT_RECORDER_FAILURE_DRAIN_SECONDS
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return
            try:
                task = await asyncio.wait_for(
                    self.state.queue_in.get(),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                return
            try:
                if isinstance(task, dict) and task.get('type') == 'data':
                    self.state.release_audio_chunk()
                if isinstance(task, dict) and task.get('type') == 'finish':
                    return
            finally:
                self.state.queue_in.task_done()

    def _discard_pending_audio(self) -> None:
        """Release every callback reservation after cancellation/failure."""

        while True:
            try:
                task = self.state.queue_in.get_nowait()
            except asyncio.QueueEmpty:
                break
            try:
                if isinstance(task, dict) and task.get('type') == 'data':
                    self.state.release_audio_chunk()
            finally:
                self.state.queue_in.task_done()
        self._cache.clear()

    def _finish_audio_file(self) -> None:
        """Finalize the current writer at most once on every exit path."""

        if self._file_manager is None or self._file_manager_finished:
            return
        self._file_manager_finished = True
        self._file_manager.finish()
        logger.debug("完成音频文件写入")

    def _discard_audio_file_registration(self) -> None:
        """Remove a file that cannot receive a final recognition result."""

        if self.task_id:
            self.state.pop_audio_file(self.task_id)

    async def _abort_transmission(self, reason: str) -> None:
        logger.error(reason)
        console.print(f'    [bold red]{reason}[/bold red]')
        self._discard_audio_file_registration()
        try:
            # Disconnecting is the protocol-level cancellation signal. It
            # ensures the server purges any partial stream before reconnecting.
            await self._ws_manager.close()
        except Exception as error:
            logger.debug(f"关闭失败的录音连接时发生错误: {error}")
        finally:
            self._stream_started = False

    async def _cleanup_cancelled_recording(self) -> None:
        """Complete protocol and queue cleanup after recorder cancellation."""

        try:
            if self._stream_started:
                await self._abort_transmission("录音已取消")
        finally:
            self._discard_pending_audio()

    async def _wait_for_cancellation_cleanup(self) -> None:
        """Wait for cleanup despite repeated cancellation requests."""

        cleanup_task = asyncio.create_task(self._cleanup_cancelled_recording())
        cancelled_again = False
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                cancelled_again = True
            except Exception:
                # Preserve the recorder's cancellation; collect/log the cleanup
                # error below after the cleanup task has completed.
                break

        try:
            cleanup_task.result()
        except Exception as error:
            logger.error(f"取消录音后的资源清理失败: {error}", exc_info=True)

        if cancelled_again:
            raise asyncio.CancelledError()
    
    async def record_and_send(self) -> None:
        """
        录音并发送数据
        
        从队列中读取音频数据，保存到文件（如果启用），
        并发送到服务端进行识别。
        """
        try:
            # 生成唯一任务 ID
            self.task_id = str(uuid.uuid4())
            logger.debug(f"创建录音任务，任务ID: {self.task_id}")
            
            self._start_time = 0.0
            self._duration = 0.0
            self._cache = []
            self._stream_started = False
            self._file_manager = None
            self._file_manager_finished = False
            
            # 音频文件管理
            file_path = None
            transmission_aborted = False
            if Config.save_audio:
                self._file_manager = AudioFileManager()
            
            # 从队列读取数据
            while True:
                task = await self.state.queue_in.get()
                try:
                    if task['type'] == 'data':
                        self.state.release_audio_chunk()

                    if (
                        self.state.consume_audio_input_overflow()
                        and not transmission_aborted
                    ):
                        transmission_aborted = True
                        self._cache.clear()
                        await self._abort_transmission(
                            "录音输入超过本机缓冲上限；本次录音已取消，请重试"
                        )

                    if task['type'] == 'begin':
                        self._start_time = task['time']
                        logger.debug(f"录音开始，时间戳: {self._start_time}")

                    elif task['type'] == 'data':
                        if transmission_aborted:
                            continue
                    # 在阈值之前积攒音频数据
                        if task['time'] - self._start_time < Config.threshold:
                            self._cache.append(task['data'])
                            continue

                        # 创建音频文件
                        if Config.save_audio and self._file_manager and file_path is None:
                            file_path, _ = self._file_manager.create(
                                task['data'].shape[1],
                                self._start_time
                            )
                            self.state.register_audio_file(self.task_id, file_path)
                            logger.debug(f"创建音频文件: {file_path}")

                        # 获取音频数据
                        if self._cache:
                            data = np.concatenate(self._cache)
                            self._cache.clear()
                        else:
                            data = task['data']

                        # 保存音频至本地文件
                        self._duration += len(data) / 48000
                        if Config.save_audio and self._file_manager:
                            self._file_manager.write(data)

                        # Awaiting preserves wire order and bounds retained audio.
                        message = AudioMessage(
                            task_id=self.task_id,
                            source='mic',
                            data=base64.b64encode(
                                np.mean(data[::3], axis=1).tobytes()
                            ).decode('utf-8'),
                            is_final=False,
                            time_start=self._start_time,
                            seg_duration=Config.mic_seg_duration,
                            seg_overlap=Config.mic_seg_overlap,
                            context=Config.context,
                            language=Config.language,
                        )
                        # Treat an in-flight first send as server-visible.  A
                        # cancellation can arrive after bytes leave this process
                        # but before send() reports success, so cleanup must close
                        # the connection as the protocol cancellation signal.
                        self._stream_started = True
                        if not await self._send_message(message):
                            transmission_aborted = True
                            await self._abort_transmission(
                                "录音资料传送失败；本次录音已取消"
                            )

                    elif task['type'] == 'finish':
                        # 如果有缓存的数据未发送，先发送缓存，并严格等候完成。
                        if self._cache and not transmission_aborted:
                            data = np.concatenate(self._cache)
                            self._cache.clear()

                            self._duration += len(data) / 48000
                            if Config.save_audio and self._file_manager:
                                self._file_manager.write(data)

                            message = AudioMessage(
                                task_id=self.task_id,
                                source='mic',
                                data=base64.b64encode(
                                    np.mean(data[::3], axis=1).tobytes()
                                ).decode('utf-8'),
                                is_final=False,
                                time_start=self._start_time,
                                seg_duration=Config.mic_seg_duration,
                                seg_overlap=Config.mic_seg_overlap,
                                context=Config.context,
                                language=Config.language,
                            )
                            self._stream_started = True
                            if not await self._send_message(message):
                                transmission_aborted = True
                                await self._abort_transmission(
                                    "录音资料传送失败；本次录音已取消"
                                )

                        # 完成写入本地文件
                        self._finish_audio_file()

                        console.print(f'任务标识：{self.task_id}')
                        console.print(f'    录音时长：{self._duration:.2f}s')
                        logger.info(
                            f"录音任务完成，任务ID: {self.task_id}, "
                            f"时长: {self._duration:.2f}s, "
                            f"已取消: {transmission_aborted}"
                        )

                        if not transmission_aborted:
                            # The final marker is sent only after all data sends.
                            message = AudioMessage(
                                task_id=self.task_id,
                                source='mic',
                                data='',
                                is_final=True,
                                time_start=self._start_time,
                                seg_duration=Config.mic_seg_duration,
                                seg_overlap=Config.mic_seg_overlap,
                                context=Config.context,
                                language=Config.language,
                            )
                            self._stream_started = True
                            if await self._send_message(message):
                                self._stream_started = False
                            else:
                                transmission_aborted = True
                                await self._abort_transmission(
                                    "录音结束标志传送失败；本次录音已取消"
                                )
                        break
                finally:
                    self.state.queue_in.task_done()

        except asyncio.CancelledError:
            self._finish_audio_file()
            self._discard_audio_file_registration()
            await self._wait_for_cancellation_cleanup()
            raise
        except Exception as e:
            logger.error(f"录音任务错误: {e}", exc_info=True)
            self._finish_audio_file()
            self._discard_audio_file_registration()
            if self._stream_started:
                await self._abort_transmission(
                    "录音处理发生错误；服务器端串流已取消"
                )
            # If the failing item was already the finish marker there will be
            # no later control event; otherwise keep a bounded consumer alive.
            if not (
                'task' in locals()
                and isinstance(task, dict)
                and task.get('type') == 'finish'
            ):
                await self._drain_until_finish()
            self._discard_pending_audio()
        finally:
            self._finish_audio_file()
    
    def get_file_manager(self) -> Optional[AudioFileManager]:
        """获取当前的文件管理器"""
        return self._file_manager
