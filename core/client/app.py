# coding: utf-8
"""
CapsWriter Offline 客户端主程序门面类 (Facade)

采用外观模式统一管理音频流 (AudioStreamManager)、
识别结果处理 (ResultProcessor) 和快捷键管理 (ShortcutManager)。
"""

import os
import sys
import asyncio
import threading
from pathlib import Path

from .state import ClientState
from . import logger
from config_client import ClientConfig as Config, __version__
from core.tools.signal_handler import register_signal
from .state import console
from .connection import WebSocketManager
from typing import TYPE_CHECKING, Optional
from .manager import (
    TrayManager,
    MicRunner, FileRunner
)
from .audio.stream import AudioStreamManager
from .shortcut.shortcut_manager import ShortcutManager
from .shortcut.shortcut_config import Shortcut

from .udp.udp_control import UDPController

from .hotword.manager import HotwordManager
from .llm.llm_handler import LLMHandler
from .output.text_output import TextOutput
from .diary.diary_writer import DiaryWriter
from core.tools.empty_working_set import empty_current_working_set
from platform import system



class CapsWriterClient:
    """
    CapsWriter 客户端门面类
    
    管理的外部接口简洁：start()。
    """
    def __init__(self):
        # 确保正确的工作目录
        self.base_dir = Path(__file__).parents[2]
        os.chdir(self.base_dir)
            
        # 初始化事件循环
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
            
        # 初始化状态容器
        self.state = ClientState(app=self)
        self._result_processor = None
        self._runner_task = None
        self._shutdown_future = None
        self._shutdown_lock = threading.Lock()
        self._stop_requested = threading.Event()

        # 初始化热词管理器
        self.hotword = HotwordManager(
            hotword_files=None,
            threshold=Config.hot_thresh,
            similar_threshold=Config.hot_similar
        )

        # 4. 初始化 LLM 润色系统
        self.llm = LLMHandler(app=self)
        
        self.output = TextOutput()
        self.diary = DiaryWriter(base_path=self.base_dir)

        # 初始化各管理器
        self.ws = WebSocketManager(self)
        self.tray = TrayManager(self)

        # 实例化硬件资源管理组件
        self.stream = AudioStreamManager(self)
        self.shortcut = ShortcutManager(self, [Shortcut(**sc) for sc in Config.shortcuts])
        self.udp = UDPController(self.shortcut)

        # 内存清理
        empty_current_working_set()

    def stop(self):
        """
        统一释放所有资源（清理顺序：硬件 -> 托盘 -> WebSocket -> State）
        """
        if self._stop_requested.is_set():
            return
        self._stop_requested.set()
        # 与 start() 的 finally 串行，确保它不会在 shutdown future 发布前返回。
        with self._shutdown_lock:
            self._schedule_stop()

    def _schedule_stop(self) -> None:
        """停止同步组件，并发布由事件循环执行的最终清理。"""
        logger.info("正在执行 CapsWriterClient 资源释放...")

        processor = self._result_processor
        if processor is not None:
            # 退出事件必须先于 close，避免接收循环抢先进入下一次重连。
            processor.request_exit()

        # 1. 停止核心运行组件
        self.udp.stop()
        self.shortcut.stop()
        self.stream.stop()

        # 2. 托盘资源
        self.tray.stop()

        # 3. 关闭监控
        self.hotword.stop()
        self.llm.stop()

        # 4. WebSocket 与 receiver 必须由所属事件循环依序关闭。
        if self.loop.is_running():
            self._shutdown_future = asyncio.run_coroutine_threadsafe(
                self._finish_stop(processor),
                self.loop,
            )
        else:
            self.state.reset()
            self.loop.stop()

        logger.info("资源释放完成")
        console.print('[green4]再见！')

    async def _finish_stop(self, processor) -> None:
        runner_task = self._runner_task
        try:
            if processor is not None:
                await processor.stop()
            else:
                if runner_task is not None and not runner_task.done():
                    runner_task.cancel()
                await self.ws.close()
        except Exception as error:
            logger.warning(f"关闭客户端连接时发生错误: {error}")
            # close 失败后不再让 reset 调度另一个无法等待的关闭协程。
            self.state.websocket = None
        if runner_task is not None and runner_task is not asyncio.current_task():
            await asyncio.gather(runner_task, return_exceptions=True)
        try:
            self.state.reset()
        except Exception as error:
            logger.warning(f"重置状态时发生错误: {error}")

    def _drain_shutdown(self) -> None:
        """在 start() 返回前完成托盘线程提交的异步清理。"""
        shutdown_lock = getattr(self, "_shutdown_lock", None)
        if shutdown_lock is None:
            return
        with shutdown_lock:
            shutdown_future = getattr(self, "_shutdown_future", None)
        if shutdown_future is None:
            return
        try:
            if not shutdown_future.done():
                self.loop.run_until_complete(
                    asyncio.wrap_future(shutdown_future, loop=self.loop)
                )
            else:
                shutdown_future.result()
        except Exception as error:
            logger.warning(f"等待客户端异步清理时发生错误: {error}")


    def start(self):
        """
        启动客户端 (唯一入口)
        
        自动根据命令行参数识别模式。内部管理异步循环。
        """

        # 注册退出函数
        register_signal(self.stop)

        # FileRunner reports invalid inputs; supplied paths still select file mode.
        files = [Path(f) for f in sys.argv[1:]]

        if files:
            # 文件转录模式
            runner = FileRunner(self, files)
        else:
            # 麦克风实时模式
            runner = MicRunner(self)
        
        self._runner_task = self.loop.create_task(
            runner.run(),
            name="capswriter-client-runner",
        )
        try:
            self.loop.run_until_complete(self._runner_task)
        except asyncio.CancelledError:
            if not self._stop_requested.is_set():
                raise
        except RuntimeError:
            ...
        finally:
            self._drain_shutdown()
            self._runner_task = None
