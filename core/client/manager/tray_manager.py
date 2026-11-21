# coding: utf-8
import os
import subprocess
import sys
import threading
import time

from . import logger
from config_client import ClientConfig as Config
from fork_client.autostart import Autostart
from fork_client.instance import (
    RESTART_WAIT_ARGUMENT, SETTINGS_FROM_CLIENT_ARGUMENT, SETTINGS_RESTART_EXIT_CODE,
)


def _detached_popen_kwargs() -> dict:
    kwargs = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if os.name == "nt":
        flags = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
        if flags:
            kwargs["creationflags"] = flags
    else:
        kwargs["start_new_session"] = True
    return kwargs


def _launch_detached_process(cmd: list[str]) -> subprocess.Popen:
    return subprocess.Popen(cmd, **_detached_popen_kwargs())


def _open_with_default_app(target: str) -> None:
    if sys.platform == 'win32':
        os.startfile(target)
    elif sys.platform == 'darwin':
        _launch_detached_process(['open', target])
    else:
        _launch_detached_process(['xdg-open', target])


class TrayManager:
    """
    托盘管理器：负责系统托盘图标的初始化、菜单构建及回调处理。
    """
    def __init__(self, app):
        self.app = app
        self._settings_process = None
        self._settings_lock = threading.Lock()

    @property
    def state(self):
        return self.app.state

    def start(self):
        """初始化系统托盘图标"""
        if not Config.enable_tray:
            return

        try:
            from ..ui import enable_min_to_tray
        except ImportError as e:
            logger.warning(f"托盘模块导入失败，跳过托盘功能: {e}")
            self._restore_console()
            return

        # 获取图标路径
        icon_path = os.path.join(self.app.base_dir, 'assets', 'icon.ico')
        
        # 启用托盘
        started = enable_min_to_tray(
            'CapsWriter Client',
            icon_path,
            exit_callback=self.app.stop,
            restart_args=[RESTART_WAIT_ARGUMENT],
            more_options=[
                ('⚙️ 設定', self._open_settings),
                ('🚀 開機自動啟動', self._toggle_autostart, self._autostart_checked),
                ('📄 日誌', self._open_logs),
                ('📋 复制结果', self._copy_last_result),
                ('📁 日记', self._open_diary),
                ('📝 上下文', self._add_context),
                ('✨ 热词', self._add_hotword),
                ('🧹 清除记忆', self._clear_memory),
                ('♻️ 重开音频', self._restart_audio),
            ]
        )
        if not started:
            self._restore_console()
            return
        logger.info("托盘图标已启用")

    @staticmethod
    def _restore_console():
        # 開機自動啟動會先隱藏主控台；沒有托盤時必須還原，否則無法操作或退出。
        from fork_client.console_window import show_console
        show_console()

    def stop(self):
        """停止托盘图标"""
        if not Config.enable_tray:
            return
            
        try:
            from ..ui import stop_tray
            stop_tray()
            logger.info("TrayManager: 托盘图标已卸载")
        except Exception as e:
            logger.debug(f"TrayManager: 卸载托盘时发生错误: {e}")

    def _restart_audio(self):
        """重启音频流回调"""
        if hasattr(self.app, 'stream') and self.app.stream:
            self.app.stream.reopen()
            logger.info("用户请求重启音频")

    def _open_settings(self):
        """A separate process owns the Tk main loop; recording stays uninterrupted."""
        with self._settings_lock:
            if self._settings_process is not None and self._settings_process.poll() is None:
                logger.info("設定視窗已開啟，不重複啟動")
                return
            arguments = ['--settings', SETTINGS_FROM_CLIENT_ARGUMENT]
            if getattr(sys, "frozen", False):
                command = [sys.executable, *arguments]
            else:
                from pathlib import Path
                entrypoint = Path(__file__).resolve().parents[3] / 'start_client.py'
                command = [sys.executable, str(entrypoint), *arguments]
            try:
                process = _launch_detached_process(command)
            except OSError as exc:
                logger.error(f"無法開啟設定：{exc}")
                return
            self._settings_process = process
        threading.Thread(target=self._wait_settings, args=(process,), daemon=True, name="CapsWriterSettingsWatcher").start()

    def _wait_settings(self, process):
        """設定視窗選擇「儲存並重新啟動」時，以托盤相同流程重啟本 Client。"""
        from core.ui.tray import refresh_menu, request_restart
        if process.wait() != SETTINGS_RESTART_EXIT_CODE:
            # 設定視窗可能已切換開機自動啟動；同步托盤勾選狀態。
            refresh_menu()
            return
        logger.info("設定已儲存，依使用者要求重新啟動 Client")
        if not request_restart():
            logger.warning("托盤已停止，無法自動重新啟動；請手動重新啟動 Client")

    def _autostart_checked(self, _item=None) -> bool:
        try:
            return Autostart().status().enabled
        except OSError as exc:
            logger.debug(f"無法讀取開機自動啟動狀態: {exc}")
            return False

    def _toggle_autostart(self):
        from ..ui import toast
        enabled = not self._autostart_checked()
        try:
            status = Autostart().set_enabled(enabled)
        except OSError as exc:
            logger.warning(f"無法更新開機自動啟動: {exc}")
            toast(f"無法更新開機自動啟動：{exc}", duration=4000, bg="#8B1A1A")
            return
        logger.info(f"開機自動啟動已{'啟用' if status.enabled else '停用'}: {status.registered or '-'}")
        toast("已啟用開機自動啟動" if status.enabled else "已停用開機自動啟動", duration=2500, bg="#075077")

    def _open_logs(self):
        """开启日志資料夾，排查问题的唯一入口"""
        try:
            # 與 core.logger 使用同一個根目錄，托盤與設定視窗開啟的是同一個資料夾。
            from config_client import BASE_DIR
            folder = os.environ.get('CAPSWRITER_LOG_DIR') or os.path.join(BASE_DIR, 'logs')
            os.makedirs(folder, exist_ok=True)
            _open_with_default_app(str(folder))
        except Exception as error:
            logger.warning(f"无法打开日志文件夹: {error}")

    def _clear_memory(self):
        """清除 LLM 对话历史回调"""
        from ..ui import toast
        if self.app.llm:
            self.app.llm.clear_history()
            toast("清除成功：已清除所有角色的对话历史记录", duration=3000, bg="#075077")

    def _open_diary(self):
        """用系统默认方式打开当月日记文件夹回调"""
        try:
            base = self.app.diary.base_path
            folder = base / time.strftime('%Y') / time.strftime('%m')
            folder.mkdir(parents=True, exist_ok=True)
            _open_with_default_app(str(folder))
        except Exception as error:
            logger.warning(f"无法打开日记文件夹: {error}")

    def _add_hotword(self):
        """用系统默认方式打开热词文件回调"""
        
        try:
            from user_config import writable_data_file
            target = str(writable_data_file('hot.txt'))
            _open_with_default_app(target)
        except Exception as e:
            logger.warning(f"无法打开热词文件: {e}")

    def _add_context(self):
        """打开编辑上下文界面回调"""
        try:
            from ..ui import on_edit_context
            on_edit_context()
        except ImportError as e:
            logger.warning(f"无法导入上下文菜单处理器: {e}")

    def _copy_last_result(self):
        """复制最后一次识别结果到剪贴板回调"""
        text = self.state.last_output_text
        if text:
            from ..llm.llm_clipboard import copy_to_clipboard
            copy_to_clipboard(text)

    def _request_exit(self, icon=None, item=None):
        """托盘图标引用的退出回调"""
        logger.info("托盘退出: 用户点击退出菜单，准备清理资源并退出")
        self.app.stop()
