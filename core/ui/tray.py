# coding: utf-8
"""
托盘图标模块

提供最小化到系统托盘的功能。
仅在 Windows 平台有效。

功能：
- 保留控制台窗口的关闭按钮
- 最小化时自动隐藏到托盘
- 双击托盘图标显示/隐藏窗口
- 托盘菜单退出程序

注意：pystray 在 Linux 无 GUI 环境下无法导入，因此采用延迟导入。
"""

import os
import sys
import time
import threading
import platform
import subprocess
from typing import Optional
from . import logger, set_ui_logger

# 退出回调函数（由主程序传入）
_exit_callback = None

# 是否可用（在 enable_min_to_tray 时检测）
_tray_available = None

def _set_exit_callback(callback):
    """设置退出回调函数"""
    global _exit_callback
    _exit_callback = callback

def _get_exit_callback():
    return _exit_callback


def _check_tray_available() -> bool:
    """
    检查托盘功能是否可用
    
    Returns:
        bool: 是否可用
    """
    global _tray_available
    
    if _tray_available is not None:
        return _tray_available
    
    # 非 Windows 系统不支持
    if platform.system() != 'Windows':
        _tray_available = False
        return False
    
    # 尝试导入 pystray
    try:
        import pystray
        from PIL import Image
        _tray_available = True
    except ImportError as e:
        logger.warning(f"托盘功能不可用: {e}")
        _tray_available = False
    except Exception as e:
        logger.warning(f"托盘功能检测失败: {e}")
        _tray_available = False
    
    return _tray_available


# Windows API（延迟初始化）
_win_api_initialized = False
user32 = None
kernel32 = None

SW_HIDE = 0
SW_RESTORE = 9
SW_SHOW = 5
SC_CLOSE = 0xF060
MF_BYCOMMAND = 0x00000000
GA_ROOT = 3


def _init_win_api():
    """初始化 Windows API"""
    global _win_api_initialized, user32, kernel32
    
    if _win_api_initialized:
        return
    
    if platform.system() != 'Windows':
        return
    
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        # HWND/HMENU 是指標；ctypes 預設的 int 會在 64 位元 Windows 截斷它們。
        kernel32.GetConsoleWindow.restype = wintypes.HWND
        kernel32.SetConsoleTitleW.argtypes = [wintypes.LPCWSTR]
        user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        user32.GetAncestor.restype = wintypes.HWND
        user32.GetSystemMenu.argtypes = [wintypes.HWND, wintypes.BOOL]
        user32.GetSystemMenu.restype = wintypes.HMENU
        user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.SetForegroundWindow.argtypes = [wintypes.HWND]
        user32.IsIconic.argtypes = [wintypes.HWND]
        user32.IsWindowVisible.argtypes = [wintypes.HWND]
        _win_api_initialized = True
    except Exception as e:
        logger.warning(f"Windows API 初始化失败: {e}")


# 全局变量
_tray_instance: Optional['_TraySystem'] = None
_lock = threading.Lock()


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


def _get_console_hwnd():
    _init_win_api()
    hwnd = kernel32.GetConsoleWindow()
    if hwnd:
        # 在 Windows Terminal 中，GetConsoleWindow 返回的是内部窗口句柄。
        # 为了能让“退出按钮不可用”以及“从任务栏消失”生效，我们需要操作最外层的顶层窗口。
        # 对于普通 CMD，GetAncestor(hwnd, GA_ROOT) 依然返回 hwnd 本身。
        root_hwnd = user32.GetAncestor(hwnd, GA_ROOT)
        if root_hwnd:
            return root_hwnd
    return hwnd


def _disable_close_button(hwnd: int) -> None:
    """禁用窗口的关闭按钮"""
    if user32 is None:
        return
    h_menu = user32.GetSystemMenu(hwnd, False)
    if h_menu:
        user32.DeleteMenu(h_menu, SC_CLOSE, MF_BYCOMMAND)


def _enable_close_button(hwnd: int) -> None:
    """恢复窗口的关闭按钮"""
    if user32 is None:
        return
    user32.GetSystemMenu(hwnd, True)


def _is_window_minimized(hwnd: int) -> bool:
    """检查窗口是否最小化"""
    if user32 is None:
        return False
    return user32.IsIconic(hwnd) != 0


def _is_window_visible(hwnd: int) -> bool:
    """检查窗口是否可见"""
    if user32 is None:
        return False
    return user32.IsWindowVisible(hwnd) != 0


def _create_icon(icon_path: Optional[str] = None):
    """
    创建托盘图标
    
    优先从指定路径加载图标文件，如果不存在则动态生成。
    
    Args:
        icon_path: 图标文件路径
        
    Returns:
        PIL Image 对象
    """
    from PIL import Image, ImageDraw
    
    # 如果图标文件存在，直接加载
    if icon_path and os.path.exists(icon_path):
        try:
            image = Image.open(icon_path)
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            return image.resize((64, 64), Image.Resampling.LANCZOS)
        except Exception:
            pass  # 加载失败则使用动态生成

    # 动态生成图标
    size = 64
    scale = 4
    real_size = size * scale

    image = Image.new('RGBA', (real_size, real_size), (0, 0, 0, 0))
    dc = ImageDraw.Draw(image)

    blue = (55, 118, 171)
    yellow = (255, 211, 67)
    white = (255, 255, 255)

    # 蓝色圆角背景
    m = 2 * scale
    dc.rounded_rectangle(
        [m, m, real_size - m, real_size - m],
        radius=real_size // 4,
        fill=blue
    )

    # 黄色圆圈
    center = real_size // 2
    r = real_size // 3.5
    dc.ellipse([center - r, center - r, center + r, center + r], fill=yellow)

    # 白色圆点
    r2 = r // 2
    dc.ellipse([center - r2, center - r2, center + r2, center + r2], fill=white)

    return image.resize((size, size), Image.Resampling.LANCZOS)


class _TraySystem:
    """托盘系统内部类"""
    
    def __init__(self, name: Optional[str] = None, icon_path: Optional[str] = None, more_options: list = None, restart_args: Optional[list[str]] = None):
        # 延迟导入 pystray
        import pystray
        from pystray import MenuItem as item
        
        self.hwnd = _get_console_hwnd()
        self.should_exit = False
        self.restart_args = restart_args
        self._tray_thread = None
        self._monitor_thread = None
        self.title = name if name else (os.path.basename(sys.argv[0]) or "Console App")

        # 托盤不可成為唯一的關閉入口，保留視窗的 × 並顯示可辨認的名稱。
        if self.hwnd:
            _enable_close_button(self.hwnd)
            kernel32.SetConsoleTitleW(self.title)

        # 定义菜单
        menu_items = [
            item(f"{self.title}", lambda: None, enabled=False),
            item('👁️ 显示/隐藏', self.toggle_window, default=True),
        ]

        # 添加额外选项
        if more_options:
            for opt_name, opt_func in more_options:
                menu_items.append(item(opt_name, opt_func))

        menu_items.append(item('🔄 重启', self.on_restart))
        menu_items.append(item('❌ 退出', self.on_exit))

        self.icon = pystray.Icon(
            "console_tray",
            _create_icon(icon_path),
            title=f"{self.title}",
            menu=tuple(menu_items)
        )

    def toggle_window(self) -> None:
        """切换窗口显示状态"""
        if not self.hwnd or user32 is None:
            return

        if _is_window_visible(self.hwnd):
            user32.ShowWindow(self.hwnd, SW_HIDE)
        else:
            user32.ShowWindow(self.hwnd, SW_RESTORE)
            user32.SetForegroundWindow(self.hwnd)

    def monitor_loop(self) -> None:
        """监控线程：检测最小化操作"""
        while not self.should_exit:
            if self.hwnd and user32:
                # 窗口可见且最小化 -> 隐藏到托盘
                if _is_window_visible(self.hwnd) and _is_window_minimized(self.hwnd):
                    user32.ShowWindow(self.hwnd, SW_HIDE)
            time.sleep(0.2)

    def on_restart(self, icon, item) -> None:
        """新客戶端等待舊實例釋放鎖；退出走與選單相同的完整流程。"""
        logger.info("托盘重启: 用户点击重启菜单，准备重启程序")
        try:
            if getattr(sys, 'frozen', False):
                cmd = [sys.executable]
            else:
                cmd = [sys.executable, sys.argv[0]]
            cmd += self.restart_args if self.restart_args is not None else sys.argv[1:]
            _launch_detached_process(cmd)
        except Exception as e:
            logger.error(f"重启失败: {e}")
            return

        self.on_exit(icon, item)

    def on_exit(self, icon, item) -> None:
        """托盘退出处理"""
        exit_callback = _get_exit_callback()

        logger.info("托盘退出: 用户点击退出菜单，准备清理资源并退出")

        # 1. 设置退出标志，停止监控循环
        self.should_exit = True
        logger.debug("已设置托盘退出标志")

        # 2. 恢复窗口关闭按钮并显示窗口
        if self.hwnd and user32:
            _enable_close_button(self.hwnd)
            user32.ShowWindow(self.hwnd, SW_RESTORE)
            logger.debug("已恢复窗口显示")

        # 3. 调用退出回调函数，请求主程序退出
        if exit_callback:
            try:
                logger.debug("正在调用退出回调函数...")
                exit_callback()
                logger.info("退出回调函数已调用")
            except Exception as e:
                logger.error(f"调用退出回调函数时发生错误: {e}")



        self.stop()

    def stop(self) -> None:
        """停止圖示與監控；可由托盤執行緒或主執行緒重複呼叫。"""
        self.should_exit = True
        if self.hwnd and user32:
            _enable_close_button(self.hwnd)
        try:
            self.icon.stop()
        except Exception as e:
            logger.warning(f"停止托盘图标时发生错误: {e}")
        for thread in (self._tray_thread, self._monitor_thread):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=2)

    def _run_icon(self) -> None:
        try:
            self.icon.run(setup=self._on_ready)
        except Exception as e:
            logger.error(f"托盤啟動失敗: {e}")
        finally:
            self.should_exit = True
            # 後端失敗時恢復視窗，避免留下沒有可操作入口的背景程序。
            if self.hwnd and user32:
                user32.ShowWindow(self.hwnd, SW_RESTORE)

    def _on_ready(self, icon) -> None:
        icon.visible = True
        if self.hwnd and user32 and not self.should_exit:
            user32.ShowWindow(self.hwnd, SW_HIDE)

    def start(self) -> None:
        """启动托盘系统"""
        # 托盘图标线程
        self._tray_thread = threading.Thread(target=self._run_icon, daemon=True, name="CapsWriterTray")
        self._tray_thread.start()

        # 状态监控线程
        self._monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True, name="CapsWriterTrayMonitor")
        self._monitor_thread.start()


def enable_min_to_tray(name: Optional[str] = None, icon_path: Optional[str] = None, exit_callback=None, more_options: list = None, restart_args: Optional[list[str]] = None) -> None:
    """
    启用最小化到托盘功能

    沒有主控台視窗（如 detached 重啟）仍建立托盤，保留退出入口。
    如果在 Linux 等无 GUI 环境下运行，也会跳过。

    Args:
        name: 托盘图标显示的名称，默认使用程序名称
        icon_path: 图标文件路径，默认动态生成
        exit_callback: 退出回调函数，当用户点击托盘退出菜单时调用
        more_options: 额外菜单项列表，格式为 [(名称, 回调函数), ...]
    """
    global _tray_instance

    # 设置退出回调函数
    if exit_callback is not None:
        _set_exit_callback(exit_callback)

    # 检查托盘功能是否可用
    if not _check_tray_available():
        logger.info("托盘功能不可用，跳过启用")
        return

    # DPI 感知设置
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass

    with _lock:
        if _tray_instance is not None:
            return  # 已启动

        _tray_instance = _TraySystem(name, icon_path, more_options, restart_args)
        _tray_instance.start()



def stop_tray() -> None:
    """停止托盘图标"""
    global _tray_instance
    with _lock:
        instance, _tray_instance = _tray_instance, None
    if instance is not None:
        instance.stop()


if __name__ == "__main__":
    enable_min_to_tray()
    print("程序运行中... 你可以双击托盘图标隐藏我。")
    while True:
        time.sleep(1)
