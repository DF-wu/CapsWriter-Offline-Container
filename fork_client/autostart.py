"""Windows 登入時自動啟動 Client；使用者層級 HKCU Run，不需系統管理員或排程工作。

狀態以登錄檔為唯一來源，不另存於 client-settings.json，避免與「工作管理員 →
啟動應用程式」的切換互相矛盾。Import 本模組不會初始化音訊或快捷鍵。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys


AUTOSTART_ARGUMENT = "--autostart"
VALUE_NAME = "CapsWriter Client"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
# 工作管理員／設定 App 停用啟動項時寫入此處；首位元組為奇數（0x03）代表停用。
APPROVED_KEY = r"Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run"

DISABLED = "disabled"
ENABLED = "enabled"
BLOCKED = "blocked"     # Run 值存在，但被工作管理員停用
STALE = "stale"         # Run 值指向其他位置（例如資料夾已搬移）
UNSUPPORTED = "unsupported"


def supported() -> bool:
    return sys.platform == "win32"


def launch_command() -> list[str]:
    """登入時執行的命令；PyInstaller 版與原始碼版分別處理。"""
    if getattr(sys, "frozen", False):
        return [str(Path(sys.executable).resolve()), AUTOSTART_ARGUMENT]
    entrypoint = Path(__file__).resolve().parents[1] / "start_client.py"
    return [str(Path(sys.executable).resolve()), str(entrypoint), AUTOSTART_ARGUMENT]


def command_line(command: list[str] | None = None) -> str:
    return subprocess.list2cmdline(launch_command() if command is None else command)


class _WinRegistry:
    """HKCU 讀寫的最小介面；測試可替換成其他 key 或假物件。"""

    def __init__(self):
        import winreg

        self.winreg = winreg

    def get(self, key: str, name: str):
        try:
            with self.winreg.OpenKey(self.winreg.HKEY_CURRENT_USER, key) as handle:
                return self.winreg.QueryValueEx(handle, name)[0]
        except FileNotFoundError:
            return None

    def set(self, key: str, name: str, value: str) -> None:
        with self.winreg.CreateKeyEx(self.winreg.HKEY_CURRENT_USER, key, 0, self.winreg.KEY_SET_VALUE) as handle:
            self.winreg.SetValueEx(handle, name, 0, self.winreg.REG_SZ, value)

    def delete(self, key: str, name: str) -> None:
        try:
            with self.winreg.OpenKey(self.winreg.HKEY_CURRENT_USER, key, 0, self.winreg.KEY_SET_VALUE) as handle:
                self.winreg.DeleteValue(handle, name)
        except FileNotFoundError:
            pass


@dataclass(frozen=True)
class AutostartStatus:
    state: str
    registered: str | None = None
    expected: str | None = None

    @property
    def enabled(self) -> bool:
        """只有確實會在下次登入執行目前這份 Client 才算啟用。"""
        return self.state == ENABLED

    def describe(self) -> str:
        if self.state == ENABLED:
            return "已啟用：登入 Windows 後會在背景啟動 Client，並常駐於系統匣。"
        if self.state == BLOCKED:
            return "已登錄，但在「工作管理員 → 啟動應用程式」中被停用；勾選並套用即可重新啟用。"
        if self.state == STALE:
            return f"已登錄的路徑與目前程式不同（可能已搬移資料夾）：{self.registered}。勾選並套用即可更新。"
        if self.state == UNSUPPORTED:
            return "僅 Windows 支援開機自動啟動。"
        return "未啟用：需要時請手動執行 start_client.exe。"


class Autostart:
    def __init__(self, registry=None, run_key: str = RUN_KEY, approved_key: str = APPROVED_KEY,
                 name: str = VALUE_NAME, command: list[str] | None = None):
        self._registry = registry
        self.run_key = run_key
        self.approved_key = approved_key
        self.name = name
        self.command = command

    @property
    def registry(self):
        if self._registry is None:
            self._registry = _WinRegistry()
        return self._registry

    def expected(self) -> str:
        return command_line(self.command)

    def status(self) -> AutostartStatus:
        if self._registry is None and not supported():
            return AutostartStatus(UNSUPPORTED)
        expected = self.expected()
        registered = self.registry.get(self.run_key, self.name)
        if not isinstance(registered, str) or not registered:
            return AutostartStatus(DISABLED, expected=expected)
        if registered.strip().casefold() != expected.casefold():
            return AutostartStatus(STALE, registered, expected)
        approved = self.registry.get(self.approved_key, self.name)
        if isinstance(approved, (bytes, bytearray)) and approved and approved[0] & 1:
            return AutostartStatus(BLOCKED, registered, expected)
        return AutostartStatus(ENABLED, registered, expected)

    def set_enabled(self, enabled: bool) -> AutostartStatus:
        if self._registry is None and not supported():
            raise OSError("僅 Windows 支援開機自動啟動")
        if enabled:
            self.registry.set(self.run_key, self.name, self.expected())
        else:
            self.registry.delete(self.run_key, self.name)
        # 明確在此切換時以本次選擇為準，清除工作管理員留下的停用標記。
        self.registry.delete(self.approved_key, self.name)
        return self.status()
