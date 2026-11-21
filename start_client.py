# coding: utf-8
from __future__ import annotations

import sys


def _tolerate_unencodable_output() -> None:
    """輸出導向舊式 code page（例如 cp1252）時，中文狀態訊息不可讓程式崩潰。

    凍結版不採用 PYTHONIOENCODING／PYTHONUTF8，因此在 import core 之前比照
    core/__init__.py 保留原編碼，只把無法編碼的字元改為跳脫序列。
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="backslashreplace")
            except (AttributeError, OSError, ValueError):
                pass


IMPORT_CONFIG_ARGUMENT = "--import-config"


def _import_config(old_root: str) -> int:
    """把舊版程式資料夾裡改過的設定、熱詞與角色匯入升級安全的 user/。"""
    from user_config import UserConfigError, import_legacy, user_dir

    try:
        report = import_legacy(old_root)
    except (UserConfigError, OSError) as exc:
        print(f"匯入設定失敗：{exc}", file=sys.stderr)
        return 2
    for line in report or ["沒有與預設值不同的設定需要匯入。"]:
        print(line)
    print(f"使用者設定資料夾：{user_dir()}")
    return 0


def main(argv: list[str] | None = None) -> int | None:
    selected_args = sys.argv[1:] if argv is None else argv
    _tolerate_unencodable_output()
    if selected_args == ["--artifact-self-check"]:
        from artifact_self_check import run_artifact_self_check

        return run_artifact_self_check("client")

    if selected_args == ["--settings"]:
        from fork_client.settings_ui import run_settings

        run_settings()
        return 0

    if len(selected_args) == 2 and selected_args[0] == IMPORT_CONFIG_ARGUMENT:
        return _import_config(selected_args[1])

    from fork_client.autostart import AUTOSTART_ARGUMENT
    from fork_client.instance import (
        ClientInstance, RESTART_WAIT_ARGUMENT, RESTART_WAIT_SECONDS,
        SETTINGS_FROM_CLIENT_ARGUMENT, SETTINGS_RESTART_EXIT_CODE, report_status,
    )

    if selected_args == ["--settings", SETTINGS_FROM_CLIENT_ARGUMENT]:
        from fork_client.settings_ui import open_settings

        # 由執行中 Client 的托盤開啟；以結束代碼要求該 Client 自行重新啟動。
        window = open_settings(from_client=True)
        return SETTINGS_RESTART_EXIT_CODE if window.restart_requested else 0

    if selected_args == ["--status"]:
        return report_status()
    restart = selected_args == [RESTART_WAIT_ARGUMENT]
    autostart = selected_args == [AUTOSTART_ARGUMENT]
    if restart or autostart:
        selected_args = []
    if autostart:
        _hide_console_for_autostart()
    # 在首次設定、logger、音訊與快捷鍵 import 之前取得鎖。檔案轉錄不占用麥克風鎖。
    instance = ClientInstance() if not selected_args else None
    try:
        if instance is not None:
            try:
                acquired = instance.acquire(RESTART_WAIT_SECONDS if restart else 0)
            except OSError as exc:
                print(f"無法取得 CapsWriter Client 執行鎖：{exc}", file=sys.stderr)
                return 2
            if not acquired:
                print(f"CapsWriter Client 已在執行，PID={instance.owner().get('pid', '未知')}；請使用現有托盤圖示。")
                return 1 if restart else 0
        return _run_client(selected_args)
    finally:
        if instance is not None:
            instance.release()


def _hide_console_for_autostart() -> None:
    """登入啟動時立即隱藏主控台；只在托盤可作為操作入口時才隱藏。"""
    from config_client import ClientConfig

    if ClientConfig.enable_tray:
        from fork_client.console_window import hide_console

        hide_console()


def _run_client(selected_args: list[str]) -> int | None:
    from fork_client.settings import SettingsError, apply_overrides, settings_path
    from config_client import ClientConfig
    from user_config import seed

    # 第一次啟動時建立升級安全的 user/（範本、熱詞、角色）；既有檔案永不覆寫。
    seed()

    # First-run Windows setup precedes imports that initialize hooks/audio.
    if (sys.platform == "win32" and not selected_args and not settings_path().exists()
            and ClientConfig.addr in ("127.0.0.1", "localhost", "::1")):
        from fork_client.settings_ui import run_settings

        if not run_settings():
            return 0
    try:
        apply_overrides(ClientConfig)
    except SettingsError as exc:
        print(f"Client 設定無效：{exc}\n請執行 start_client.py --settings 修正設定。", file=sys.stderr)
        return 2

    from core.client import CapsWriterClient

    # 直接实例化并启动门面类即可；环境初始化职责已下放至 CapsWriterClient。
    CapsWriterClient().start(selected_args)

if __name__ == "__main__":
    status = main()
    if status is not None:
        raise SystemExit(status)
