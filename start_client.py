# coding: utf-8
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int | None:
    selected_args = sys.argv[1:] if argv is None else argv
    if selected_args == ["--artifact-self-check"]:
        from artifact_self_check import run_artifact_self_check

        return run_artifact_self_check("client")

    if selected_args == ["--settings"]:
        from fork_client.settings_ui import run_settings

        run_settings()
        return 0

    from fork_client.instance import (
        ClientInstance, RESTART_WAIT_ARGUMENT, RESTART_WAIT_SECONDS, report_status,
    )

    if selected_args == ["--status"]:
        return report_status()
    restart = selected_args == [RESTART_WAIT_ARGUMENT]
    if restart:
        selected_args = []
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


def _run_client(selected_args: list[str]) -> int | None:
    from fork_client.settings import SettingsError, apply_overrides, settings_path
    from config_client import ClientConfig

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
