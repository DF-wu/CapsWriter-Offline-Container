# coding: utf-8
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int | None:
    selected_args = sys.argv[1:] if argv is None else argv
    if selected_args == ["--artifact-self-check"]:
        from artifact_self_check import run_artifact_self_check

        return run_artifact_self_check("client")

    from fork_client.settings import SettingsError, apply_overrides, settings_path
    from config_client import ClientConfig

    if selected_args == ["--settings"]:
        from fork_client.settings_ui import run_settings

        run_settings()
        return 0

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
    CapsWriterClient().start()

if __name__ == "__main__":
    status = main()
    if status is not None:
        raise SystemExit(status)
