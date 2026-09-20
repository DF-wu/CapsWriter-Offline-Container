# coding: utf-8
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int | None:
    selected_args = sys.argv[1:] if argv is None else argv
    if selected_args == ["--artifact-self-check"]:
        from artifact_self_check import run_artifact_self_check

        return run_artifact_self_check("client")

    from core.client import CapsWriterClient

    # 直接实例化并启动门面类即可；环境初始化职责已下放至 CapsWriterClient。
    CapsWriterClient().start()

if __name__ == "__main__":
    status = main()
    if status is not None:
        raise SystemExit(status)
