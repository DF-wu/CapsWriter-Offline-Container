# coding: utf-8
"""
start_server_docker.py — fork-only server entrypoint

容器內 ENTRYPOINT 透過 docker/server/entrypoint.sh 呼叫此檔。
與上游 start_server.py 並存, 上游 start_server.py 仍可在裸機跑 (不啟用 HTTP API)。

工作流:
1. fork_server.bootstrap.apply_env_config()
   — 必須在 import core.server.* 之前, 把 env 變數塞進 ServerConfig 等 class 屬性,
     否則 core/server/__init__.py 的 setup_logger() 會用錯誤的 log_level。
2. fork_server.bootstrap.create_server()
   — 回傳 ForkedCapsWriterServer 實例 (上游 CapsWriterServer 子類)。
3. server.start()
   — 阻塞主迴圈。若 http_api_enable=true, asyncio.gather 並行
     WebSocket send 迴圈與 uvicorn server。
"""

import sys
from multiprocessing import freeze_support

from fork_server.http_api.runtime_config import ConfigError


def run_configured_server(apply_env_config, create_server) -> None:
    try:
        apply_env_config()
    except ConfigError as exc:
        print(f"CapsWriter configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    server = create_server()
    server.start()


def main() -> None:
    from fork_server.bootstrap import apply_env_config, create_server

    run_configured_server(apply_env_config, create_server)


if __name__ == '__main__':
    # PyInstaller 多進程相容 (與上游 start_server.py 一致)
    freeze_support()
    main()
