# coding: utf-8
"""
HTTP API uvicorn cotask

與上游 ws_send 並行於同一 asyncio loop。
"""

from __future__ import annotations
import asyncio
import shutil

from config_server import ServerConfig as Config
from core.server import logger

from .api import create_app
from .task_router import router as task_router


async def run_http_server(cw_server) -> None:
    """
    Programmatic uvicorn launch, 共用 CapsWriterServer 的 asyncio loop。

    Args:
        cw_server: ForkedCapsWriterServer instance (用於存取 state)
    """
    import uvicorn  # 延遲匯入: enable=False 時不必引入 fastapi/uvicorn

    # task_router 必須在 register() 之前 bind state + loop
    task_router.bind(cw_server.state, asyncio.get_running_loop())

    app = create_app()
    config = uvicorn.Config(
        app,
        host=getattr(Config, "http_api_bind", "127.0.0.1"),
        port=int(getattr(Config, "http_api_port", 6017)),
        log_level=str(getattr(Config, "log_level", "INFO")).lower(),
        access_log=False,
        loop="asyncio",
    )
    server = uvicorn.Server(config)

    api_key = getattr(Config, "http_api_key", "") or ""
    logger.info(
        f"HTTP API 監聽 {config.host}:{config.port} "
        f"(auth={'on' if api_key else 'off'})"
    )

    if shutil.which("ffmpeg") is None:
        logger.warning(
            "HTTP API 已啟用但系統找不到 ffmpeg; "
            "/v1/audio/transcriptions 對非 raw PCM 上傳會回 500。"
        )

    await server.serve()
