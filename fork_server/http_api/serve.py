# coding: utf-8
"""
HTTP API uvicorn cotask

與上游 ws_send 並行於同一 asyncio loop。
"""

from __future__ import annotations
import asyncio
import shutil

from config_server import ServerConfig as Config
from core.server import logger, console

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
        timeout_graceful_shutdown=5,
    )
    server = uvicorn.Server(config)
    cw_server._http_server = server
    if getattr(cw_server, "_http_stop_requested", False):
        server.should_exit = True

    api_key = getattr(Config, "http_api_key", "") or ""
    max_upload = getattr(Config, "http_api_max_upload_mb", 100)
    max_audio_seconds = getattr(Config, "http_api_max_audio_seconds", 3600)
    max_concurrent = getattr(Config, "http_api_max_concurrent_requests", 2)
    max_pending = getattr(Config, "http_api_max_pending_requests", 4)
    logger.info(
        f"HTTP API 監聽 {config.host}:{config.port} "
        f"(auth={'on' if api_key else 'off'}, max_upload={max_upload}MB, "
        f"max_audio={max_audio_seconds}s, max_concurrent={max_concurrent}, "
        f"max_pending={max_pending})"
    )
    auth_flag = f'-H "Authorization: Bearer $KEY"' if api_key else ""
    auth_hint = "  [yellow](需 API key)[/]" if api_key else ""
    key_arg = "--key $KEY " if api_key else ""

    console.print()
    console.print(
        f"  [bold cyan]OpenAI 相容 API[/] — [green]http://{config.host}:{config.port}[/]"
        f"{auth_hint}"
    )
    console.print(f"    • 健康检查: curl http://{config.host}:{config.port}/health")
    console.print(
        f"    • 模型列表: curl {auth_flag} http://{config.host}:{config.port}/v1/models"
    )
    console.print(
        f"    • 语音转写: curl -X POST {auth_flag} "
        f"http://{config.host}:{config.port}/v1/audio/transcriptions "
        f'-F "file=@test.wav" -F "model=whisper-1"'
    )
    console.print(f"    • 诊断工具: python check_http_api.py {key_arg}--audio test.wav")

    if shutil.which("ffmpeg") is None:
        console.print(f"    [red]⚠ ffmpeg 未安装[/] — 非 raw PCM 音频会上传失败")
    else:
        console.print(f"    • ffmpeg: [green]已安装[/]")

    console.print()

    try:
        await server.serve()
    finally:
        if getattr(cw_server, "_http_server", None) is server:
            cw_server._http_server = None
