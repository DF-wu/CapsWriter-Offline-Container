# coding: utf-8
"""
OpenAI-compatible HTTP API

提供与 OpenAI Whisper /v1/audio/transcriptions 接口同形的 REST 端点,
让任何使用 OpenAI Python SDK 的客户端只需切换 base_url 即可接入本服务。

实作要点:
- 与既有的 WebSocket 服务并存于同一 asyncio loop (core_server.py 中 gather)。
- HTTP 任务通过 task_router 注册 Future, 复用 Cosmic.queue_in / queue_out
  以及单一识别子进程, 不另起 worker pool。
- 自然并发 = 队列 backpressure + asyncio 调度: 识别本身严格串行,
  解码/格式化阶段可在不同请求间重叠。

详尽规格见 docs/HTTP_API.md。
"""

import asyncio
import hmac
import shutil
import time
import uuid
from typing import List, Literal, Optional, Tuple

from fastapi import (
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import JSONResponse, PlainTextResponse

from config_server import ServerConfig as Config, __version__
from util.constants import AudioFormat
from util.server.audio_decoder import (
    AudioDecodeError,
    FFmpegNotFoundError,
    decode_to_pcm,
)
from util.server.openai_formatter import format_response
from util.server.http_limits import UploadTooLargeError, read_upload_limited
from util.server.server_classes import Task, Result
from util.server.server_cosmic import Cosmic
from util.server.task_router import router as task_router

from . import logger


# HTTP 任务的分段参数与客户端文件模式一致 (util/client/transcribe)
DEFAULT_SEG_DURATION = 60.0
DEFAULT_SEG_OVERLAP = 4.0

SUPPORTED_FORMATS = ("json", "text", "srt", "verbose_json", "vtt")


def _check_auth(authorization: Optional[str]) -> None:
    """空字串的 api_key 视为关闭认证 (默认本地使用场景)。"""
    api_key = Config.http_api_key
    if not api_key:
        return
    if not authorization:
        raise HTTPException(401, "Missing or invalid Authorization header")
    parts = authorization.strip().split()
    if len(parts) != 2 or parts[0].casefold() != "bearer" or not parts[1]:
        raise HTTPException(401, "Missing or invalid Authorization header")
    token = parts[1]
    if not hmac.compare_digest(token.encode("utf-8"), api_key.encode("utf-8")):
        raise HTTPException(401, "Invalid API key")


def _check_transcription_content_type(content_type: Optional[str]) -> None:
    """Reject parser surfaces that the legacy endpoint never supports."""
    media_type = (content_type or "").partition(";")[0].strip().casefold()
    if media_type != "multipart/form-data":
        raise HTTPException(415, "Content-Type must be multipart/form-data")


def _split_and_submit(task_id: str, pcm: bytes) -> None:
    """
    把 PCM 切分成 60s + 4s overlap 的片段送入 queue_in,
    最后一段标记 is_final=True。

    与 server_ws_recv.py 内的分段逻辑同算法, 复用 AudioFormat 转换工具,
    保持与 WebSocket 路径相同的识别质量。

    在 thread 中调用以避免阻塞 event loop (multiprocessing.Queue.put 是阻塞 IO)。
    """
    queue_in = Cosmic.queue_in
    socket_id = task_router.synthetic_socket_id(task_id)

    segment_bytes = AudioFormat.seconds_to_bytes(DEFAULT_SEG_DURATION + DEFAULT_SEG_OVERLAP)
    stride_bytes = AudioFormat.seconds_to_bytes(DEFAULT_SEG_DURATION)
    total_bytes = len(pcm)

    time_start = time.time()
    offset = 0.0
    pos = 0

    # 短音频: 单一 final task 即可
    if total_bytes <= segment_bytes:
        queue_in.put(Task(
            source="file",
            data=pcm,
            offset=0.0,
            overlap=DEFAULT_SEG_OVERLAP,
            task_id=task_id,
            socket_id=socket_id,
            is_final=True,
            time_start=time_start,
            time_submit=time.time(),
            context="",
        ))
        return

    # 长音频: 多段中间 + 1 段 final
    while pos + segment_bytes < total_bytes:
        queue_in.put(Task(
            source="file",
            data=pcm[pos:pos + segment_bytes],
            offset=offset,
            overlap=DEFAULT_SEG_OVERLAP,
            task_id=task_id,
            socket_id=socket_id,
            is_final=False,
            time_start=time_start,
            time_submit=time.time(),
            context="",
        ))
        offset += DEFAULT_SEG_DURATION
        pos += stride_bytes

    queue_in.put(Task(
        source="file",
        data=pcm[pos:],
        offset=offset,
        overlap=DEFAULT_SEG_OVERLAP,
        task_id=task_id,
        socket_id=socket_id,
        is_final=True,
        time_start=time_start,
        time_submit=time.time(),
        context="",
    ))


def _wrap_response(body, media_type: str) -> Response:
    if isinstance(body, (dict, list)):
        return JSONResponse(content=body, media_type=media_type)
    return PlainTextResponse(content=body, media_type=media_type)


def create_app() -> FastAPI:
    app = FastAPI(
        title="CapsWriter-Offline OpenAI-Compatible ASR",
        version=__version__,
        description=(
            "Drop-in replacement for OpenAI Whisper transcription API, "
            "backed by local CapsWriter-Offline recognition (offline, private)."
        ),
    )

    @app.middleware("http")
    async def guard_transcription_request(request: Request, call_next):
        # Use the ASGI scope path rather than request.url so malformed Host
        # headers cannot influence this security decision. Authentication and
        # media-type rejection happen before Starlette parses multipart fields.
        if (
            request.method == "POST"
            and request.scope.get("path") == "/v1/audio/transcriptions"
        ):
            try:
                _check_auth(request.headers.get("authorization"))
                _check_transcription_content_type(request.headers.get("content-type"))
            except HTTPException as error:
                return JSONResponse(
                    status_code=error.status_code,
                    content={"detail": error.detail},
                    headers=error.headers,
                )
        return await call_next(request)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": Config.model_type,
            "version": __version__,
        }

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{
                "id": Config.model_type,
                "object": "model",
                "owned_by": "capswriter-offline",
                "created": 0,
            }],
        }

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        file: UploadFile = File(..., description="Audio file"),
        model: str = Form("whisper-1"),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] = Form("json"),
        temperature: float = Form(0.0),
        authorization: Optional[str] = Header(None),
    ):
        _check_auth(authorization)
        del model, temperature  # 仅作 OpenAI 兼容占位, 本地模型由 Config.model_type 决定

        max_bytes = Config.http_api_max_upload_mb * 1024 * 1024
        try:
            audio_bytes = await read_upload_limited(file, max_bytes)
        except UploadTooLargeError:
            raise HTTPException(413, f"File too large (>{Config.http_api_max_upload_mb} MB)")
        if not audio_bytes:
            raise HTTPException(400, "Empty file")

        try:
            pcm = await decode_to_pcm(audio_bytes)
        except FFmpegNotFoundError as e:
            # 服务器侧配置问题, 不是上传者的错。
            raise HTTPException(500, f"Server misconfigured: {e}")
        except AudioDecodeError as e:
            raise HTTPException(400, f"Audio decode failed: {e}")

        duration = AudioFormat.bytes_to_seconds(len(pcm))
        if duration < 0.05:
            raise HTTPException(400, "Audio too short to transcribe")

        task_id = uuid.uuid4().hex
        future = task_router.register(task_id)
        logger.info(
            f"[HTTP] task={task_id[:8]} duration={duration:.2f}s "
            f"fmt={response_format} bytes={len(audio_bytes)}"
        )

        try:
            await asyncio.to_thread(_split_and_submit, task_id, pcm)
            if prompt:
                # 当前 Task.context 是按片段设置的; 这里只记录长度，避免逐字稿内容落盘。
                # 完整的 prompt-as-context 注入留待 Fun-ASR-Nano 整段 prompt 支援。
                logger.debug(f"[HTTP] task={task_id[:8]} prompt_chars={len(prompt)}")
            result: Result = await asyncio.wait_for(
                future, timeout=Config.http_api_task_timeout
            )
        except asyncio.CancelledError:
            task_router.cancel(task_id)
            logger.warning(f"[HTTP] task={task_id[:8]} request cancelled")
            raise
        except asyncio.TimeoutError:
            task_router.cancel(task_id)
            logger.error(
                f"[HTTP] task={task_id[:8]} timeout after {Config.http_api_task_timeout:.0f}s"
            )
            raise HTTPException(504, "Recognition timeout")
        except Exception as e:
            task_router.cancel(task_id)
            logger.error(f"[HTTP] task={task_id[:8]} error: {e}", exc_info=True)
            raise HTTPException(500, "Recognition failed")

        body, media_type = format_response(result, response_format, language=language)
        text = result.text_accu or result.text
        logger.info(f"[HTTP] task={task_id[:8]} done, text_len={len(text)}")
        return _wrap_response(body, media_type)

    @app.post("/v1/audio/translations")
    async def translations():
        # CapsWriter 本地模型不做语种翻译, 明确返回 501
        raise HTTPException(
            501,
            "translations endpoint is not implemented; CapsWriter only transcribes",
        )

    return app


async def run_http_server() -> None:
    """以 programmatic 模式启动 uvicorn, 与 WebSocket server 共用同一 asyncio loop。"""
    import uvicorn  # 延迟导入: 当 enable=False 时不必引入 fastapi/uvicorn

    app = create_app()
    config = uvicorn.Config(
        app,
        host=Config.http_api_bind,
        port=int(Config.http_api_port),
        log_level=Config.log_level.lower(),
        access_log=False,
        loop="asyncio",
    )
    server = uvicorn.Server(config)
    task_router.bind_loop(asyncio.get_running_loop())
    logger.info(
        f"HTTP API 监听 {Config.http_api_bind}:{Config.http_api_port} "
        f"(auth={'on' if Config.http_api_key else 'off'})"
    )
    if shutil.which("ffmpeg") is None:
        logger.warning(
            "HTTP API 已启用但系统找不到 ffmpeg; "
            "/v1/audio/transcriptions 对非 raw PCM 上传会回 500。"
            "请在 OS 安装 ffmpeg, 或在 Docker image 中确保已包含。"
        )
    await server.serve()
