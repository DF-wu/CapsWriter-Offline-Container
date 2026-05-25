# coding: utf-8
"""
OpenAI-compatible HTTP API

提供與 OpenAI Whisper /v1/audio/transcriptions 接口同形的 REST 端點,
讓任何使用 OpenAI SDK 的客戶端只需切換 base_url 即可接入本服務。

整合方式:
- HTTP 任務透過 task_router 註冊 Future, 複用 state.queue_in/queue_out
  與單一識別子進程 (不另起 worker pool)。
- 並發 = 隊列 backpressure + asyncio 調度: 識別本身嚴格串行,
  解碼 / 格式化階段可在不同請求間重疊。

詳盡規格見 docs/HTTP_API.md。
"""

from __future__ import annotations
import asyncio
import time
import uuid
from typing import Literal, Optional

from fastapi import (
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Response,
    UploadFile,
)
from fastapi.responses import JSONResponse, PlainTextResponse

from config_server import ServerConfig as Config, __version__
from core.constants import AudioFormat
from core.server.schema import Task, Result
from core.server import logger

from .audio_decoder import AudioDecodeError, FFmpegNotFoundError, decode_to_pcm
from .openai_formatter import format_response
from .task_router import router as task_router


# HTTP 任務的分段參數與客戶端檔案模式一致
DEFAULT_SEG_DURATION = 60.0
DEFAULT_SEG_OVERLAP = 4.0


def _check_auth(authorization: Optional[str]) -> None:
    """空字串的 api_key 視為關閉認證 (預設本地使用場景)。"""
    api_key = getattr(Config, "http_api_key", "") or ""
    if not api_key:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header")
    token = authorization[len("Bearer "):].strip()
    if token != api_key:
        raise HTTPException(401, "Invalid API key")


def _split_and_submit(task_id: str, pcm: bytes) -> None:
    """
    把 PCM 切分成 60s + 4s overlap 的片段送入 queue_in,
    最後一段標記 is_final=True。

    與 ws_recv 內的分段邏輯同算法, 複用 AudioFormat 轉換工具。
    在 thread 中呼叫以避免阻塞 event loop。
    """
    queue_in = task_router.queue_in
    socket_id = task_router.synthetic_socket_id(task_id)

    segment_bytes = AudioFormat.seconds_to_bytes(DEFAULT_SEG_DURATION + DEFAULT_SEG_OVERLAP)
    stride_bytes = AudioFormat.seconds_to_bytes(DEFAULT_SEG_DURATION)
    total_bytes = len(pcm)

    time_start = time.time()
    offset = 0.0
    pos = 0

    # 短音訊: 單一 final task 即可
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

    # 長音訊: 多段中間 + 1 段 final
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
        del model, temperature  # OpenAI 相容占位; 本地模型由 Config.model_type 決定

        max_bytes = int(getattr(Config, "http_api_max_upload_mb", 100)) * 1024 * 1024
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(400, "Empty file")
        if len(audio_bytes) > max_bytes:
            raise HTTPException(413, f"File too large (>{max_bytes // (1024*1024)} MB)")

        try:
            pcm = await decode_to_pcm(audio_bytes)
        except FFmpegNotFoundError as e:
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

        timeout = float(getattr(Config, "http_api_task_timeout", 600.0))
        try:
            await asyncio.to_thread(_split_and_submit, task_id, pcm)
            if prompt:
                logger.debug(f"[HTTP] task={task_id[:8]} prompt={prompt[:50]!r}")
            result: Result = await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            task_router.cancel(task_id)
            logger.error(f"[HTTP] task={task_id[:8]} timeout after {timeout:.0f}s")
            raise HTTPException(504, "Recognition timeout")
        except Exception as e:
            task_router.cancel(task_id)
            logger.error(f"[HTTP] task={task_id[:8]} error: {e}", exc_info=True)
            raise HTTPException(500, f"Recognition error: {e}")

        body, media_type = format_response(result, response_format, language=language)
        text = result.text_accu or result.text
        logger.info(f"[HTTP] task={task_id[:8]} done, text_len={len(text)}")
        return _wrap_response(body, media_type)

    @app.post("/v1/audio/translations")
    async def translations():
        # CapsWriter 本地模型不做語種翻譯, 明確返回 501
        raise HTTPException(
            501,
            "translations endpoint is not implemented; CapsWriter only transcribes",
        )

    return app
