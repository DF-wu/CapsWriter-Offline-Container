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
import shutil
import time
import uuid
from typing import Literal, Optional

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Response,
    UploadFile,
)
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from config_server import ServerConfig as Config, __version__
from core.constants import AudioFormat
from core.server.schema import Task, Result
from core.server import logger, console

from .audio_decoder import AudioDecodeError, FFmpegNotFoundError, decode_to_pcm
from .auth import auth_enabled, bearer_token_matches, extract_bearer_token
from .errors import http_exception_handler, validation_exception_handler
from .limits import (
    UploadTooLargeError,
    read_upload_limited,
    task_timeout_seconds,
    upload_limit_bytes,
)
from .openai_formatter import format_response
from .privacy import log_prompt_context, log_transcription_result
from .readiness import build_readiness, readiness_auth_enabled
from .runtime_config import normalize_cors_origins
from .task_router import router as task_router
from .transcription_tasks import (
    iter_transcription_task_specs,
    normalize_language_hint,
    normalize_prompt_context,
)


def _check_auth(authorization: Optional[str]) -> None:
    """空字串的 api_key 視為關閉認證 (預設本地使用場景)。"""
    api_key = getattr(Config, "http_api_key", "") or ""
    if not auth_enabled(api_key):
        return
    if extract_bearer_token(authorization) is None:
        raise HTTPException(401, "Missing or invalid Authorization header")
    if not bearer_token_matches(authorization, api_key):
        raise HTTPException(401, "Invalid API key")


def _split_and_submit(
    task_id: str,
    pcm: bytes,
    *,
    context: str = "",
    language: str = "auto",
) -> None:
    """
    把 PCM 切分成 60s + 4s overlap 的片段送入 queue_in,
    最後一段標記 is_final=True。

    與 ws_recv 內的分段邏輯同算法, 但純切分規則放在
    transcription_tasks.py 以便在無 FastAPI/server dependency 的環境測試。
    在 thread 中呼叫以避免阻塞 event loop。
    """
    queue_in = task_router.queue_in
    socket_id = task_router.synthetic_socket_id(task_id)

    time_start = time.time()
    for spec in iter_transcription_task_specs(
        task_id=task_id,
        socket_id=socket_id,
        pcm=pcm,
        time_start=time_start,
        context=context,
        language=language,
    ):
        queue_in.put(
            Task(
                **spec,
                time_submit=time.time(),
            )
        )


def _wrap_response(
    body,
    media_type: str,
    headers: Optional[dict[str, str]] = None,
) -> Response:
    if isinstance(body, (dict, list)):
        return JSONResponse(content=body, media_type=media_type, headers=headers)
    return PlainTextResponse(content=body, media_type=media_type, headers=headers)


def _cors_origins() -> list[str]:
    value = getattr(Config, "http_api_cors_origins", []) or []
    return list(normalize_cors_origins(value))


def _log_sensitive_text_enabled() -> bool:
    return bool(getattr(Config, "http_api_log_transcripts", False))


def create_app() -> FastAPI:
    app = FastAPI(
        title="CapsWriter-Offline OpenAI-Compatible ASR",
        version=__version__,
        description=(
            "Drop-in replacement for OpenAI Whisper transcription API, "
            "backed by local CapsWriter-Offline recognition (offline, private)."
        ),
    )

    cors_origins = _cors_origins()
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type"],
            max_age=600,
        )

    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)

    max_concurrent_requests = int(
        getattr(Config, "http_api_max_concurrent_requests", 2)
    )
    transcription_slots = asyncio.Semaphore(max(1, max_concurrent_requests))

    async def transcription_slot():
        async with transcription_slots:
            yield

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": Config.model_type,
            "version": __version__,
        }

    @app.get("/ready")
    async def ready():
        payload, status_code = build_readiness(
            model=Config.model_type,
            version=__version__,
            task_router_bound=task_router.is_bound(),
            ffmpeg_available=shutil.which("ffmpeg") is not None,
            auth_enabled=readiness_auth_enabled(getattr(Config, "http_api_key", "")),
            max_upload_mb=int(getattr(Config, "http_api_max_upload_mb", 100)),
            task_timeout=float(getattr(Config, "http_api_task_timeout", 600.0)),
            max_concurrent_requests=int(
                getattr(Config, "http_api_max_concurrent_requests", 2)
            ),
            cors_origins=cors_origins,
            log_transcripts=bool(
                getattr(Config, "http_api_log_transcripts", False)
            ),
        )
        return JSONResponse(content=payload, status_code=status_code)

    @app.get("/v1/models")
    async def list_models(authorization: Optional[str] = Header(None)):
        _check_auth(authorization)
        return {
            "object": "list",
            "data": [
                {
                    "id": Config.model_type,
                    "object": "model",
                    "owned_by": "capswriter-offline",
                    "created": 0,
                }
            ],
        }

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        file: UploadFile = File(..., description="Audio file"),
        model: str = Form("whisper-1"),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] = Form(
            "json"
        ),
        temperature: float = Form(0.0),
        authorization: Optional[str] = Header(None),
        _slot: None = Depends(transcription_slot),
    ):
        del _slot
        _check_auth(authorization)
        del model, temperature  # OpenAI 相容占位; 本地模型由 Config.model_type 決定
        request_start = time.time()
        task_id = uuid.uuid4().hex

        max_bytes, max_mb = upload_limit_bytes(
            getattr(Config, "http_api_max_upload_mb", 100)
        )
        try:
            audio_bytes = await read_upload_limited(file, max_bytes)
        except UploadTooLargeError:
            logger.warning(
                f"[HTTP] task={task_id[:8]} upload rejected: >{max_mb} MB"
            )
            raise HTTPException(413, f"File too large (>{max_mb} MB)")
        if not audio_bytes:
            raise HTTPException(400, "Empty file")

        try:
            pcm = await decode_to_pcm(audio_bytes)
        except FFmpegNotFoundError as e:
            logger.error(
                f"[HTTP] ffmpeg not found: {e}. "
                "Install ffmpeg or upload raw PCM (16kHz/f32le/mono)."
            )
            raise HTTPException(500, f"Server misconfigured: {e}")
        except AudioDecodeError as e:
            logger.warning(f"[HTTP] audio decode failed: {e}")
            raise HTTPException(400, f"Audio decode failed: {e}")

        duration = AudioFormat.bytes_to_seconds(len(pcm))
        if duration < 0.05:
            raise HTTPException(400, "Audio too short to transcribe")

        future = task_router.register(task_id)
        context = normalize_prompt_context(prompt)
        language_hint = normalize_language_hint(language)
        logger.info(
            f"[HTTP] task={task_id[:8]} duration={duration:.2f}s "
            f"fmt={response_format} lang={language_hint} bytes={len(audio_bytes)}"
        )

        timeout = task_timeout_seconds(getattr(Config, "http_api_task_timeout", 600.0))
        try:
            await asyncio.to_thread(
                _split_and_submit,
                task_id,
                pcm,
                context=context,
                language=language_hint,
            )
            log_prompt_context(
                logger,
                task_id,
                context,
                log_sensitive_text=_log_sensitive_text_enabled(),
            )
            result: Result = await asyncio.wait_for(future, timeout=timeout)
        except asyncio.CancelledError:
            task_router.cancel(task_id)
            logger.warning(f"[HTTP] task={task_id[:8]} request cancelled")
            raise
        except asyncio.TimeoutError:
            task_router.cancel(task_id)
            logger.error(
                f"[HTTP] task={task_id[:8]} timeout after {timeout:.0f}s "
                f"(duration={duration:.1f}s). Audio may be too long or model too slow."
            )
            raise HTTPException(504, "Recognition timeout")
        except Exception as e:
            task_router.cancel(task_id)
            logger.error(
                f"[HTTP] task={task_id[:8]} unexpected error: {e}", exc_info=True
            )
            raise HTTPException(500, f"Recognition error: {e}")

        body, media_type = format_response(
            result, response_format, language=language_hint
        )
        text = result.text_accu or result.text
        delay = max(0.0, time.time() - request_start)
        log_transcription_result(
            logger,
            console,
            task_id,
            delay,
            text,
            log_sensitive_text=_log_sensitive_text_enabled(),
        )
        return _wrap_response(
            body,
            media_type,
            headers={"X-CapsWriter-Task-ID": task_id},
        )

    @app.post("/v1/audio/translations")
    async def translations():
        # CapsWriter 本地模型不做語種翻譯, 明確返回 501
        raise HTTPException(
            501,
            "translations endpoint is not implemented; CapsWriter only transcribes",
        )

    return app
