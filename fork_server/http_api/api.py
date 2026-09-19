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
from contextlib import asynccontextmanager
from dataclasses import dataclass
import math
import queue
import shutil
import threading
import time
import uuid
from typing import Optional

from fastapi import (
    FastAPI,
    Header,
    HTTPException,
    Request,
    Response,
)
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.formparsers import MultiPartException
from starlette.requests import ClientDisconnect
from python_multipart.exceptions import MultipartParseError

from config_server import ServerConfig as Config, __version__
from core.constants import AudioFormat
from core.server.schema import Task, Result
from core.server import logger, console
from core.server.queue_limits import QUEUE_PUT_RETRY_SECONDS

from .admission import (
    AdmissionController,
    AdmissionQueueFullError,
    ReplayableReceive,
)
from .audio_decoder import (
    AudioDecodeError,
    AudioDecodeTimeoutError,
    AudioTooLongError,
    FFmpegNotFoundError,
    decode_to_pcm,
)
from .auth import auth_enabled, bearer_token_matches, extract_bearer_token
from .body_limit import RequestBodyLimitMiddleware
from .errors import http_exception_handler, validation_exception_handler
from .limits import (
    RequestBodyTooLargeError,
    UploadTooLargeError,
    audio_limit_bytes,
    read_upload_limited,
    request_body_limit_bytes,
    task_timeout_seconds,
    upload_limit_bytes,
)
from .multipart_form import close_form_files, parse_multipart_form
from .openai_formatter import format_response
from .privacy import (
    log_internal_exception,
    log_prompt_context,
    log_transcription_result,
)
from .readiness import build_readiness, readiness_auth_enabled
from .runtime_config import ConfigError, normalize_cors_origins, normalize_http_origin
from .task_router import router as task_router
from .transcription_tasks import (
    iter_transcription_task_specs,
    normalize_language_hint,
    normalize_prompt_context,
)


OPENAI_TRANSCRIPTION_MODEL = "whisper-1"
MAX_MODEL_NAME_CHARS = 64
SUPPORTED_RESPONSE_FORMATS = frozenset(
    {"json", "text", "srt", "verbose_json", "vtt"}
)
SUPPORTED_TIMESTAMP_GRANULARITIES = frozenset({"segment", "word"})
MAX_MULTIPART_FILES = 1
MAX_MULTIPART_FIELDS = 12
_SCALAR_FORM_FIELDS = frozenset(
    {"file", "model", "language", "prompt", "response_format", "temperature", "stream"}
)
_SUPPORTED_FORM_FIELDS = _SCALAR_FORM_FIELDS | frozenset(
    {"timestamp_granularities", "timestamp_granularities[]"}
)
_UNSUPPORTED_CAPABILITY_FIELDS = {
    "chunking_strategy": "speaker-aware chunking",
    "include": "log probabilities",
    "include[]": "log probabilities",
    "known_speaker_names": "known-speaker diarization",
    "known_speaker_names[]": "known-speaker diarization",
    "known_speaker_references": "known-speaker diarization",
    "known_speaker_references[]": "known-speaker diarization",
    "logprobs": "log probabilities",
}

CLIENT_DISCONNECT_POLL_SECONDS = 0.05

_TIMESTAMP_GRANULARITY_SCHEMA = {
    "type": "array",
    "items": {
        "type": "string",
        "enum": sorted(SUPPORTED_TIMESTAMP_GRANULARITIES),
    },
}
_TRANSCRIPTION_JSON_SCHEMA = {
    "type": "object",
    "required": ["text"],
    "additionalProperties": False,
    "properties": {"text": {"type": "string"}},
}
_TRANSCRIPTION_SEGMENT_SCHEMA = {
    "type": "object",
    "required": [
        "id",
        "seek",
        "start",
        "end",
        "text",
        "tokens",
        "temperature",
        "avg_logprob",
        "compression_ratio",
        "no_speech_prob",
    ],
    "properties": {
        "id": {"type": "integer"},
        "seek": {"type": "integer"},
        "start": {"type": "number"},
        "end": {"type": "number"},
        "text": {"type": "string"},
        "tokens": {"type": "array", "items": {"type": "integer"}},
        "temperature": {"type": "number"},
        "avg_logprob": {"type": "number"},
        "compression_ratio": {"type": "number"},
        "no_speech_prob": {"type": "number"},
    },
}
_TRANSCRIPTION_WORD_SCHEMA = {
    "type": "object",
    "required": ["word", "start", "end"],
    "properties": {
        "word": {"type": "string"},
        "start": {"type": "number"},
        "end": {"type": "number"},
    },
}
_VERBOSE_TRANSCRIPTION_SCHEMA = {
    "type": "object",
    "required": ["task", "language", "duration", "text"],
    "additionalProperties": False,
    "properties": {
        "task": {"type": "string", "enum": ["transcribe"]},
        "language": {
            "type": "string",
            "description": (
                "Normalized request hint, or the local 'auto' sentinel when "
                "no reliable detected language is available."
            ),
        },
        "duration": {"type": "number", "minimum": 0.0},
        "text": {"type": "string"},
        "segments": {
            "type": "array",
            "items": _TRANSCRIPTION_SEGMENT_SCHEMA,
        },
        "words": {
            "type": "array",
            "items": _TRANSCRIPTION_WORD_SCHEMA,
        },
    },
}
_OPENAI_ERROR_SCHEMA = {
    "type": "object",
    "required": ["error"],
    "properties": {
        "error": {
            "type": "object",
            "required": ["message", "type", "param", "code"],
            "properties": {
                "message": {"type": "string"},
                "type": {"type": "string"},
                "param": {"type": ["string", "null"]},
                "code": {"type": ["string", "null"]},
            },
        }
    },
}

TRANSCRIPTION_OPENAPI_EXTRA = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["file", "model"],
                    "properties": {
                        "file": {"type": "string", "format": "binary"},
                        "model": {
                            "type": "string",
                            "enum": [OPENAI_TRANSCRIPTION_MODEL],
                        },
                        "language": {"type": "string"},
                        "prompt": {"type": "string"},
                        "response_format": {
                            "type": "string",
                            "enum": sorted(SUPPORTED_RESPONSE_FORMATS),
                            "default": "json",
                        },
                        "temperature": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.0,
                        },
                        "stream": {
                            "type": "boolean",
                            "enum": [False],
                            "default": False,
                        },
                        "timestamp_granularities": _TIMESTAMP_GRANULARITY_SCHEMA,
                        "timestamp_granularities[]": _TIMESTAMP_GRANULARITY_SCHEMA,
                    },
                }
            }
        },
    },
    "responses": {
        "200": {
            "description": "Transcription completed",
            "headers": {
                "X-CapsWriter-Task-ID": {
                    "description": "Correlation ID for local server logs",
                    "schema": {"type": "string"},
                }
            },
            "content": {
                "application/json": {
                    "schema": {
                        "anyOf": [
                            _TRANSCRIPTION_JSON_SCHEMA,
                            _VERBOSE_TRANSCRIPTION_SCHEMA,
                        ]
                    }
                },
                "text/plain": {"schema": {"type": "string"}},
                "application/x-subrip": {"schema": {"type": "string"}},
                "text/vtt": {"schema": {"type": "string"}},
            },
        },
        **{
            str(status): {
                "description": description,
                "content": {
                    "application/json": {"schema": _OPENAI_ERROR_SCHEMA}
                },
            }
            for status, description in (
                (400, "Invalid request or unsupported capability"),
                (401, "Authentication failed"),
                (403, "Browser origin is not allowed"),
                (413, "Upload, request body, or decoded audio is too large"),
                (422, "Request validation failed"),
                (429, "Admission queue is full"),
                (500, "Internal transcription or decoder failure"),
                (503, "Recognizer is unavailable or fail-stopping"),
                (504, "End-to-end transcription deadline expired"),
            )
        },
    },
}


@dataclass(frozen=True)
class ParsedTranscriptionRequest:
    audio_bytes: bytes
    temperature: float
    response_format: str
    context: str
    language_hint: str
    timestamp_granularities: tuple[str, ...]


def _request_error(
    message: str,
    *,
    status_code: int = 400,
    headers: Optional[dict[str, str]] = None,
) -> HTTPException:
    """Build an error handled by the OpenAI-style exception envelope."""
    if headers is None:
        return HTTPException(status_code=status_code, detail=message)
    return HTTPException(status_code=status_code, detail=message, headers=headers)


def _validate_model(model: str) -> str:
    value = (model or "").strip()
    if not value:
        raise _request_error("Invalid model: a non-empty model is required")
    if len(value) > MAX_MODEL_NAME_CHARS:
        raise _request_error(
            f"Invalid model: must be at most {MAX_MODEL_NAME_CHARS} characters"
        )
    if value != OPENAI_TRANSCRIPTION_MODEL:
        raise _request_error(
            f"Unsupported model {value!r}; this server implements only "
            f"{OPENAI_TRANSCRIPTION_MODEL!r} non-streaming transcriptions"
        )
    return value


def _validate_temperature(temperature: float) -> float:
    try:
        value = float(temperature)
    except (TypeError, ValueError) as exc:
        raise _request_error("Invalid temperature: must be a number from 0 to 1") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise _request_error("Invalid temperature: must be a finite number from 0 to 1")
    return value


def _validate_response_format(response_format: str) -> str:
    value = (response_format or "").strip().lower()
    if value == "diarized_json":
        raise _request_error(
            "Unsupported capability 'diarized_json': this server does not provide "
            "speaker diarization"
        )
    if value not in SUPPORTED_RESPONSE_FORMATS:
        choices = ", ".join(sorted(SUPPORTED_RESPONSE_FORMATS))
        raise _request_error(
            f"Invalid response_format {value!r}; supported values: {choices}"
        )
    return value


def _validate_stream_value(value: Optional[str]) -> None:
    if value is None:
        return
    normalized = str(value).strip().lower()
    if normalized in {"", "0", "false"}:
        return
    if normalized in {"1", "true"}:
        raise _request_error(
            "Unsupported capability 'stream': this server implements only "
            "non-streaming whisper-1 transcriptions"
        )
    raise _request_error("Invalid stream: expected true or false")


def _validate_form_fields(form) -> None:
    keys = {str(key) for key in form.keys()}
    for field in sorted(keys & _UNSUPPORTED_CAPABILITY_FIELDS.keys()):
        capability = _UNSUPPORTED_CAPABILITY_FIELDS[field]
        raise _request_error(
            f"Unsupported capability in field {field!r}: this server does not "
            f"provide {capability}"
        )

    unknown = sorted(keys - _SUPPORTED_FORM_FIELDS - _UNSUPPORTED_CAPABILITY_FIELDS.keys())
    if unknown:
        raise _request_error(f"Unsupported transcription field: {unknown[0]!r}")

    for field in _SCALAR_FORM_FIELDS:
        if len(form.getlist(field)) > 1:
            raise _request_error(f"Invalid request: duplicate field {field!r}")


def _text_form_values(form, field: str) -> list[str]:
    values = form.getlist(field)
    if any(not isinstance(value, str) for value in values):
        raise _request_error(f"Invalid {field}: expected a text value")
    return values


def _text_form_value(form, field: str, default: Optional[str]) -> Optional[str]:
    values = _text_form_values(form, field)
    return values[0] if values else default


def _validate_timestamp_granularities(
    values: list[str],
    response_format: str,
) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw in values:
        value = str(raw).strip().lower()
        if value not in SUPPORTED_TIMESTAMP_GRANULARITIES:
            choices = ", ".join(sorted(SUPPORTED_TIMESTAMP_GRANULARITIES))
            raise _request_error(
                f"Invalid timestamp_granularities value {value!r}; "
                f"supported values: {choices}"
            )
        if value not in normalized:
            normalized.append(value)

    if normalized and response_format != "verbose_json":
        raise _request_error(
            "timestamp_granularities[] requires response_format='verbose_json'"
        )
    if response_format == "verbose_json" and not normalized:
        return ("segment",)
    return tuple(normalized)


def _validate_multipart_headers(request: Request, max_body_bytes: int) -> None:
    close_headers = {"Connection": "close"}
    content_type = request.headers.get("content-type", "")
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type != "multipart/form-data":
        raise _request_error(
            "Content-Type must be multipart/form-data",
            headers=close_headers,
        )

    content_lengths = request.headers.getlist("content-length")
    if len(content_lengths) > 1:
        raise _request_error("Invalid Content-Length header", headers=close_headers)
    if not content_lengths:
        return
    try:
        content_length = int(content_lengths[0])
    except (TypeError, ValueError) as exc:
        raise _request_error(
            "Invalid Content-Length header",
            headers=close_headers,
        ) from exc
    if content_length < 0:
        raise _request_error("Invalid Content-Length header", headers=close_headers)
    if content_length > max_body_bytes:
        raise _request_error(
            "Request body is too large",
            status_code=413,
            headers=close_headers,
        )


def _uploaded_file(form) -> UploadFile:
    files = form.getlist("file")
    if not files:
        raise _request_error("Missing required field: 'file'")
    if len(files) != 1:
        raise _request_error("Invalid request: duplicate field 'file'")
    upload = files[0]
    if not isinstance(upload, UploadFile):
        raise _request_error("Invalid file: expected a multipart file upload")
    return upload


def _remaining_seconds(deadline: float) -> float:
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        raise asyncio.TimeoutError
    return remaining


async def _await_cleanup_task(cleanup_task: asyncio.Task):
    """Join independent cleanup despite repeated caller cancellation."""
    cancellation = None
    while True:
        try:
            result = await asyncio.shield(cleanup_task)
            break
        except asyncio.CancelledError as exc:
            cancellation = cancellation or exc
            if cleanup_task.cancelled():
                raise
            if cleanup_task.done():
                result = cleanup_task.result()
                break

    if cancellation is not None:
        raise cancellation
    return result


async def _wait_for_client_disconnect(
    request: Request,
    stop_event: asyncio.Event,
) -> None:
    """Observe a disconnect after the multipart body has been consumed."""
    while not stop_event.is_set():
        if await request.is_disconnected():
            return
        if stop_event.is_set():
            return
        await asyncio.sleep(CLIENT_DISCONNECT_POLL_SECONDS)


@asynccontextmanager
async def _admission_slot_or_disconnect(
    request: Request,
    controller: AdmissionController,
):
    """Hold an admission slot while preserving pre-body disconnect probes."""
    replay_receive = ReplayableReceive(request.receive)
    # Starlette's Request.stream() reads this callable.  Replacing it before
    # the first receive lets the admission watcher buffer and later replay any
    # http.request events it observes while waiting for a slot.
    request._receive = replay_receive

    acquired = asyncio.get_running_loop().create_future()
    release = asyncio.Event()

    async def hold_slot() -> None:
        async with controller.slot():
            if not acquired.done():
                acquired.set_result(None)
            await release.wait()

    holder = asyncio.create_task(
        hold_slot(),
        name="capswriter-http-admission-holder",
    )
    disconnect_watcher: asyncio.Task | None = None
    try:
        # Give an immediately available slot or a full queue one turn to
        # resolve.  Neither case should read even one byte from the body.
        await asyncio.sleep(0)
        if holder.done():
            holder.result()

        if not acquired.done():
            disconnect_watcher = asyncio.create_task(
                replay_receive.wait_for_disconnect(),
                name="capswriter-http-admission-disconnect",
            )
            done, _pending = await asyncio.wait(
                (acquired, holder, disconnect_watcher),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if disconnect_watcher in done:
                disconnect_watcher.result()
                raise ClientDisconnect
            if holder in done:
                holder.result()

        # If the peer and slot became ready in one loop turn, do not start
        # expensive multipart/decode work for a connection already gone.
        if disconnect_watcher is not None and disconnect_watcher.done():
            disconnect_watcher.result()
            raise ClientDisconnect

        if disconnect_watcher is not None:
            disconnect_watcher.cancel()
            await asyncio.gather(disconnect_watcher, return_exceptions=True)
            disconnect_watcher = None
        await acquired
        yield
    finally:
        async def finish_cleanup() -> None:
            if disconnect_watcher is not None:
                disconnect_watcher.cancel()
            release.set()
            if not acquired.done():
                holder.cancel()
            try:
                await asyncio.gather(
                    holder,
                    *(
                        ()
                        if disconnect_watcher is None
                        else (disconnect_watcher,)
                    ),
                    return_exceptions=True,
                )
            finally:
                replay_receive.close()

        cleanup_task = asyncio.create_task(
            finish_cleanup(),
            name="capswriter-http-admission-cleanup",
        )
        await _await_cleanup_task(cleanup_task)


async def _await_result_or_disconnect(
    request: Request,
    future: asyncio.Future,
    *,
    timeout: float,
) -> Result:
    """Race inference against disconnect without transferring future ownership."""
    disconnect_stop = asyncio.Event()
    disconnect_task = asyncio.create_task(
        _wait_for_client_disconnect(request, disconnect_stop),
        name="capswriter-http-client-disconnect",
    )
    try:
        done, _pending = await asyncio.wait(
            (future, disconnect_task),
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        # Prefer a completed result when both events become visible in the same
        # event-loop turn. asyncio.wait itself never cancels the router future.
        if future in done:
            return future.result()
        if disconnect_task in done:
            disconnect_task.result()
            await asyncio.sleep(0)
            if future.done():
                return future.result()
            raise ClientDisconnect
        raise asyncio.TimeoutError
    finally:
        # Starlette's nonblocking disconnect probe uses its own cancellation
        # scope, which may consume an unlucky task cancellation. The explicit
        # stop flag guarantees the next probe boundary still exits.
        disconnect_stop.set()
        disconnect_task.cancel()

        async def finish_cleanup() -> None:
            await asyncio.gather(disconnect_task, return_exceptions=True)

        cleanup_task = asyncio.create_task(
            finish_cleanup(),
            name="capswriter-http-client-disconnect-cleanup",
        )
        await _await_cleanup_task(cleanup_task)


async def _parse_transcription_request(
    request: Request,
    *,
    max_upload_bytes: int,
) -> ParsedTranscriptionRequest:
    try:
        form = await parse_multipart_form(
            request,
            max_files=MAX_MULTIPART_FILES,
            max_fields=MAX_MULTIPART_FIELDS,
        )
    except RequestBodyTooLargeError:
        raise
    except ClientDisconnect:
        raise
    except MultipartParseError as exc:
        raise _request_error(
            "Invalid multipart form",
            headers={"Connection": "close"},
        ) from exc
    except MultiPartException as exc:
        raise _request_error(
            f"Invalid multipart form: {exc.message}",
            headers={"Connection": "close"},
        ) from exc

    try:
        _validate_form_fields(form)
        file = _uploaded_file(form)
        model = _text_form_value(form, "model", None)
        if model is None:
            raise _request_error("Missing required field: 'model'")
        _validate_model(model)
        temperature = _validate_temperature(
            _text_form_value(form, "temperature", "0.0")
        )
        response_format = _validate_response_format(
            _text_form_value(form, "response_format", "json") or ""
        )
        _validate_stream_value(_text_form_value(form, "stream", None))
        context = normalize_prompt_context(_text_form_value(form, "prompt", None))
        try:
            language_hint = normalize_language_hint(
                _text_form_value(form, "language", None)
            )
        except ValueError as exc:
            raise _request_error(str(exc)) from exc

        timestamp_values = _text_form_values(form, "timestamp_granularities[]")
        timestamp_values.extend(
            _text_form_values(form, "timestamp_granularities")
        )
        timestamp_granularities = _validate_timestamp_granularities(
            timestamp_values,
            response_format,
        )
        audio_bytes = await read_upload_limited(file, max_upload_bytes)
        return ParsedTranscriptionRequest(
            audio_bytes=audio_bytes,
            temperature=temperature,
            response_format=response_format,
            context=context,
            language_hint=language_hint,
            timestamp_granularities=timestamp_granularities,
        )
    finally:
        # Directly close the single bounded spool without another await point;
        # task cancellation therefore cannot interrupt the cleanup itself.
        close_form_files(form)


def _check_auth(
    authorization: Optional[str],
    *,
    close_connection: bool = False,
) -> None:
    """空字串的 api_key 視為關閉認證 (預設本地使用場景)。"""
    api_key = getattr(Config, "http_api_key", "") or ""
    if not auth_enabled(api_key):
        return
    headers = {"Connection": "close"} if close_connection else None
    if extract_bearer_token(authorization) is None:
        raise HTTPException(
            401,
            "Missing or invalid Authorization header",
            headers=headers,
        )
    if not bearer_token_matches(authorization, api_key):
        raise HTTPException(401, "Invalid API key", headers=headers)


def _check_browser_origin(request: Request, allowed_origins: list[str]) -> None:
    """Reject disallowed browser origins before reading a simple POST body."""
    origins = request.headers.getlist("origin")
    if not origins:
        return
    if len(origins) != 1:
        raise _request_error(
            "Invalid Origin header",
            status_code=403,
            headers={"Connection": "close"},
        )
    raw_origin = origins[0].strip()
    try:
        origin = normalize_http_origin(raw_origin)
    except ConfigError as exc:
        raise _request_error(
            "Origin is not allowed",
            status_code=403,
            headers={"Connection": "close"},
        ) from exc
    if raw_origin != origin:
        raise _request_error(
            "Origin is not allowed",
            status_code=403,
            headers={"Connection": "close"},
        )
    if "*" not in allowed_origins and origin not in allowed_origins:
        raise _request_error(
            "Origin is not allowed",
            status_code=403,
            headers={"Connection": "close"},
        )


def _check_recognizer_available() -> None:
    """Reject new bodies immediately once the worker enters fail-stop."""
    checker = getattr(task_router, "recognizer_process_alive", None)
    if checker is None:
        # Lightweight test routers predating the liveness surface remain valid;
        # the production TaskRouter always provides this check.
        return
    try:
        available = bool(checker())
    except (AssertionError, OSError, RuntimeError, ValueError):
        available = False
    if not available:
        raise HTTPException(
            503,
            "Recognition service is unavailable",
            headers={"Connection": "close", "Retry-After": "5"},
        )


def _split_and_submit(
    task_id: str,
    pcm: bytes,
    *,
    context: str = "",
    language: str = "auto",
    cancel_event: threading.Event | None = None,
    log_transcript: bool = False,
    deadline_monotonic: float | None = None,
    socket_is_live=None,
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
    specs = iter(
        iter_transcription_task_specs(
            task_id=task_id,
            socket_id=socket_id,
            pcm=pcm,
            time_start=time_start,
            context=context,
            language=language,
            log_transcript=log_transcript,
            deadline_monotonic=deadline_monotonic,
        )
    )
    while cancel_event is None or not cancel_event.is_set():
        try:
            spec = next(specs)
        except StopIteration:
            break
        if cancel_event is not None and cancel_event.is_set():
            break
        task = Task(
            **spec,
            time_submit=time.time(),
        )
        while cancel_event is None or not cancel_event.is_set():
            if socket_is_live is not None:
                try:
                    if not socket_is_live():
                        return
                except (
                    BrokenPipeError,
                    EOFError,
                    OSError,
                    RuntimeError,
                    ValueError,
                ):
                    return
            put_timeout = QUEUE_PUT_RETRY_SECONDS
            if deadline_monotonic is not None:
                remaining = deadline_monotonic - time.monotonic()
                if remaining <= 0:
                    raise asyncio.TimeoutError
                put_timeout = min(put_timeout, remaining)
            try:
                queue_in.put(task, timeout=put_timeout)
                break
            except queue.Full:
                continue


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
            "OpenAI-compatible whisper-1 file-transcription subset, backed by "
            "local CapsWriter-Offline recognition (offline, private)."
        ),
    )

    max_upload_bytes, max_upload_mb = upload_limit_bytes(
        getattr(Config, "http_api_max_upload_mb", 100)
    )
    max_body_bytes = request_body_limit_bytes(max_upload_bytes)
    max_audio_bytes, max_audio_seconds = audio_limit_bytes(
        getattr(Config, "http_api_max_audio_seconds", 3600.0)
    )
    max_concurrent_requests = max(
        1,
        int(getattr(Config, "http_api_max_concurrent_requests", 2)),
    )
    max_pending_requests = max(
        0,
        int(getattr(Config, "http_api_max_pending_requests", 4)),
    )
    transcription_admission = AdmissionController(
        max_concurrent_requests,
        max_pending_requests,
    )
    app.state.transcription_admission = transcription_admission

    # This middleware counts raw ASGI body bytes, including chunked requests,
    # before Starlette's multipart parser can grow its temporary spool without
    # bound.  Authentication and admission still run before the first receive().
    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_body_bytes=max_body_bytes,
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

    async def client_disconnect_exception_handler(_request, _exc):
        # The peer cannot receive this response; 499 prevents the framework's
        # generic 500 path in direct ASGI hosts and keeps disconnects distinct.
        return Response(status_code=499)

    app.add_exception_handler(
        ClientDisconnect,
        client_disconnect_exception_handler,
    )

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
            recognizer_process_alive=task_router.recognizer_process_alive(),
            ffmpeg_available=shutil.which("ffmpeg") is not None,
            auth_enabled=readiness_auth_enabled(getattr(Config, "http_api_key", "")),
            max_upload_mb=max_upload_mb,
            max_audio_seconds=max_audio_seconds,
            task_timeout=float(getattr(Config, "http_api_task_timeout", 600.0)),
            max_concurrent_requests=max_concurrent_requests,
            max_pending_requests=max_pending_requests,
            max_websocket_connections=int(
                getattr(Config, "max_websocket_connections", 8)
            ),
            max_websocket_task_seconds=float(
                getattr(Config, "max_websocket_task_seconds", 3600.0)
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
                    "id": OPENAI_TRANSCRIPTION_MODEL,
                    "object": "model",
                    "owned_by": "capswriter-offline",
                    "created": 0,
                }
            ],
        }

    @app.post(
        "/v1/audio/transcriptions",
        openapi_extra=TRANSCRIPTION_OPENAPI_EXTRA,
    )
    async def transcriptions(
        request: Request,
        authorization: Optional[str] = Header(None),
    ):
        request_start = time.monotonic()
        task_id = uuid.uuid4().hex

        # No body dependency is declared on this route.  Consequently these
        # checks happen before FastAPI/Starlette calls receive(), which matters
        # for Expect: 100-continue clients and hostile unauthenticated uploads.
        _check_auth(authorization, close_connection=True)
        _check_browser_origin(request, cors_origins)
        _check_recognizer_available()
        _validate_multipart_headers(request, max_body_bytes)

        timeout = task_timeout_seconds(
            getattr(Config, "http_api_task_timeout", 600.0)
        )
        deadline = asyncio.get_running_loop().time() + timeout
        worker_deadline_monotonic = time.monotonic() + timeout

        async def run_admitted() -> Response:
            async with _admission_slot_or_disconnect(
                request,
                transcription_admission,
            ):
                # Parsing is inside both admission and the end-to-end deadline.
                # The owned parser closes partial spools even when disconnect or
                # cancellation happens before a FormData context can be returned.
                try:
                    parsed = await _parse_transcription_request(
                        request,
                        max_upload_bytes=max_upload_bytes,
                    )
                except UploadTooLargeError:
                    logger.warning(
                        f"[HTTP] task={task_id[:8]} upload rejected: "
                        f">{max_upload_mb} MB"
                    )
                    raise HTTPException(
                        413,
                        f"File too large (>{max_upload_mb} MB)",
                    )

                if not parsed.audio_bytes:
                    raise HTTPException(400, "Empty file")

                try:
                    pcm = await decode_to_pcm(
                        parsed.audio_bytes,
                        timeout=_remaining_seconds(deadline),
                        max_output_bytes=max_audio_bytes,
                    )
                except FFmpegNotFoundError as exc:
                    log_internal_exception(
                        logger,
                        "[HTTP] ffmpeg not found; install ffmpeg",
                        exc,
                        log_sensitive_text=_log_sensitive_text_enabled(),
                    )
                    raise HTTPException(500, "Server audio decoder is unavailable")
                except AudioDecodeTimeoutError:
                    raise asyncio.TimeoutError
                except AudioTooLongError as exc:
                    log_internal_exception(
                        logger,
                        f"[HTTP] task={task_id[:8]} audio rejected as too long",
                        exc,
                        log_sensitive_text=_log_sensitive_text_enabled(),
                        level="warning",
                    )
                    raise HTTPException(
                        413,
                        f"Audio is too long (>{max_audio_seconds:g} seconds)",
                    )
                except AudioDecodeError as exc:
                    log_internal_exception(
                        logger,
                        f"[HTTP] task={task_id[:8]} audio decode failed",
                        exc,
                        log_sensitive_text=_log_sensitive_text_enabled(),
                        level="warning",
                    )
                    raise HTTPException(400, "Audio could not be decoded")

                duration = AudioFormat.bytes_to_seconds(len(pcm))
                if duration < 0.05:
                    raise HTTPException(400, "Audio too short to transcribe")

                logger.info(
                    f"[HTTP] task={task_id[:8]} duration={duration:.2f}s "
                    f"fmt={parsed.response_format} lang={parsed.language_hint} "
                    f"bytes={len(parsed.audio_bytes)}"
                )

                try:
                    # register() mutates router state before appending the
                    # synthetic socket, so even partial failure shares this
                    # unconditional cleanup path.
                    future = task_router.register(task_id)
                    submission_cancel = threading.Event()
                    submission_task = asyncio.create_task(
                        asyncio.to_thread(
                            _split_and_submit,
                            task_id,
                            pcm,
                            context=parsed.context,
                            language=parsed.language_hint,
                            cancel_event=submission_cancel,
                            log_transcript=_log_sensitive_text_enabled(),
                            deadline_monotonic=worker_deadline_monotonic,
                            socket_is_live=lambda: task_router.socket_is_active(
                                task_id
                            ),
                        )
                    )
                    try:
                        # Submission can itself wait on bounded cross-process
                        # capacity, so observe the peer throughout this phase.
                        await _await_result_or_disconnect(
                            request,
                            submission_task,
                            timeout=_remaining_seconds(deadline),
                        )
                    except BaseException:
                        submission_cancel.set()
                        try:
                            await asyncio.shield(submission_task)
                        except BaseException:
                            pass
                        raise
                    log_prompt_context(
                        logger,
                        task_id,
                        parsed.context,
                        log_sensitive_text=_log_sensitive_text_enabled(),
                    )
                    result = await _await_result_or_disconnect(
                        request,
                        future,
                        timeout=_remaining_seconds(deadline),
                    )
                    error_code = getattr(result, "error_code", None)
                    if error_code:
                        logger.error(
                            f"[HTTP] task={task_id[:8]} recognition worker "
                            f"failed: code={error_code!r}"
                        )
                        raise HTTPException(500, "Recognition failed")

                    body, media_type = format_response(
                        result,
                        parsed.response_format,
                        language=parsed.language_hint,
                        temperature=parsed.temperature,
                        timestamp_granularities=(
                            parsed.timestamp_granularities
                        ),
                    )
                    text = result.text_accu or result.text
                    delay = max(0.0, time.monotonic() - request_start)
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
                except (
                    asyncio.CancelledError,
                    asyncio.TimeoutError,
                    ClientDisconnect,
                    HTTPException,
                ):
                    raise
                except Exception as exc:
                    log_internal_exception(
                        logger,
                        f"[HTTP] task={task_id[:8]} recognition failed",
                        exc,
                        log_sensitive_text=_log_sensitive_text_enabled(),
                    )
                    raise HTTPException(500, "Recognition failed") from None
                finally:
                    if "submission_cancel" in locals():
                        submission_cancel.set()
                    try:
                        task_router.cancel(task_id)
                    except Exception as cleanup_error:
                        log_internal_exception(
                            logger,
                            f"[HTTP] task={task_id[:8]} router cleanup failed",
                            cleanup_error,
                            log_sensitive_text=_log_sensitive_text_enabled(),
                        )

        try:
            # One deadline covers queue wait, multipart ingestion, ffmpeg,
            # submission, inference, response formatting, and cleanup.
            return await asyncio.wait_for(run_admitted(), timeout=timeout)
        except AdmissionQueueFullError:
            logger.warning(
                f"[HTTP] task={task_id[:8]} rejected: admission queue full"
            )
            raise HTTPException(
                429,
                "Server is busy; retry later",
                headers={"Retry-After": "1", "Connection": "close"},
            )
        except RequestBodyTooLargeError:
            # The outer ASGI guard owns the 413 response because it can stop a
            # chunked body before the multipart parser allocates further.
            raise
        except ClientDisconnect:
            logger.info(f"[HTTP] task={task_id[:8]} client disconnected")
            raise
        except asyncio.CancelledError:
            logger.warning(f"[HTTP] task={task_id[:8]} request cancelled")
            raise
        except asyncio.TimeoutError:
            logger.error(
                f"[HTTP] task={task_id[:8]} exceeded end-to-end deadline "
                f"of {timeout:g}s"
            )
            raise HTTPException(
                504,
                "Recognition timeout",
                headers={"Connection": "close"},
            )
        except HTTPException:
            raise
        except Exception as exc:
            log_internal_exception(
                logger,
                f"[HTTP] task={task_id[:8]} unexpected error",
                exc,
                log_sensitive_text=_log_sensitive_text_enabled(),
            )
            raise HTTPException(500, "Recognition failed") from None

    @app.post("/v1/audio/translations", include_in_schema=False)
    async def translations(authorization: Optional[str] = Header(None)):
        _check_auth(authorization, close_connection=True)
        # CapsWriter 本地模型不做語種翻譯, 明確返回 501
        raise HTTPException(
            501,
            "translations endpoint is not implemented; CapsWriter only transcribes",
            headers={"Connection": "close"},
        )

    return app
