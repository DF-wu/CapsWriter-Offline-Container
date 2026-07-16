"""Bounded, cancellable HTTP client for CapsWriter's OpenAI-compatible API."""

from __future__ import annotations

import asyncio
import json
import math
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterable
from urllib.parse import urlsplit, urlunsplit

import httpx


DEFAULT_BASE_URL: Final = "http://127.0.0.1:6017"
DEFAULT_MODEL: Final = "whisper-1"
DEFAULT_DIAGNOSTIC_TIMEOUT: Final = 10.0
DEFAULT_TRANSCRIPTION_TIMEOUT: Final = 600.0
DEFAULT_MAX_RESPONSE_BYTES: Final = 16 * 1024 * 1024
MAX_TIMEOUT_SECONDS: Final = 900.0
MAX_RESPONSE_BYTES: Final = 64 * 1024 * 1024
MAX_ERROR_PREVIEW_CHARS: Final = 500
MAX_ERROR_SOURCE_SCAN_CHARS: Final = MAX_ERROR_PREVIEW_CHARS * 4
MAX_PROMPT_CHARS: Final = 2048
MAX_LANGUAGE_CHARS: Final = 32
MAX_MODEL_CHARS: Final = 128
RESPONSE_FORMATS: Final = ("text", "json", "verbose_json", "srt", "vtt")


class ApiError(RuntimeError):
    """A safe-to-display API failure with optional HTTP status."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        self.status = status
        super().__init__(message)


class ResponseTooLarge(ApiError):
    """The peer exceeded the configured response-body limit."""


@dataclass(frozen=True)
class EndpointResult:
    status: int
    payload: dict[str, Any]

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    response_format: str
    status: int


def normalize_base_url(value: str) -> str:
    """Validate an HTTP(S) API root and remove only a trailing /v1."""

    candidate = (value or DEFAULT_BASE_URL).strip().rstrip("/")
    parts = urlsplit(candidate)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("API root must be an absolute http:// or https:// URL")
    if parts.username or parts.password:
        raise ValueError("API root must not contain a username or password")
    if parts.query or parts.fragment:
        raise ValueError("API root must not contain a query string or fragment")
    try:
        parts.port
    except ValueError as exc:
        raise ValueError("API root has an invalid port") from exc
    path = parts.path.rstrip("/")
    if path == "/v1":
        path = ""
    elif path.endswith("/v1"):
        path = path[:-3].rstrip("/")
    return urlunsplit(parts._replace(path=path, query="", fragment=""))


def _bounded_positive(value: float, *, name: str, upper: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    if value > upper:
        raise ValueError(f"{name} must not exceed {upper:g}")
    return value


def _redact_truncated_secret_prefix(value: str, secret: str) -> str:
    """Redact a reflected key cut at a bounded source-scan boundary."""

    if not secret or not value or value.endswith(secret):
        return value
    for length in range(min(len(secret) - 1, len(value)), 0, -1):
        if value.endswith(secret[:length]):
            return f"{value[:-length]}[REDACTED]"
    return value


def _compact_error_text(value: str, secret: str = "") -> str:
    source_limit = MAX_ERROR_SOURCE_SCAN_CHARS + len(secret)
    source_was_truncated = len(value) > source_limit
    bounded_source = value[:source_limit]
    if source_was_truncated:
        bounded_source = _redact_truncated_secret_prefix(
            bounded_source,
            secret,
        )
    bounded_source = redact_secret(bounded_source, secret)
    printable = "".join(
        character if character.isprintable() else " "
        for character in bounded_source
    )
    message = " ".join(printable.split())
    if source_was_truncated or len(message) > MAX_ERROR_PREVIEW_CHARS:
        return f"{message[: MAX_ERROR_PREVIEW_CHARS - 1].rstrip()}…"
    return message


def _safe_error_message(body: bytes, secret: str = "") -> str:
    decoded = body.decode("utf-8", errors="replace").strip()
    if not decoded:
        return "request failed without an error body"
    try:
        payload = json.loads(decoded)
    except json.JSONDecodeError:
        message = decoded
    else:
        message = decoded
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict) and error.get("message"):
                message = str(error["message"])
            elif payload.get("detail"):
                message = str(payload["detail"])
    return _compact_error_text(message, secret)


def redact_secret(message: str, secret: str) -> str:
    """Remove the in-memory token if a peer unexpectedly reflects it."""

    if secret:
        return message.replace(secret, "[REDACTED]")
    return message


def _monotonic() -> float:
    return asyncio.get_running_loop().time()


def _remaining_seconds(deadline: float) -> float:
    remaining = deadline - _monotonic()
    if remaining <= 0:
        raise asyncio.TimeoutError
    return remaining


class CapsWriterApi:
    """One-request-per-client API surface with bounded streaming reads.

    Each operation owns its ``AsyncClient``. Cancelling the coroutine exits the
    stream context immediately, closing the underlying connection rather than
    leaving a blocking worker thread behind.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        *,
        diagnostic_timeout: float = DEFAULT_DIAGNOSTIC_TIMEOUT,
        transcription_timeout: float = DEFAULT_TRANSCRIPTION_TIMEOUT,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = normalize_base_url(base_url)
        self._api_key = api_key.strip()
        self.diagnostic_timeout = _bounded_positive(
            float(diagnostic_timeout), name="diagnostic timeout", upper=MAX_TIMEOUT_SECONDS
        )
        self.transcription_timeout = _bounded_positive(
            float(transcription_timeout), name="transcription timeout", upper=MAX_TIMEOUT_SECONDS
        )
        if not isinstance(max_response_bytes, int) or max_response_bytes <= 0:
            raise ValueError("response limit must be a positive integer")
        if max_response_bytes > MAX_RESPONSE_BYTES:
            raise ValueError(f"response limit must not exceed {MAX_RESPONSE_BYTES} bytes")
        self.max_response_bytes = max_response_bytes
        self._transport = transport

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            return {}
        return {"Authorization": f"Bearer {self._api_key}"}

    @staticmethod
    def _timeout(seconds: float) -> httpx.Timeout:
        return httpx.Timeout(seconds, connect=min(10.0, seconds), pool=min(10.0, seconds))

    async def _bounded_body(
        self,
        response: httpx.Response,
        *,
        deadline: float,
    ) -> bytes:
        declared = response.headers.get("content-length")
        if declared:
            try:
                declared_size = int(declared)
            except ValueError:
                declared_size = -1
            if declared_size > self.max_response_bytes:
                raise ResponseTooLarge(
                    f"response body exceeds {self.max_response_bytes} bytes",
                    status=response.status_code,
                )
        body = bytearray()
        _remaining_seconds(deadline)
        async for chunk in response.aiter_bytes():
            body.extend(chunk)
            if len(body) > self.max_response_bytes:
                raise ResponseTooLarge(
                    f"response body exceeds {self.max_response_bytes} bytes",
                    status=response.status_code,
                )
            _remaining_seconds(deadline)
        return bytes(body)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: float,
        data: dict[str, str] | None = None,
        files: dict[str, tuple[str, Any, str]] | None = None,
        allowed_statuses: Iterable[int] = (),
    ) -> tuple[int, bytes, str]:
        allowed = set(allowed_statuses)
        deadline = _monotonic() + timeout

        async def perform_request() -> tuple[int, bytes, str]:
            async with httpx.AsyncClient(
                timeout=self._timeout(timeout),
                follow_redirects=False,
                trust_env=False,
                transport=self._transport,
            ) as client:
                async with client.stream(
                    method,
                    self._url(path),
                    headers=self._headers(),
                    data=data,
                    files=files,
                ) as response:
                    body = await self._bounded_body(
                        response,
                        deadline=deadline,
                    )
                    content_type = response.headers.get("content-type", "")
                    return response.status_code, body, content_type

        try:
            status, body, content_type = await asyncio.wait_for(
                perform_request(),
                timeout=_remaining_seconds(deadline),
            )
        except ResponseTooLarge:
            raise
        except asyncio.TimeoutError as exc:
            raise ApiError(f"request timed out after {timeout:g} seconds") from exc
        except httpx.TimeoutException as exc:
            raise ApiError(f"request timed out after {timeout:g} seconds") from exc
        except httpx.HTTPError as exc:
            message = _compact_error_text(str(exc), self._api_key)
            raise ApiError(message or "HTTP request failed") from exc

        if not (200 <= status < 300) and status not in allowed:
            message = _safe_error_message(body, self._api_key)
            raise ApiError(f"HTTP {status}: {message}", status=status)
        return status, body, content_type

    async def _json_endpoint(
        self, path: str, *, allowed_statuses: Iterable[int] = ()
    ) -> EndpointResult:
        status, body, _content_type = await self._request(
            "GET",
            path,
            timeout=self.diagnostic_timeout,
            allowed_statuses=allowed_statuses,
        )
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApiError(f"HTTP {status}: expected a JSON response from {path}", status=status) from exc
        if not isinstance(payload, dict):
            raise ApiError(f"HTTP {status}: expected a JSON object from {path}", status=status)
        return EndpointResult(status=status, payload=payload)

    async def health(self) -> EndpointResult:
        return await self._json_endpoint("/health")

    async def ready(self) -> EndpointResult:
        return await self._json_endpoint("/ready", allowed_statuses=(503,))

    async def models(self) -> EndpointResult:
        return await self._json_endpoint("/v1/models")

    async def transcribe(
        self,
        audio_path: Path,
        *,
        response_format: str,
        model: str = DEFAULT_MODEL,
        language: str = "",
        prompt: str = "",
    ) -> TranscriptionResult:
        source = audio_path.expanduser()
        if not source.is_file():
            raise ValueError(f"audio file does not exist: {source}")
        if response_format not in RESPONSE_FORMATS:
            raise ValueError(f"unsupported response format: {response_format}")
        model = model.strip() or DEFAULT_MODEL
        language = language.strip()
        prompt = prompt.replace("\r\n", "\n").replace("\r", "\n")
        if len(model) > MAX_MODEL_CHARS:
            raise ValueError(f"model must not exceed {MAX_MODEL_CHARS} characters")
        if len(language) > MAX_LANGUAGE_CHARS:
            raise ValueError(f"language must not exceed {MAX_LANGUAGE_CHARS} characters")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError(f"prompt must not exceed {MAX_PROMPT_CHARS} characters")

        mime_type = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
        with source.open("rb") as audio:
            status, body, _content_type = await self._request(
                "POST",
                "/v1/audio/transcriptions",
                timeout=self.transcription_timeout,
                data={
                    "model": model,
                    "response_format": response_format,
                    "language": language,
                    "prompt": prompt,
                },
                files={"file": (source.name, audio, mime_type)},
            )

        if response_format in {"text", "srt", "vtt"}:
            rendered = body.decode("utf-8", errors="replace")
        else:
            try:
                payload = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ApiError(
                    f"HTTP {status}: expected JSON transcription output", status=status
                ) from exc
            rendered = json.dumps(payload, ensure_ascii=False, indent=2)
        return TranscriptionResult(rendered, response_format, status)
