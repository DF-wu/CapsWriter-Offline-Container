# coding: utf-8
"""ASGI receive guard for bounded multipart request bodies."""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from .errors import openai_error_payload
from .limits import RequestBodyTooLargeError


class RequestBodyLimitMiddleware:
    """Stop reading a transcription body once its configured cap is exceeded."""

    def __init__(
        self,
        app: Callable[..., Awaitable[None]],
        *,
        max_body_bytes: int,
        path: str = "/v1/audio/transcriptions",
    ) -> None:
        self.app = app
        self.max_body_bytes = max(1, int(max_body_bytes))
        self.path = path

    async def __call__(self, scope: dict[str, Any], receive, send) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != self.path
        ):
            await self.app(scope, receive, send)
            return

        received = 0
        response_started = False

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    raise RequestBodyTooLargeError
            return message

        async def tracking_send(message):
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except RequestBodyTooLargeError:
            if response_started:
                raise
            body = json.dumps(
                openai_error_payload(
                    message="Request body is too large",
                    status_code=413,
                    param="file",
                    code="request_too_large",
                ),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            await send(
                {
                    "type": "http.response.start",
                    "status": 413,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode("ascii")),
                        (b"connection", b"close"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
