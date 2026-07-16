# coding: utf-8
"""OpenAI-style HTTP error payload helpers."""

from __future__ import annotations

from typing import Any

from .limits import REQUEST_BODY_TOO_LARGE_MESSAGE


ERROR_TYPE_BY_STATUS = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "invalid_request_error",
    413: "invalid_request_error",
    422: "invalid_request_error",
    429: "rate_limit_error",
    500: "server_error",
    501: "not_implemented_error",
    503: "server_error",
    504: "timeout_error",
}


def error_type_for_status(status_code: int) -> str:
    if status_code >= 500:
        return ERROR_TYPE_BY_STATUS.get(status_code, "server_error")
    return ERROR_TYPE_BY_STATUS.get(status_code, "invalid_request_error")


def openai_error_payload(
    *,
    message: str,
    status_code: int,
    error_type: str | None = None,
    param: str | None = None,
    code: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Return a compact OpenAI-compatible error object."""
    return {
        "error": {
            "message": message,
            "type": error_type or error_type_for_status(status_code),
            "param": param,
            "code": code,
        }
    }


def _string_detail(detail: Any) -> str:
    if isinstance(detail, str):
        return detail
    if detail is None:
        return "Request failed"
    return str(detail)


def validation_error_message(exc: Any) -> tuple[str, str | None]:
    errors = exc.errors()
    if not errors:
        return "Invalid request", None
    first = errors[0]
    loc = [str(item) for item in first.get("loc", ()) if item not in ("body", "query")]
    param = ".".join(loc) if loc else None
    msg = str(first.get("msg") or "Invalid value")
    if param:
        return f"Invalid request: {param}: {msg}", param
    return f"Invalid request: {msg}", None


async def http_exception_handler(
    request: Any,
    exc: Any,
) -> Any:
    from fastapi.responses import JSONResponse

    del request
    detail = _string_detail(exc.detail)
    if exc.status_code == 400 and detail == REQUEST_BODY_TOO_LARGE_MESSAGE:
        return JSONResponse(
            status_code=413,
            content=openai_error_payload(
                message="Request body is too large",
                status_code=413,
                param="file",
                code="request_too_large",
            ),
            headers={"Connection": "close"},
        )
    return JSONResponse(
        status_code=exc.status_code,
        content=openai_error_payload(
            message=detail,
            status_code=exc.status_code,
        ),
        headers=getattr(exc, "headers", None),
    )


async def validation_exception_handler(
    request: Any,
    exc: Any,
) -> Any:
    from fastapi.responses import JSONResponse

    del request
    message, param = validation_error_message(exc)
    return JSONResponse(
        status_code=422,
        content=openai_error_payload(
            message=message,
            status_code=422,
            param=param,
        ),
    )
