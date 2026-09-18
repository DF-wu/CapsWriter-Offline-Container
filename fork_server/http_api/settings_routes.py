"""Authenticated, bounded management routes isolated from inference imports."""
from __future__ import annotations
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from fork_server.settings import ConfigError, RevisionConflict, decode_settings_json
from .auth import auth_enabled, bearer_token_matches

MAX_PATCH_BYTES = 16_384
DISABLED_REASON = (
    "Settings management is disabled; enable CAPSWRITER_SETTINGS_ENABLE and "
    "configure CAPSWRITER_SETTINGS_PATH with an HTTP API key."
)


def _error(status, message, fields=None):
    detail = {"message": message, "type": "invalid_request_error", "param": None, "code": None}
    if fields:
        detail["fields"] = fields
    return JSONResponse(status_code=status, content={"error": detail},
                        headers={"WWW-Authenticate": "Bearer"} if status == 401 else None)


def settings_router(config):
    router = APIRouter()

    def access(request):
        store = getattr(config, "settings_store", None)
        if store is None or not store.enabled:
            return None, None
        key = getattr(config, "http_api_key", "")
        if not auth_enabled(key):
            return None, _error(503, "Settings management requires an HTTP API key")
        if not bearer_token_matches(request.headers.get("authorization"), key):
            return None, _error(401, "A valid Bearer API key is required")
        return store, None

    @router.get("/v1/settings")
    async def get_settings(request: Request):
        store, error = access(request)
        if error is not None:
            return error
        if store is None:
            return {"enabled": False, "reason": DISABLED_REASON, "fields": [],
                    "restart_required": False, "revision": None}
        try:
            return await run_in_threadpool(store.snapshot)
        except (ConfigError, OSError):
            return _error(503, "Unable to read server settings; check the server configuration")

    @router.patch("/v1/settings")
    async def patch_settings(request: Request):
        store, error = access(request)
        if error is not None:
            return error
        if store is None:
            return _error(403, DISABLED_REASON)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_PATCH_BYTES:
                return _error(413, "Settings request exceeds size limit")
        try:
            data = decode_settings_json(body)
            if not isinstance(data, dict) or set(data) != {"values", "revision"}:
                raise ConfigError("Request must contain values and revision only")
            return await run_in_threadpool(store.update, data["values"], data["revision"])
        except RevisionConflict:
            return _error(409, "Settings changed; reload before saving")
        except (ValueError, UnicodeError) as exc:
            from fork_server.settings import FIELD_MAP
            message = str(exc) if isinstance(exc, ConfigError) else "Request must be valid JSON"
            key = message.split(":", 1)[0]
            return _error(422, message, {key: message} if key in FIELD_MAP else None)
        except OSError:
            return _error(503, "Unable to save server settings; check the settings volume permissions")

    return router
