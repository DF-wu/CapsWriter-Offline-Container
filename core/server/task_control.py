# coding: utf-8
"""Stable internal control messages shared by ingress and the worker."""

WEBSOCKET_TASK_LIMIT_COMMAND = "reject_websocket_task_audio_limit"
WEBSOCKET_TASK_LIMIT_ERROR_CODE = "websocket_task_audio_limit_exceeded"
WEBSOCKET_TASK_LIMIT_ERROR_MESSAGE = (
    "WebSocket task audio exceeds the configured server limit."
)
