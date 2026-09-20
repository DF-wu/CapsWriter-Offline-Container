# coding: utf-8
"""Shared bounds for audio work crossing the server process boundary."""

from __future__ import annotations


# One task carries at most 60 seconds plus 4 seconds of overlap as float32,
# 16 kHz mono PCM: 64 * 16_000 * 4 = 4_096_000 bytes.
MAX_TASK_AUDIO_SECONDS = 64.0
MAX_TASK_AUDIO_BYTES = 4_096_000

# The multiprocessing queue and child-local fair buffer are independently
# bounded. Eight slots apiece cap their logical PCM payload at 65,536,000 bytes.
INPUT_QUEUE_MAX_TASKS = 8
WORKER_BUFFER_MAX_TASKS = 8
OUTPUT_QUEUE_MAX_RESULTS = 8
QUEUE_PUT_RETRY_SECONDS = 0.05

RESULT_QUEUE_GET_TIMEOUT_SECONDS = 0.1
WEBSOCKET_SEND_TIMEOUT_SECONDS = 5.0
WEBSOCKET_CLOSE_TIMEOUT_SECONDS = 1.0
MAX_PENDING_RESULTS_PER_WEBSOCKET = 8

# 6 MiB admits a base64-encoded maximum-size task (5,461,336 bytes) plus JSON
# metadata, while preventing a single WebSocket frame from being unbounded.
WEBSOCKET_MAX_MESSAGE_BYTES = 6 * 1024 * 1024
WEBSOCKET_MAX_QUEUED_MESSAGES = 1
