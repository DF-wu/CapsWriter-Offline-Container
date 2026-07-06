# coding: utf-8
"""
fork_server.http_api — OpenAI Whisper API 相容的 REST 端點

模組:
- api:                FastAPI app + 4 endpoints
- task_router:        HTTP task ↔ asyncio.Future routing + 與 sockets_id 整合
- audio_decoder:      ffmpeg subprocess → 16kHz/float32/mono PCM
- errors:             OpenAI-style JSON error payload handlers
- readiness:          /ready payload and status code helper
- runtime_config:     CAPSWRITER_HTTP_API_* validation and normalization
- openai_formatter:   Result → json/text/srt/vtt/verbose_json
- ws_send_with_http:  上游 ws_send 的 fork 版, 先試 HTTP future, 否則走原 ws 廣播
- serve:              uvicorn cotask, 與上游 ws_send 並行於同一 asyncio loop
"""
