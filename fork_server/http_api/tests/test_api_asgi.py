# coding: utf-8
"""Real multipart/ASGI contract tests when server dependencies are installed."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack, asynccontextmanager
import importlib.util
import json
import logging
import queue
import threading
import time
from unittest.mock import AsyncMock, patch
import unittest


SERVER_DEPS_AVAILABLE = (
    importlib.util.find_spec("fastapi") is not None
    and importlib.util.find_spec("multipart") is not None
)


def _multipart_body(fields: list[tuple[str, str]]) -> tuple[bytes, str]:
    boundary = "----CapsWriterContractBoundary"
    chunks = [
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="file"; filename="sample.wav"\r\n',
        b"Content-Type: audio/wav\r\n\r\n",
        b"RIFF-test-audio",
        b"\r\n",
    ]
    for name, value in fields:
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode(),
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), boundary


async def _asgi_post(
    app,
    body: bytes,
    boundary: str,
    *,
    authorization: str | None = None,
    content_length: int | None = None,
    origin: str | None = None,
):
    messages = []
    delivered = False
    receive_calls = 0

    async def receive():
        nonlocal delivered, receive_calls
        receive_calls += 1
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)

    headers = [
        (b"host", b"testserver"),
        (b"content-type", f"multipart/form-data; boundary={boundary}".encode()),
        (
            b"content-length",
            str(len(body) if content_length is None else content_length).encode(),
        ),
    ]
    if authorization is not None:
        headers.append((b"authorization", authorization.encode()))
    if origin is not None:
        headers.append((b"origin", origin.encode()))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/audio/transcriptions",
        "raw_path": b"/v1/audio/transcriptions",
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "state": {},
    }
    await app(scope, receive, send)
    start = next(message for message in messages if message["type"] == "http.response.start")
    response_body = b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )
    return start["status"], dict(start["headers"]), response_body, receive_calls


@unittest.skipUnless(SERVER_DEPS_AVAILABLE, "FastAPI server dependencies are not installed")
class TranscriptionAsgiContractTest(unittest.TestCase):
    @staticmethod
    def _capture_api_logs(logger):
        records = []

        class CaptureHandler(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = CaptureHandler()
        logger.addHandler(handler)
        return handler, records

    def test_split_submitter_stops_enqueuing_after_cancellation(self) -> None:
        from fork_server.http_api import api

        cancel = threading.Event()

        class CancellingQueue:
            def __init__(self):
                self.items = []

            def put(self, item, *, timeout):
                self.assert_timeout = timeout
                self.items.append(item)
                cancel.set()

        queue = CancellingQueue()
        router = type(
            "FakeRouter",
            (),
            {
                "queue_in": queue,
                "synthetic_socket_id": staticmethod(lambda task_id: f"http:{task_id}"),
            },
        )()
        specs = [
            {
                "type": "file",
                "data": bytes([index]),
                "offset": float(index),
                "overlap": 0.0,
                "task_id": "task",
                "socket_id": "http:task",
                "is_final": index == 2,
                "time_start": 1.0,
                "context": "",
                "language": "auto",
            }
            for index in range(3)
        ]

        with (
            patch.object(api, "task_router", router),
            patch.object(api, "iter_transcription_task_specs", return_value=specs),
        ):
            api._split_and_submit("task", b"pcm", cancel_event=cancel)

        self.assertEqual(len(queue.items), 1)

    def test_split_submitter_retries_full_queue_and_checks_liveness(self) -> None:
        from fork_server.http_api import api

        class BackpressureQueue:
            def __init__(self):
                self.attempts = 0
                self.items = []

            def put(self, item, *, timeout):
                self.attempts += 1
                self.assert_timeout = timeout
                if self.attempts < 3:
                    raise queue.Full
                self.items.append(item)

        target_queue = BackpressureQueue()
        router = type(
            "FakeRouter",
            (),
            {
                "queue_in": target_queue,
                "synthetic_socket_id": staticmethod(lambda task_id: f"http:{task_id}"),
            },
        )()
        liveness_checks = 0

        def socket_is_live():
            nonlocal liveness_checks
            liveness_checks += 1
            return True

        with patch.object(api, "task_router", router):
            api._split_and_submit(
                "task",
                b"pcm",
                deadline_monotonic=time.monotonic() + 1.0,
                socket_is_live=socket_is_live,
            )

        self.assertEqual(target_queue.attempts, 3)
        self.assertEqual(len(target_queue.items), 1)
        self.assertGreaterEqual(liveness_checks, 3)
        self.assertLessEqual(
            target_queue.assert_timeout,
            api.QUEUE_PUT_RETRY_SECONDS,
        )

    def test_split_submitter_stops_when_synthetic_socket_is_gone(self) -> None:
        from fork_server.http_api import api

        class NeverQueue:
            def put(self, _item, *, timeout):
                self.fail(f"queue put should not run (timeout={timeout})")

        router = type(
            "FakeRouter",
            (),
            {
                "queue_in": NeverQueue(),
                "synthetic_socket_id": staticmethod(lambda task_id: f"http:{task_id}"),
            },
        )()
        with patch.object(api, "task_router", router):
            api._split_and_submit(
                "task",
                b"pcm",
                socket_is_live=lambda: False,
            )

    def test_same_turn_result_wins_disconnect_probe(self) -> None:
        from core.server.schema import Result
        from fork_server.http_api import api

        async def scenario() -> None:
            future = asyncio.get_running_loop().create_future()
            result = Result(
                task_id="same-turn",
                socket_id="http:same-turn",
                type="file",
                text="done",
                is_final=True,
            )

            class SameTurnRequest:
                async def is_disconnected(self):
                    future.set_result(result)
                    return True

            resolved = await api._await_result_or_disconnect(
                SameTurnRequest(),
                future,
                timeout=1.0,
            )
            await asyncio.sleep(0)

            self.assertIs(resolved, result)
            self.assertFalse(
                any(
                    task is not asyncio.current_task()
                    and not task.done()
                    and task.get_name() == "capswriter-http-client-disconnect"
                    for task in asyncio.all_tasks()
                )
            )

        asyncio.run(scenario())

    def test_repeated_cancellation_reaps_disconnect_probe(self) -> None:
        from fork_server.http_api import api

        async def scenario() -> None:
            future = asyncio.get_running_loop().create_future()
            probe_started = asyncio.Event()
            cleanup_started = asyncio.Event()
            release_cleanup = asyncio.Event()
            cleanup_finished = asyncio.Event()
            cleanup_cancelled = asyncio.Event()

            class BlockingRequest:
                async def is_disconnected(self):
                    probe_started.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        # Mirrors Starlette's internal nonblocking cancel scope,
                        # which can consume a cancellation aimed at this probe.
                        cleanup_started.set()
                        try:
                            await release_cleanup.wait()
                        except asyncio.CancelledError:
                            cleanup_cancelled.set()
                            raise
                        cleanup_finished.set()
                        return False

            waiter = asyncio.create_task(
                api._await_result_or_disconnect(
                    BlockingRequest(),
                    future,
                    timeout=60.0,
                )
            )
            await probe_started.wait()
            waiter.cancel()
            await asyncio.wait_for(cleanup_started.wait(), timeout=0.5)

            try:
                waiter.cancel()
                await asyncio.sleep(0)
                self.assertFalse(waiter.done())
                self.assertFalse(cleanup_cancelled.is_set())
            finally:
                release_cleanup.set()

            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(waiter, timeout=0.5)
            await asyncio.sleep(0)

            self.assertTrue(cleanup_finished.is_set())
            self.assertFalse(cleanup_cancelled.is_set())
            self.assertFalse(future.cancelled())
            self.assertFalse(
                any(
                    task is not asyncio.current_task()
                    and not task.done()
                    and task.get_name() == "capswriter-http-client-disconnect"
                    for task in asyncio.all_tasks()
                )
            )

        asyncio.run(scenario())

    def test_repeated_cancellation_finishes_admission_cleanup(self) -> None:
        from fork_server.http_api import api

        async def scenario() -> None:
            cleanup_started = asyncio.Event()
            release_cleanup = asyncio.Event()
            cleanup_finished = asyncio.Event()
            cleanup_cancelled = asyncio.Event()
            replay_instances = []

            class GatedController:
                @asynccontextmanager
                async def slot(self):
                    try:
                        yield
                    finally:
                        cleanup_started.set()
                        try:
                            await release_cleanup.wait()
                        except asyncio.CancelledError:
                            cleanup_cancelled.set()
                            raise
                        cleanup_finished.set()

            class TrackingReplayableReceive(api.ReplayableReceive):
                def __init__(self, receive):
                    super().__init__(receive)
                    self.close_calls = 0
                    replay_instances.append(self)

                def close(self) -> None:
                    self.close_calls += 1
                    super().close()

            async def receive():
                await asyncio.Event().wait()

            class FakeRequest:
                def __init__(self) -> None:
                    self.receive = receive
                    self._receive = receive

            entered = asyncio.Event()

            async def use_slot() -> None:
                async with api._admission_slot_or_disconnect(
                    FakeRequest(),
                    GatedController(),
                ):
                    entered.set()
                    await asyncio.Event().wait()

            with patch.object(api, "ReplayableReceive", TrackingReplayableReceive):
                task = asyncio.create_task(use_slot())
                await asyncio.wait_for(entered.wait(), timeout=0.5)
                task.cancel()
                await asyncio.wait_for(cleanup_started.wait(), timeout=0.5)

                try:
                    task.cancel()
                    await asyncio.sleep(0)
                    self.assertFalse(task.done())
                    self.assertFalse(cleanup_cancelled.is_set())
                finally:
                    release_cleanup.set()

                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=0.5)

            self.assertTrue(cleanup_finished.is_set())
            self.assertFalse(cleanup_cancelled.is_set())
            self.assertEqual(len(replay_instances), 1)
            self.assertEqual(replay_instances[0].close_calls, 1)

        asyncio.run(scenario())

    def test_result_race_timeout_preserves_router_future_ownership(self) -> None:
        from fork_server.http_api import api

        async def scenario() -> None:
            future = asyncio.get_running_loop().create_future()

            class BlockingRequest:
                async def is_disconnected(self):
                    await asyncio.Event().wait()

            with self.assertRaises(asyncio.TimeoutError):
                await api._await_result_or_disconnect(
                    BlockingRequest(),
                    future,
                    timeout=0.0,
                )
            await asyncio.sleep(0)

            self.assertFalse(future.cancelled())
            self.assertFalse(future.done())
            self.assertFalse(
                any(
                    task is not asyncio.current_task()
                    and not task.done()
                    and task.get_name() == "capswriter-http-client-disconnect"
                    for task in asyncio.all_tasks()
                )
            )

        asyncio.run(scenario())

    def _request(
        self,
        fields,
        *,
        result_error: Exception | None = None,
        result_error_code: str | None = None,
        register_error: Exception | None = None,
        result_pending: bool = False,
        cancel_error: Exception | None = None,
        recognizer_available: bool = True,
        config_overrides: dict[str, object] | None = None,
        authorization: str | None = None,
        content_length: int | None = None,
        decode_delay: float = 0.0,
        preoccupy_admission: bool = False,
        body_override: bytes | None = None,
        boundary_override: str | None = None,
        origin: str | None = None,
    ):
        from core.server.schema import Result
        from fork_server.http_api import api

        class FakeRouter:
            def __init__(self):
                self.register_calls = 0
                self.cancel_calls = 0

            def register(self, task_id):
                self.register_calls += 1
                if register_error is not None:
                    raise register_error
                future = asyncio.get_running_loop().create_future()
                if result_pending:
                    pass
                elif result_error is not None:
                    future.set_exception(result_error)
                else:
                    future.set_result(
                        Result(
                            task_id=task_id,
                            socket_id=f"http:{task_id}",
                            type="file",
                            duration=1.0,
                            text="hello",
                            text_accu="hello",
                            tokens=["hello"],
                            timestamps=[0.0],
                            is_final=True,
                            error_code=result_error_code,
                            error_message=(
                                "Recognition failed while processing the audio."
                                if result_error_code
                                else None
                            ),
                        )
                    )
                return future

            def cancel(self, task_id):
                self.cancel_calls += 1
                if cancel_error is not None:
                    raise cancel_error

            def recognizer_process_alive(self):
                return recognizer_available

        router = FakeRouter()
        async def decode_audio(*args, **kwargs):
            del args, kwargs
            if decode_delay:
                await asyncio.sleep(decode_delay)
            return b"\0" * 64_000

        decode = AsyncMock(side_effect=decode_audio)
        body, boundary = _multipart_body(fields)
        if body_override is not None:
            body = body_override
        if boundary_override is not None:
            boundary = boundary_override

        async def run():
            with ExitStack() as stack:
                for name, value in {
                    "http_api_key": "",
                    "http_api_cors_origins": [],
                    "http_api_max_concurrent_requests": 2,
                    "http_api_max_pending_requests": 4,
                    "http_api_max_upload_mb": 10,
                    "http_api_max_audio_seconds": 3600,
                    "http_api_task_timeout": 5.0,
                    "http_api_log_transcripts": False,
                    **(config_overrides or {}),
                }.items():
                    stack.enter_context(patch.object(api.Config, name, value, create=True))
                stack.enter_context(patch.object(api, "task_router", router))
                stack.enter_context(patch.object(api, "decode_to_pcm", decode))
                router.submit = stack.enter_context(
                    patch.object(api, "_split_and_submit")
                )
                stack.enter_context(patch.object(api, "log_prompt_context"))
                stack.enter_context(patch.object(api, "log_transcription_result"))
                app = api.create_app()
                if preoccupy_admission:
                    async with app.state.transcription_admission.slot():
                        response = await _asgi_post(
                            app,
                            body,
                            boundary,
                            authorization=authorization,
                            content_length=content_length,
                            origin=origin,
                        )
                else:
                    response = await _asgi_post(
                        app,
                        body,
                        boundary,
                        authorization=authorization,
                        content_length=content_length,
                        origin=origin,
                    )
                await asyncio.sleep(0)
                router.orphan_disconnect_tasks = [
                    task
                    for task in asyncio.all_tasks()
                    if task is not asyncio.current_task()
                    and not task.done()
                    and task.get_name() == "capswriter-http-client-disconnect"
                ]
                return response

        status, headers, response_body, receive_calls = asyncio.run(run())
        payload = json.loads(response_body) if response_body else None
        return status, headers, payload, router, decode, receive_calls

    def test_openapi_describes_supported_request_and_response_contract(self) -> None:
        from fork_server.http_api import api

        schema = api.create_app().openapi()
        self.assertIn("file-transcription subset", schema["info"]["description"])
        self.assertNotIn("Drop-in replacement", schema["info"]["description"])
        operation = schema["paths"]["/v1/audio/transcriptions"]["post"]
        request_schema = operation["requestBody"]["content"][
            "multipart/form-data"
        ]["schema"]
        if "$ref" in request_schema:
            component = request_schema["$ref"].rsplit("/", 1)[-1]
            request_schema = schema["components"]["schemas"][component]
        self.assertEqual(set(request_schema["required"]), {"file", "model"})
        self.assertFalse(request_schema["additionalProperties"])
        self.assertEqual(
            request_schema["properties"]["model"]["enum"],
            ["whisper-1"],
        )
        self.assertIn("timestamp_granularities", request_schema["properties"])
        self.assertIn("timestamp_granularities[]", request_schema["properties"])
        self.assertEqual(request_schema["properties"]["stream"]["enum"], [False])
        temperature = request_schema["properties"]["temperature"]
        self.assertEqual(temperature["type"], "number")
        self.assertEqual(temperature["minimum"], 0.0)
        self.assertEqual(temperature["maximum"], 1.0)

        success_content = operation["responses"]["200"]["content"]
        self.assertEqual(
            set(success_content),
            {
                "application/json",
                "application/x-subrip",
                "text/plain",
                "text/vtt",
            },
        )
        for status in (
            "400",
            "401",
            "403",
            "413",
            "422",
            "429",
            "500",
            "503",
            "504",
        ):
            error_schema = operation["responses"][status]["content"][
                "application/json"
            ]["schema"]
            self.assertIn("error", error_schema["required"])
        self.assertNotIn("/v1/audio/translations", schema["paths"])

    def test_verbose_word_timestamp_contract_and_finally_cleanup(self) -> None:
        status, _headers, payload, router, _decode, _receive_calls = self._request(
            [
                ("model", "whisper-1"),
                ("response_format", "verbose_json"),
                ("temperature", "0.25"),
                ("timestamp_granularities[]", "word"),
            ]
        )

        self.assertEqual(status, 200)
        self.assertIn("words", payload)
        self.assertNotIn("segments", payload)
        self.assertEqual(router.register_calls, 1)
        self.assertEqual(router.cancel_calls, 1)

    def test_worker_transcript_logging_follows_explicit_http_opt_in(self) -> None:
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                status, _headers, _payload, router, _decode, _receive_calls = (
                    self._request(
                        [("model", "whisper-1")],
                        config_overrides={"http_api_log_transcripts": enabled},
                    )
                )

                self.assertEqual(status, 200)
                self.assertEqual(
                    router.submit.call_args.kwargs["log_transcript"],
                    enabled,
                )
                self.assertIsInstance(
                    router.submit.call_args.kwargs["deadline_monotonic"],
                    float,
                )

    def test_plain_timestamp_granularity_spelling_is_supported(self) -> None:
        status, _headers, payload, router, _decode, _receive_calls = self._request(
            [
                ("model", "whisper-1"),
                ("response_format", "verbose_json"),
                ("timestamp_granularities", "word"),
            ]
        )

        self.assertEqual(status, 200)
        self.assertIn("words", payload)
        self.assertNotIn("segments", payload)
        self.assertEqual(router.register_calls, 1)
        self.assertEqual(router.cancel_calls, 1)

    def test_invalid_language_is_rejected_before_decode_and_register(self) -> None:
        status, _headers, payload, router, decode, _receive_calls = self._request(
            [("model", "whisper-1"), ("language", "en warning")]
        )

        self.assertEqual(status, 400)
        self.assertIn("language hint", payload["error"]["message"])
        self.assertEqual(router.register_calls, 0)
        decode.assert_not_awaited()

    def test_missing_model_is_rejected_before_decode_and_register(self) -> None:
        status, _headers, payload, router, decode, _receive_calls = self._request([])

        self.assertEqual(status, 400)
        self.assertIn("Missing required field: 'model'", payload["error"]["message"])
        self.assertEqual(router.register_calls, 0)
        decode.assert_not_awaited()

    def test_mismatched_multipart_boundary_returns_400_and_never_decodes(self) -> None:
        body, _boundary = _multipart_body([("model", "whisper-1")])
        status, headers, payload, router, decode, _receive_calls = self._request(
            [],
            body_override=body,
            boundary_override="----DifferentBoundary",
        )

        self.assertEqual(status, 400)
        self.assertEqual(headers[b"connection"], b"close")
        self.assertEqual(payload["error"]["message"], "Invalid multipart form")
        self.assertEqual(router.register_calls, 0)
        decode.assert_not_awaited()

    def test_invalid_raw_scalars_are_rejected_before_decode_and_register(self) -> None:
        for fields, expected in [
            ([('model', '')], "Invalid model"),
            ([('response_format', '')], "Invalid response_format"),
            ([('temperature', '')], "Invalid temperature"),
            ([('temperature', 'not-a-number')], "Invalid temperature"),
            (
                [('temperature', '0.25'), ('temperature', 'not-a-number')],
                "duplicate field",
            ),
        ]:
            with self.subTest(fields=fields):
                status, _headers, payload, router, decode, _receive_calls = self._request(
                    [('model', 'whisper-1'), *fields]
                    if not any(name == "model" for name, _value in fields)
                    else fields
                )
                self.assertEqual(status, 400)
                self.assertIn(expected, payload["error"]["message"])
                self.assertEqual(router.register_calls, 0)
                decode.assert_not_awaited()

    def test_current_only_capabilities_return_openai_error_envelopes(self) -> None:
        for fields, expected in [
            ([('model', 'whisper-1'), ('stream', 'true')], "stream"),
            (
                [('model', 'whisper-1'), ('response_format', 'diarized_json')],
                "diarization",
            ),
            (
                [('model', 'whisper-1'), ('include[]', 'logprobs')],
                "log probabilities",
            ),
        ]:
            with self.subTest(fields=fields):
                status, _headers, payload, router, decode, _receive_calls = self._request(fields)
                self.assertEqual(status, 400)
                self.assertIn(expected, payload["error"]["message"])
                self.assertEqual(router.register_calls, 0)
                decode.assert_not_awaited()

    def test_internal_recognition_error_is_generic_and_cleans_router(self) -> None:
        from fork_server.http_api import api

        handler, records = self._capture_api_logs(api.logger)
        try:
            status, _headers, payload, router, _decode, _receive_calls = self._request(
                [("model", "whisper-1")],
                result_error=RuntimeError("secret /model/path"),
            )
        finally:
            api.logger.removeHandler(handler)

        self.assertEqual(status, 500)
        self.assertEqual(payload["error"]["message"], "Recognition failed")
        self.assertNotIn("secret", json.dumps(payload))
        self.assertEqual(router.register_calls, 1)
        self.assertEqual(router.cancel_calls, 1)
        emitted = "\n".join(record.getMessage() for record in records)
        self.assertNotIn("secret", emitted)
        self.assertNotIn("/model/path", emitted)
        self.assertIn("details=<redacted>", emitted)
        self.assertTrue(all(not record.exc_info for record in records))

    def test_router_cleanup_error_log_is_redacted_by_default(self) -> None:
        from fork_server.http_api import api

        handler, records = self._capture_api_logs(api.logger)
        try:
            status, _headers, _payload, router, _decode, _receive_calls = self._request(
                [("model", "whisper-1")],
                cancel_error=RuntimeError("secret cleanup /model/path"),
            )
        finally:
            api.logger.removeHandler(handler)

        self.assertEqual(status, 200)
        self.assertEqual(router.cancel_calls, 1)
        emitted = "\n".join(record.getMessage() for record in records)
        self.assertNotIn("secret", emitted)
        self.assertNotIn("/model/path", emitted)
        self.assertIn("details=<redacted>", emitted)
        self.assertTrue(all(not record.exc_info for record in records))

    def test_post_upload_disconnect_cancels_router_without_orphan_watcher(self) -> None:
        status, _headers, payload, router, _decode, receive_calls = self._request(
            [("model", "whisper-1")],
            result_pending=True,
        )

        self.assertEqual(status, 499)
        self.assertIsNone(payload)
        self.assertGreaterEqual(receive_calls, 2)
        self.assertEqual(router.register_calls, 1)
        self.assertEqual(router.cancel_calls, 1)
        self.assertEqual(router.orphan_disconnect_tasks, [])

    def test_completed_result_wins_disconnect_race(self) -> None:
        status, _headers, payload, router, _decode, receive_calls = self._request(
            [("model", "whisper-1"), ("response_format", "verbose_json")]
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["text"], "hello")
        self.assertEqual(payload["language"], "auto")
        self.assertGreaterEqual(receive_calls, 1)
        self.assertEqual(router.cancel_calls, 1)
        self.assertEqual(router.orphan_disconnect_tasks, [])

    def test_worker_error_result_is_generic_and_cleans_router(self) -> None:
        status, _headers, payload, router, _decode, _receive_calls = self._request(
            [("model", "whisper-1")],
            result_error_code="worker_processing_failed",
        )

        self.assertEqual(status, 500)
        self.assertEqual(payload["error"]["message"], "Recognition failed")
        self.assertNotIn("worker_processing_failed", json.dumps(payload))
        self.assertEqual(router.register_calls, 1)
        self.assertEqual(router.cancel_calls, 1)

    def test_partial_router_registration_failure_is_cleaned_up(self) -> None:
        status, _headers, payload, router, _decode, _receive_calls = self._request(
            [("model", "whisper-1")],
            register_error=RuntimeError("partial registration failure"),
        )

        self.assertEqual(status, 500)
        self.assertEqual(payload["error"]["message"], "Recognition failed")
        self.assertEqual(router.register_calls, 1)
        self.assertEqual(router.cancel_calls, 1)

    def test_auth_and_declared_body_limit_reject_before_receive(self) -> None:
        status, headers, payload, router, decode, receive_calls = self._request(
            [("model", "whisper-1")],
            config_overrides={"http_api_key": "sk-test"},
        )
        self.assertEqual(status, 401)
        self.assertEqual(payload["error"]["type"], "authentication_error")
        self.assertEqual(headers[b"connection"], b"close")
        self.assertEqual(receive_calls, 0)
        self.assertEqual(router.register_calls, 0)
        decode.assert_not_awaited()

        status, headers, payload, router, decode, receive_calls = self._request(
            [("model", "whisper-1")],
            config_overrides={"http_api_max_upload_mb": 1},
            content_length=3 * 1024 * 1024,
        )
        self.assertEqual(status, 413)
        self.assertIn("too large", payload["error"]["message"])
        self.assertEqual(headers[b"connection"], b"close")
        self.assertEqual(receive_calls, 0)
        self.assertEqual(router.register_calls, 0)
        decode.assert_not_awaited()

    def test_disallowed_browser_origin_is_rejected_before_body_receive(self) -> None:
        status, headers, payload, router, decode, receive_calls = self._request(
            [("model", "whisper-1")],
            origin="https://evil.example",
        )

        self.assertEqual(status, 403)
        self.assertEqual(headers[b"connection"], b"close")
        self.assertEqual(payload["error"]["type"], "permission_error")
        self.assertEqual(receive_calls, 0)
        self.assertEqual(router.register_calls, 0)
        decode.assert_not_awaited()

    def test_fail_stopping_recognizer_rejects_before_body_receive(self) -> None:
        status, headers, payload, router, decode, receive_calls = self._request(
            [("model", "whisper-1")],
            recognizer_available=False,
        )

        self.assertEqual(status, 503)
        self.assertEqual(headers[b"connection"], b"close")
        self.assertEqual(headers[b"retry-after"], b"5")
        self.assertEqual(payload["error"]["type"], "server_error")
        self.assertEqual(receive_calls, 0)
        self.assertEqual(router.register_calls, 0)
        decode.assert_not_awaited()

    def test_explicitly_allowed_browser_origin_can_transcribe(self) -> None:
        status, headers, _payload, router, decode, receive_calls = self._request(
            [("model", "whisper-1")],
            origin="https://app.example",
            config_overrides={
                "http_api_cors_origins": ["https://app.example"],
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(headers[b"access-control-allow-origin"], b"https://app.example")
        self.assertGreater(receive_calls, 0)
        self.assertEqual(router.register_calls, 1)
        decode.assert_awaited_once()

    def test_noncanonical_origin_is_rejected_consistently_with_cors_middleware(self) -> None:
        status, headers, _payload, router, decode, receive_calls = self._request(
            [("model", "whisper-1")],
            origin="HTTPS://APP.EXAMPLE:443",
            config_overrides={
                "http_api_cors_origins": ["https://app.example"],
            },
        )

        self.assertEqual(status, 403)
        self.assertEqual(headers[b"connection"], b"close")
        self.assertEqual(receive_calls, 0)
        self.assertEqual(router.register_calls, 0)
        decode.assert_not_awaited()

    def test_end_to_end_deadline_includes_decode(self) -> None:
        from fork_server.http_api import api

        with patch.object(api, "task_timeout_seconds", return_value=0.01):
            status, _headers, payload, router, _decode, _receive_calls = self._request(
                [("model", "whisper-1")],
                decode_delay=1.0,
            )

        self.assertEqual(status, 504)
        self.assertEqual(payload["error"]["type"], "timeout_error")
        self.assertEqual(router.register_calls, 0)

    def test_full_admission_queue_returns_429_before_receive(self) -> None:
        status, headers, payload, router, decode, receive_calls = self._request(
            [("model", "whisper-1")],
            config_overrides={
                "http_api_max_concurrent_requests": 1,
                "http_api_max_pending_requests": 0,
            },
            preoccupy_admission=True,
        )

        self.assertEqual(status, 429)
        self.assertEqual(payload["error"]["type"], "rate_limit_error")
        self.assertEqual(headers[b"retry-after"], b"1")
        self.assertEqual(headers[b"connection"], b"close")
        self.assertEqual(receive_calls, 0)
        self.assertEqual(router.register_calls, 0)
        decode.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
