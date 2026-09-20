# coding: utf-8
"""End-to-end wire contract through the official OpenAI Python SDK."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
import importlib.util
from unittest.mock import AsyncMock, patch
import unittest


SDK_DEPS_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("fastapi", "httpx", "multipart", "openai")
)


@unittest.skipUnless(
    SDK_DEPS_AVAILABLE,
    "official OpenAI SDK contract dependencies are not installed",
)
class OpenAiPythonSdkContractTest(unittest.TestCase):
    def test_json_text_and_verbose_responses_parse_through_sdk(self) -> None:
        from core.server.schema import Result
        from fork_server.http_api import api
        import httpx
        from openai import AsyncOpenAI

        class FakeRouter:
            def __init__(self) -> None:
                self.cancelled = 0

            def register(self, task_id):
                future = asyncio.get_running_loop().create_future()
                future.set_result(
                    Result(
                        task_id=task_id,
                        socket_id=f"http:{task_id}",
                        type="file",
                        duration=1.0,
                        text="hello world",
                        text_accu="hello world",
                        tokens=["hello", " world"],
                        timestamps=[0.0, 0.5],
                        is_final=True,
                    )
                )
                return future

            def cancel(self, task_id):
                del task_id
                self.cancelled += 1

            @staticmethod
            def recognizer_process_alive():
                return True

        async def scenario() -> None:
            router = FakeRouter()
            decode = AsyncMock(return_value=b"\0" * 64_000)

            with ExitStack() as stack:
                for name, value in {
                    "http_api_key": "sk-local-contract",
                    "http_api_cors_origins": [],
                    "http_api_max_concurrent_requests": 2,
                    "http_api_max_pending_requests": 4,
                    "http_api_max_upload_mb": 10,
                    "http_api_max_audio_seconds": 3600,
                    "http_api_task_timeout": 5.0,
                    "http_api_log_transcripts": False,
                }.items():
                    stack.enter_context(
                        patch.object(api.Config, name, value, create=True)
                    )
                stack.enter_context(patch.object(api, "task_router", router))
                stack.enter_context(patch.object(api, "decode_to_pcm", decode))
                stack.enter_context(patch.object(api, "_split_and_submit"))
                stack.enter_context(patch.object(api, "log_prompt_context"))
                stack.enter_context(patch.object(api, "log_transcription_result"))

                app = api.create_app()
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://capswriter.test",
                ) as http_client:
                    client = AsyncOpenAI(
                        api_key="sk-local-contract",
                        base_url="http://capswriter.test/v1",
                        http_client=http_client,
                        max_retries=0,
                    )
                    async with client:
                        json_response = await client.audio.transcriptions.create(
                            model="whisper-1",
                            file=("sample.wav", b"RIFF-test", "audio/wav"),
                        )
                        self.assertEqual(json_response.text, "hello world")

                        text_response = await client.audio.transcriptions.create(
                            model="whisper-1",
                            file=("sample.wav", b"RIFF-test", "audio/wav"),
                            response_format="text",
                        )
                        self.assertEqual(text_response, "hello world")

                        verbose = await client.audio.transcriptions.create(
                            model="whisper-1",
                            file=("sample.wav", b"RIFF-test", "audio/wav"),
                            response_format="verbose_json",
                            timestamp_granularities=["word"],
                            temperature=0.25,
                        )
                        self.assertEqual(verbose.text, "hello world")
                        self.assertEqual(verbose.language, "auto")
                        self.assertEqual(len(verbose.words or []), 2)
                        self.assertIsNone(verbose.segments)

            self.assertEqual(router.cancelled, 3)
            self.assertEqual(decode.await_count, 3)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
