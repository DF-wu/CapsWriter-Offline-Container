# coding: utf-8

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass, field
from unittest.mock import patch


@dataclass
class Result:
    task_id: str
    socket_id: str
    type: str
    duration: float = 0.0
    text: str = ""
    text_accu: str = ""
    tokens: list[str] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)


def load_formatter():
    import importlib
    import sys
    import types

    core_module = types.ModuleType("core")
    server_module = types.ModuleType("core.server")
    schema_module = types.ModuleType("core.server.schema")
    schema_module.Result = Result

    with patch.dict(
        sys.modules,
        {
            "core": core_module,
            "core.server": server_module,
            "core.server.schema": schema_module,
        },
    ):
        sys.modules.pop("fork_server.http_api.openai_formatter", None)
        return importlib.import_module("fork_server.http_api.openai_formatter")


class OpenAIFormatterTimestampTest(unittest.TestCase):
    def test_srt_timestamp_rounding_carries_to_next_minute(self) -> None:
        openai_formatter = load_formatter()
        result = Result(
            task_id="task-srt",
            socket_id="http:task-srt",
            type="file",
            duration=59.9996,
            text="hello",
        )

        body, media_type = openai_formatter.format_response(result, "srt")

        self.assertEqual(media_type, "application/x-subrip; charset=utf-8")
        self.assertIn("00:00:00,000 --> 00:01:00,000", body)
        self.assertNotIn("00:00:60", body)

    def test_vtt_timestamp_rounding_carries_to_next_hour(self) -> None:
        openai_formatter = load_formatter()
        result = Result(
            task_id="task-vtt",
            socket_id="http:task-vtt",
            type="file",
            duration=3599.9996,
            text="hello",
        )

        body, media_type = openai_formatter.format_response(result, "vtt")

        self.assertEqual(media_type, "text/vtt; charset=utf-8")
        self.assertIn("00:00:00.000 --> 01:00:00.000", body)
        self.assertNotIn("00:59:60", body)

    def test_negative_timestamp_clamps_to_zero(self) -> None:
        openai_formatter = load_formatter()
        self.assertEqual(openai_formatter._fmt_srt_ts(-0.1), "00:00:00,000")
        self.assertEqual(openai_formatter._fmt_vtt_ts(-0.1), "00:00:00.000")

    def test_verbose_json_defaults_to_complete_segment_schema(self) -> None:
        openai_formatter = load_formatter()
        result = Result(
            task_id="task-verbose",
            socket_id="http:task-verbose",
            type="file",
            duration=1.5,
            text="hello world",
            tokens=["hello", " world"],
            timestamps=[0.0, 0.7],
        )

        body, media_type = openai_formatter.format_response(
            result,
            "verbose_json",
            language="en",
            temperature=0.25,
        )

        self.assertEqual(media_type, "application/json")
        self.assertNotIn("words", body)
        self.assertEqual(body["language"], "en")
        self.assertEqual(len(body["segments"]), 1)
        segment = body["segments"][0]
        self.assertEqual(
            set(segment),
            {
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
            },
        )
        self.assertEqual(segment["tokens"], [])
        self.assertEqual(segment["temperature"], 0.25)

    def test_verbose_json_honors_word_granularity(self) -> None:
        openai_formatter = load_formatter()
        result = Result(
            task_id="task-words",
            socket_id="http:task-words",
            type="file",
            duration=1.0,
            text="hi",
            tokens=["h", "i"],
            timestamps=[-0.1, 0.4],
        )

        body, _ = openai_formatter.format_response(
            result,
            "verbose_json",
            timestamp_granularities=("word",),
        )

        self.assertNotIn("segments", body)
        self.assertEqual(body["language"], "auto")
        self.assertEqual(
            body["words"],
            [
                {"word": "h", "start": 0.0, "end": 0.4},
                {"word": "i", "start": 0.4, "end": 1.0},
            ],
        )

    def test_verbose_json_honors_both_granularities(self) -> None:
        openai_formatter = load_formatter()
        result = Result(
            task_id="task-both",
            socket_id="http:task-both",
            type="file",
            duration=1.0,
            text="ok",
            tokens=["ok"],
            timestamps=[0.0],
        )

        body, _ = openai_formatter.format_response(
            result,
            "verbose_json",
            timestamp_granularities=("segment", "word"),
        )

        self.assertIn("segments", body)
        self.assertIn("words", body)

    def test_verbose_json_normalizes_non_monotonic_backend_timestamps(self) -> None:
        openai_formatter = load_formatter()
        result = Result(
            task_id="task-monotonic",
            socket_id="http:task-monotonic",
            type="file",
            duration=0.5,
            text="a. bc!",
            tokens=["a.", " b", "c!"],
            timestamps=[1.0, -0.25, math.nan],
        )

        body, _ = openai_formatter.format_response(
            result,
            "verbose_json",
            timestamp_granularities=("segment", "word"),
        )

        for collection in (body["segments"], body["words"]):
            previous_end = 0.0
            for item in collection:
                self.assertTrue(math.isfinite(item["start"]))
                self.assertTrue(math.isfinite(item["end"]))
                self.assertGreaterEqual(item["start"], previous_end)
                self.assertGreaterEqual(item["end"], item["start"])
                previous_end = item["end"]

    def test_verbose_json_rejects_unknown_granularity(self) -> None:
        openai_formatter = load_formatter()
        result = Result(
            task_id="task-invalid",
            socket_id="http:task-invalid",
            type="file",
        )

        with self.assertRaisesRegex(ValueError, "unsupported timestamp"):
            openai_formatter.format_response(
                result,
                "verbose_json",
                timestamp_granularities=("sentence",),
            )


if __name__ == "__main__":
    unittest.main()
