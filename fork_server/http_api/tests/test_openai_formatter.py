# coding: utf-8

from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
