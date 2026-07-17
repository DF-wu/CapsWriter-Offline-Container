import unittest

from util.server.openai_formatter import _fmt_srt_ts, _fmt_vtt_ts, format_response
from util.server.server_classes import Result


class TimestampFormattingTests(unittest.TestCase):
    def test_rounding_carries_across_minute_and_hour_boundaries(self):
        self.assertEqual(_fmt_srt_ts(59.9996), "00:01:00,000")
        self.assertEqual(_fmt_vtt_ts(3599.9996), "01:00:00.000")

    def test_negative_timestamp_is_clamped(self):
        self.assertEqual(_fmt_srt_ts(-1.0), "00:00:00,000")

    def test_non_finite_timestamp_is_clamped(self):
        self.assertEqual(_fmt_srt_ts(float("nan")), "00:00:00,000")
        self.assertEqual(_fmt_vtt_ts(float("inf")), "00:00:00.000")


class ResponseFormattingTests(unittest.TestCase):
    def setUp(self):
        self.result = Result(
            task_id="task",
            socket_id="http:task",
            source="file",
            duration=1.5,
            text="fallback",
            text_accu="你好。",
            tokens=["你", "好", "。"],
            timestamps=[0.0, 0.5, 1.0],
            is_final=True,
        )

    def test_json_prefers_accumulated_timestamp_text(self):
        body, media_type = format_response(self.result, "json")

        self.assertEqual(body, {"text": "你好。"})
        self.assertEqual(media_type, "application/json")

    def test_verbose_json_preserves_duration_and_monotonic_word_bounds(self):
        body, _ = format_response(self.result, "verbose_json", language="zh")

        self.assertEqual(body["duration"], 1.5)
        self.assertEqual(body["language"], "zh")
        self.assertEqual(body["words"][-1]["end"], 1.5)


if __name__ == "__main__":
    unittest.main()
