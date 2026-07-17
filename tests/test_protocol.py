import json
import unittest

from util.protocol import AudioMessage, RecognitionResult


class AudioMessageTests(unittest.TestCase):
    def test_round_trip_preserves_unicode_and_segment_settings(self):
        message = AudioMessage(
            task_id="工作-1",
            source="mic",
            data="AA==",
            is_final=False,
            time_start=123.5,
            seg_duration=60.0,
            seg_overlap=4.0,
        )

        restored = AudioMessage.from_dict(json.loads(message.to_json()))

        self.assertEqual(restored, message)
        self.assertIn("工作-1", message.to_json())

    def test_legacy_segment_defaults_remain_compatible(self):
        restored = AudioMessage.from_dict(
            {
                "task_id": "task",
                "source": "file",
                "data": "",
                "is_final": True,
                "time_start": 1.0,
            }
        )

        self.assertEqual(restored.seg_duration, 15.0)
        self.assertEqual(restored.seg_overlap, 2.0)


class RecognitionResultTests(unittest.TestCase):
    def test_optional_fields_default_to_independent_lists(self):
        required = {
            "task_id": "task",
            "is_final": True,
            "duration": 1.0,
            "time_start": 1.0,
            "time_submit": 2.0,
            "time_complete": 3.0,
            "text": "hello",
        }

        first = RecognitionResult.from_dict(required)
        second = RecognitionResult.from_dict(required)
        first.tokens.append("x")

        self.assertEqual(second.tokens, [])
        self.assertEqual(second.timestamps, [])
        self.assertEqual(json.loads(first.to_json())["text"], "hello")


if __name__ == "__main__":
    unittest.main()
