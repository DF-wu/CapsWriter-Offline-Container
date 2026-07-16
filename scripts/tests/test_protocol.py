# coding: utf-8

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "core" / "protocol.py"
SCHEMA_PATH = ROOT / "core" / "server" / "schema.py"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def make_message(message_type, **overrides):
    values = {
        "task_id": "task-1",
        "is_final": True,
        "duration": 1.5,
        "time_start": 10.0,
        "time_submit": 11.0,
        "time_complete": 12.0,
        "text": "hello",
        "text_accu": "hello",
        "tokens": ["hello"],
        "timestamps": [0.0],
    }
    values.update(overrides)
    return message_type(**values)


class RecognitionMessageProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_module("capswriter_protocol_under_test", PROTOCOL_PATH)
        cls.schema = load_module("capswriter_schema_under_test", SCHEMA_PATH)

    @classmethod
    def tearDownClass(cls) -> None:
        sys.modules.pop("capswriter_protocol_under_test", None)
        sys.modules.pop("capswriter_schema_under_test", None)

    def test_internal_result_has_optional_error_envelope(self) -> None:
        success = self.schema.Result("task-1", "socket-1", "mic")
        failure = self.schema.Result(
            "task-2",
            "socket-1",
            "mic",
            is_final=True,
            error_code="worker_processing_failed",
            error_message="Recognition failed while processing the audio.",
        )

        self.assertIsNone(success.error_code)
        self.assertIsNone(success.error_message)
        self.assertEqual(failure.error_code, "worker_processing_failed")

    def test_task_privacy_flag_preserves_existing_positional_command_slot(self) -> None:
        task = self.schema.Task(
            "cmd",
            b"",
            0.0,
            0.0,
            "gpu",
            "socket",
            False,
            0.0,
            0.0,
            "",
            "auto",
            16000,
            "gpu_boost",
        )

        self.assertEqual(task.command, "gpu_boost")
        self.assertTrue(task.log_transcript)
        self.assertIsNone(task.deadline_monotonic)

    def test_success_payload_remains_wire_compatible(self) -> None:
        message = make_message(self.protocol.RecognitionMessage)

        payload = message.to_dict()

        self.assertNotIn("error_code", payload)
        self.assertNotIn("error_message", payload)
        self.assertEqual(json.loads(message.to_json()), payload)

    def test_error_fields_round_trip_and_old_payload_defaults_to_none(self) -> None:
        message = make_message(
            self.protocol.RecognitionMessage,
            text="",
            text_accu="",
            tokens=[],
            timestamps=[],
            error_code="worker_processing_failed",
            error_message="Recognition failed while processing the audio.",
        )

        payload = json.loads(message.to_json())
        restored = self.protocol.RecognitionMessage.from_dict(payload)

        self.assertEqual(restored, message)
        self.assertEqual(payload["error_code"], "worker_processing_failed")

        payload.pop("error_code")
        payload.pop("error_message")
        legacy = self.protocol.RecognitionMessage.from_dict(payload)
        self.assertIsNone(legacy.error_code)
        self.assertIsNone(legacy.error_message)


if __name__ == "__main__":
    unittest.main()
