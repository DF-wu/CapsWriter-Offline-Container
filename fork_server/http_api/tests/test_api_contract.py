# coding: utf-8
"""Dependency-light request-contract tests for the FastAPI endpoint."""

from __future__ import annotations

import ast
import math
from pathlib import Path
import types
from typing import Optional
import unittest


API_PATH = Path(__file__).resolve().parents[1] / "api.py"
HELPERS = {
    "_request_error",
    "_validate_model",
    "_validate_temperature",
    "_validate_response_format",
    "_validate_stream_value",
    "_validate_form_fields",
    "_validate_timestamp_granularities",
}
CONSTANTS = {
    "OPENAI_TRANSCRIPTION_MODEL",
    "MAX_MODEL_NAME_CHARS",
    "SUPPORTED_RESPONSE_FORMATS",
    "SUPPORTED_TIMESTAMP_GRANULARITIES",
    "_SCALAR_FORM_FIELDS",
    "_SUPPORTED_FORM_FIELDS",
    "_UNSUPPORTED_CAPABILITY_FIELDS",
}


class FakeHTTPException(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class FakeForm:
    def __init__(self, values: dict[str, list[str]]):
        self.values = values

    def keys(self):
        return self.values.keys()

    def getlist(self, name: str):
        return list(self.values.get(name, []))


def load_contract_helpers():
    tree = ast.parse(API_PATH.read_text(encoding="utf-8"), filename=str(API_PATH))
    selected: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            selected.append(node)
        elif isinstance(node, ast.Import) and any(alias.name == "math" for alias in node.names):
            selected.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {target.id for target in targets if isinstance(target, ast.Name)}
            if names & CONSTANTS:
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in HELPERS:
            selected.append(node)

    namespace = {
        "HTTPException": FakeHTTPException,
        "Optional": Optional,
    }
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(API_PATH), "exec"), namespace)
    return types.SimpleNamespace(**namespace)


class TranscriptionContractValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_contract_helpers()

    def assert_request_error(self, callback, message: str) -> FakeHTTPException:
        with self.assertRaises(FakeHTTPException) as raised:
            callback()
        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn(message, raised.exception.detail)
        return raised.exception

    def test_model_is_bounded_and_whisper_only(self) -> None:
        self.assertEqual(self.contract._validate_model(" whisper-1 "), "whisper-1")
        self.assert_request_error(
            lambda: self.contract._validate_model("gpt-4o-transcribe"),
            "Unsupported model",
        )
        self.assert_request_error(
            lambda: self.contract._validate_model("x" * 65),
            "at most 64",
        )

    def test_temperature_must_be_finite_and_in_range(self) -> None:
        self.assertEqual(self.contract._validate_temperature(0.25), 0.25)
        for value in (-0.1, 1.1, math.nan, math.inf):
            with self.subTest(value=value):
                self.assert_request_error(
                    lambda value=value: self.contract._validate_temperature(value),
                    "finite number from 0 to 1",
                )

    def test_current_only_capabilities_are_explicit_errors(self) -> None:
        self.assert_request_error(
            lambda: self.contract._validate_stream_value("true"),
            "Unsupported capability 'stream'",
        )
        self.contract._validate_stream_value("false")
        self.assert_request_error(
            lambda: self.contract._validate_response_format("diarized_json"),
            "speaker diarization",
        )
        self.assert_request_error(
            lambda: self.contract._validate_form_fields(
                FakeForm({"file": ["audio"], "include[]": ["logprobs"]})
            ),
            "log probabilities",
        )
        self.assert_request_error(
            lambda: self.contract._validate_form_fields(
                FakeForm({"file": ["audio"], "known_speaker_names[]": ["agent"]})
            ),
            "known-speaker diarization",
        )

    def test_unknown_and_duplicate_fields_are_rejected(self) -> None:
        self.assert_request_error(
            lambda: self.contract._validate_form_fields(
                FakeForm({"file": ["audio"], "future_option": ["1"]})
            ),
            "Unsupported transcription field",
        )
        self.assert_request_error(
            lambda: self.contract._validate_form_fields(
                FakeForm({"file": ["one", "two"]})
            ),
            "duplicate field",
        )

    def test_timestamp_granularities_follow_whisper_contract(self) -> None:
        validate = self.contract._validate_timestamp_granularities
        self.assertEqual(validate([], "verbose_json"), ("segment",))
        self.assertEqual(
            validate(["word", "segment", "word"], "verbose_json"),
            ("word", "segment"),
        )
        self.assert_request_error(
            lambda: validate(["word"], "json"),
            "requires response_format='verbose_json'",
        )
        self.assert_request_error(
            lambda: validate(["sentence"], "verbose_json"),
            "Invalid timestamp_granularities",
        )


if __name__ == "__main__":
    unittest.main()
