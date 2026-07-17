# coding: utf-8

from __future__ import annotations

import ast
import unittest
from pathlib import Path


def _load_async_function(path: Path, name: str) -> ast.AsyncFunctionDef:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(module):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")


class HttpApiSourceTest(unittest.TestCase):
    def test_startup_curl_includes_required_model(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        source = (repo_root / "fork_server/http_api/serve.py").read_text(
            encoding="utf-8"
        )

        self.assertIn('-F "file=@test.wav" -F "model=whisper-1"', source)

    def test_worker_deadline_uses_cross_process_monotonic_clock(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        source = (repo_root / "fork_server/http_api/api.py").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "worker_deadline_monotonic = time.monotonic() + timeout",
            source,
        )
        self.assertIn(
            "deadline_monotonic=worker_deadline_monotonic",
            source,
        )

    def test_translations_endpoint_checks_auth_before_501(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        function = _load_async_function(
            repo_root / "fork_server/http_api/api.py",
            "translations",
        )

        self.assertIn("authorization", [arg.arg for arg in function.args.args])
        self.assertGreaterEqual(len(function.body), 2)
        auth_call = function.body[0]
        self.assertIsInstance(auth_call, ast.Expr)
        self.assertIsInstance(auth_call.value, ast.Call)
        self.assertIsInstance(auth_call.value.func, ast.Name)
        self.assertEqual(auth_call.value.func.id, "_check_auth")
        self.assertEqual(len(auth_call.value.args), 1)
        self.assertIsInstance(auth_call.value.args[0], ast.Name)
        self.assertEqual(auth_call.value.args[0].id, "authorization")
        self.assertIsInstance(function.body[1], ast.Raise)

    def test_transcription_validation_precedes_router_registration(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        source = (repo_root / "fork_server/http_api/api.py").read_text(
            encoding="utf-8"
        )
        function = _load_async_function(
            repo_root / "fork_server/http_api/api.py",
            "transcriptions",
        )
        parser_function = _load_async_function(
            repo_root / "fork_server/http_api/api.py",
            "_parse_transcription_request",
        )

        register_calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "register"
        ]
        self.assertEqual(len(register_calls), 1)
        register_line = register_calls[0].lineno

        required_calls = {
            "_validate_form_fields",
            "_validate_model",
            "_validate_temperature",
            "_validate_response_format",
            "_validate_stream_value",
            "normalize_prompt_context",
            "normalize_language_hint",
            "_validate_timestamp_granularities",
        }
        parser_calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_parse_transcription_request"
        ]
        self.assertEqual(len(parser_calls), 1)
        self.assertLess(parser_calls[0].lineno, register_line)

        found = {}
        for node in ast.walk(parser_function):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in required_calls:
                    found.setdefault(node.func.id, node.lineno)

        self.assertEqual(set(found), required_calls)

        self.assertIn('HTTPException(500, "Recognition failed")', source)
        self.assertNotIn('HTTPException(500, f"Recognition error:', source)

    def test_router_registration_is_guarded_by_finally_cleanup(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        function = _load_async_function(
            repo_root / "fork_server/http_api/api.py",
            "transcriptions",
        )

        guarded = False
        for node in ast.walk(function):
            if not isinstance(node, ast.Try) or not node.finalbody:
                continue
            register_calls = [
                child
                for statement in node.body
                for child in ast.walk(statement)
                if isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "register"
            ]
            cancel_calls = [
                child
                for statement in node.finalbody
                for child in ast.walk(statement)
                if isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "cancel"
            ]
            if register_calls and cancel_calls:
                guarded = True
                break
        self.assertTrue(
            guarded,
            "router registration and work must share a finally cleanup region",
        )


if __name__ == "__main__":
    unittest.main()
