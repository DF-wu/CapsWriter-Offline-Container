# coding: utf-8

from __future__ import annotations

import ast
import io
import json
import math
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
HOTWORD_STANDALONE_PATH = (
    ROOT / "core" / "client" / "hotword" / "hotword_standalone.py"
)


class JsonResponse:
    def json(self) -> dict:
        return {"message": {"content": "corrected text"}}


class StreamResponse:
    def iter_lines(self):
        yield json.dumps({"message": {"content": "hot"}, "done": False}).encode()
        yield json.dumps({"message": {"content": "word"}, "done": True}).encode()


class FakeRequests:
    def __init__(self, response) -> None:
        self.response = response
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self.response


def load_hotword_namespace(fake_requests: FakeRequests) -> dict:
    source = HOTWORD_STANDALONE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(HOTWORD_STANDALONE_PATH))
    keep_names = {
        "OLLAMA_CHAT_TIMEOUT_ENV",
        "DEFAULT_OLLAMA_CHAT_TIMEOUT_SECONDS",
    }
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = {
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            if names & keep_names:
                body.append(node)
        elif (
            isinstance(node, ast.FunctionDef)
            and node.name in {"_ollama_chat_timeout_seconds", "ollama_chat"}
        ):
            body.append(node)

    namespace = {
        "Dict": dict,
        "List": list,
        "json": json,
        "math": math,
        "os": os,
        "requests": fake_requests,
    }
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(HOTWORD_STANDALONE_PATH), "exec"), namespace)
    return namespace


class HotwordStandaloneTest(unittest.TestCase):
    def test_ollama_chat_uses_configured_timeout(self) -> None:
        fake_requests = FakeRequests(JsonResponse())
        namespace = load_hotword_namespace(fake_requests)

        with (
            patch.dict(
                os.environ,
                {namespace["OLLAMA_CHAT_TIMEOUT_ENV"]: "1.25"},
            ),
            redirect_stdout(io.StringIO()),
        ):
            result = namespace["ollama_chat"](
                [{"role": "user", "content": "input"}],
                stream=False,
            )

        self.assertEqual(result, "corrected text")
        self.assertEqual(fake_requests.calls[0]["timeout"], 1.25)
        self.assertFalse(fake_requests.calls[0]["stream"])

    def test_ollama_chat_uses_default_timeout_for_streaming(self) -> None:
        fake_requests = FakeRequests(StreamResponse())
        namespace = load_hotword_namespace(fake_requests)

        with patch.dict(os.environ, {}, clear=False), redirect_stdout(io.StringIO()):
            os.environ.pop(namespace["OLLAMA_CHAT_TIMEOUT_ENV"], None)
            result = namespace["ollama_chat"](
                [{"role": "user", "content": "input"}],
                stream=True,
            )

        self.assertEqual(result, "hotword")
        self.assertEqual(
            fake_requests.calls[0]["timeout"],
            namespace["DEFAULT_OLLAMA_CHAT_TIMEOUT_SECONDS"],
        )
        self.assertTrue(fake_requests.calls[0]["stream"])

    def test_invalid_ollama_timeout_returns_before_request(self) -> None:
        fake_requests = FakeRequests(JsonResponse())
        namespace = load_hotword_namespace(fake_requests)

        with (
            patch.dict(
                os.environ,
                {namespace["OLLAMA_CHAT_TIMEOUT_ENV"]: "nan"},
            ),
            redirect_stdout(io.StringIO()),
        ):
            result = namespace["ollama_chat"](
                [{"role": "user", "content": "input"}],
                stream=False,
            )

        self.assertEqual(result, "")
        self.assertEqual(fake_requests.calls, [])


if __name__ == "__main__":
    unittest.main()
