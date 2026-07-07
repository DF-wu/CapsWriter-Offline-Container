# coding: utf-8

from __future__ import annotations

import ast
import json
import math
import os
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
UTILITY_PATHS = [
    ROOT / "core" / "server" / "engines" / "qwen_asr_gguf" / "export" / "gguf" / "utility.py",
    ROOT / "core" / "server" / "engines" / "force_aligner_gguf" / "export" / "gguf" / "utility.py",
    ROOT / "core" / "server" / "engines" / "fun_asr_gguf" / "export" / "gguf" / "utility.py",
]


class FakeNumpy:
    uint8 = "uint8"

    def frombuffer(self, raw_data, dtype):
        return {"raw_data": raw_data, "dtype": dtype}


class FakeResponse:
    def __init__(self, *, content: bytes = b"abcdef", status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        return None


class FakeRequests(types.SimpleNamespace):
    class RequestException(Exception):
        pass

    def __init__(self) -> None:
        super().__init__()
        self.get_calls = []
        self.head_calls = []
        self.get_response = FakeResponse()
        self.head_response = FakeResponse(status_code=204)
        self.head_side_effect = None

    def get(self, url, *, allow_redirects, headers, timeout):
        self.get_calls.append(
            {
                "url": url,
                "allow_redirects": allow_redirects,
                "headers": dict(headers),
                "timeout": timeout,
            }
        )
        return self.get_response

    def head(self, url, *, allow_redirects, headers, timeout):
        self.head_calls.append(
            {
                "url": url,
                "allow_redirects": allow_redirects,
                "headers": dict(headers),
                "timeout": timeout,
            }
        )
        if self.head_side_effect is not None:
            raise self.head_side_effect
        return self.head_response


def load_utility_namespace(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    keep_names = {
        "GGUF_EXPORT_HTTP_TIMEOUT_ENV",
        "DEFAULT_GGUF_EXPORT_HTTP_TIMEOUT_SECONDS",
    }
    body = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            body.append(node)
        elif isinstance(node, ast.Assign):
            names = {
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            if names & keep_names:
                body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in {
            "RemoteTensor",
            "SafetensorRemote",
        }:
            body.append(node)

    namespace = {
        "dataclass": dataclass,
        "json": json,
        "math": math,
        "np": FakeNumpy(),
        "os": os,
        "Path": Path,
    }
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


class GgufExportUtilityTest(unittest.TestCase):
    def test_http_timeout_accepts_default_and_configured_values(self) -> None:
        for path in UTILITY_PATHS:
            with self.subTest(path=path):
                namespace = load_utility_namespace(path)
                remote = namespace["SafetensorRemote"]

                with patch.dict(os.environ, {}, clear=True):
                    self.assertEqual(
                        remote._http_timeout_seconds(),
                        namespace["DEFAULT_GGUF_EXPORT_HTTP_TIMEOUT_SECONDS"],
                    )

                with patch.dict(
                    os.environ,
                    {namespace["GGUF_EXPORT_HTTP_TIMEOUT_ENV"]: "7.5"},
                ):
                    self.assertEqual(remote._http_timeout_seconds(), 7.5)

    def test_http_timeout_rejects_invalid_values(self) -> None:
        for path in UTILITY_PATHS:
            namespace = load_utility_namespace(path)
            remote = namespace["SafetensorRemote"]
            for value in ("bad", "0", "-2", "nan", "inf"):
                with self.subTest(path=path, value=value):
                    with patch.dict(
                        os.environ,
                        {namespace["GGUF_EXPORT_HTTP_TIMEOUT_ENV"]: value},
                    ):
                        with self.assertRaises(ValueError):
                            remote._http_timeout_seconds()

    def test_get_data_by_range_passes_configured_timeout(self) -> None:
        for path in UTILITY_PATHS:
            with self.subTest(path=path):
                namespace = load_utility_namespace(path)
                remote = namespace["SafetensorRemote"]
                fake_requests = FakeRequests()

                with (
                    patch.dict(
                        os.environ,
                        {namespace["GGUF_EXPORT_HTTP_TIMEOUT_ENV"]: "4.25"},
                    ),
                    patch.dict(sys.modules, {"requests": fake_requests}),
                ):
                    data = remote.get_data_by_range(
                        "https://example.test/model.safetensors",
                        start=10,
                        size=3,
                    )

                self.assertEqual(data, b"abc")
                self.assertEqual(len(fake_requests.get_calls), 1)
                call = fake_requests.get_calls[0]
                self.assertEqual(call["timeout"], 4.25)
                self.assertEqual(call["headers"]["Range"], "bytes=10-13")
                self.assertTrue(call["allow_redirects"])

    def test_check_file_exist_passes_default_timeout_and_handles_request_error(self) -> None:
        for path in UTILITY_PATHS:
            with self.subTest(path=path):
                namespace = load_utility_namespace(path)
                remote = namespace["SafetensorRemote"]
                fake_requests = FakeRequests()
                fake_requests.head_side_effect = fake_requests.RequestException("boom")

                with (
                    patch.dict(os.environ, {}, clear=True),
                    patch.dict(sys.modules, {"requests": fake_requests}),
                ):
                    exists = remote.check_file_exist(
                        "https://example.test/model.safetensors"
                    )

                self.assertFalse(exists)
                self.assertEqual(len(fake_requests.head_calls), 1)
                call = fake_requests.head_calls[0]
                self.assertEqual(
                    call["timeout"],
                    namespace["DEFAULT_GGUF_EXPORT_HTTP_TIMEOUT_SECONDS"],
                )
                self.assertEqual(call["headers"]["Range"], "bytes=0-0")
                self.assertTrue(call["allow_redirects"])

    def test_invalid_timeout_rejects_before_opening_request(self) -> None:
        for path in UTILITY_PATHS:
            with self.subTest(path=path):
                namespace = load_utility_namespace(path)
                remote = namespace["SafetensorRemote"]
                fake_requests = FakeRequests()

                with (
                    patch.dict(
                        os.environ,
                        {namespace["GGUF_EXPORT_HTTP_TIMEOUT_ENV"]: "nan"},
                    ),
                    patch.dict(sys.modules, {"requests": fake_requests}),
                ):
                    with self.assertRaises(ValueError):
                        remote.get_data_by_range("https://example.test/file", 0)

                self.assertEqual(fake_requests.get_calls, [])


if __name__ == "__main__":
    unittest.main()
