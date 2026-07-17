# coding: utf-8

from __future__ import annotations

import ast
import unittest
from pathlib import Path


class HttpApiSourceTest(unittest.TestCase):
    def test_transcription_decode_uses_remaining_end_to_end_deadline(self) -> None:
        tree = ast.parse(Path("fork_server/http_api/api.py").read_text())
        decode_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "decode_to_pcm"
        ]

        self.assertTrue(decode_calls)
        self.assertTrue(
            any(
                keyword.arg == "timeout"
                and isinstance(keyword.value, ast.Call)
                and isinstance(keyword.value.func, ast.Name)
                and keyword.value.func.id == "_remaining_seconds"
                and len(keyword.value.args) == 1
                and isinstance(keyword.value.args[0], ast.Name)
                and keyword.value.args[0].id == "deadline"
                for call in decode_calls
                for keyword in call.keywords
            )
        )


if __name__ == "__main__":
    unittest.main()
