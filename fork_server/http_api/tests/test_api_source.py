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


if __name__ == "__main__":
    unittest.main()
