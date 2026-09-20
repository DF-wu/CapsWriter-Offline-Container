# coding: utf-8

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOCAL_PROVIDERS = {"lmstudio", "ollama"}
API_KEY_LIKE = re.compile(r"\bsk-[A-Za-z0-9*_=-]{8,}")


def role_assignments(path: Path) -> dict[str, object]:
    values: dict[str, object] = {}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
            values[target.id] = node.value.value
    return values


class RoleTemplateTest(unittest.TestCase):
    def test_tracked_role_templates_do_not_embed_api_keys(self) -> None:
        for path in sorted((ROOT / "LLM").glob("*.py")):
            if path.name == "__init__.py":
                continue
            with self.subTest(path=path.name):
                values = role_assignments(path)
                api_key = values.get("api_key", "")
                self.assertIsInstance(api_key, str)
                self.assertFalse(API_KEY_LIKE.search(api_key), path.name)

    def test_network_role_templates_are_not_enabled_without_key(self) -> None:
        for path in sorted((ROOT / "LLM").glob("*.py")):
            if path.name == "__init__.py":
                continue
            with self.subTest(path=path.name):
                values = role_assignments(path)
                provider = str(values.get("provider", "")).casefold()
                enabled = bool(values.get("enabled", False))
                api_key = str(values.get("api_key", ""))
                if provider and provider not in LOCAL_PROVIDERS and not api_key:
                    self.assertFalse(enabled, f"{path.name} enables {provider} without an API key")


if __name__ == "__main__":
    unittest.main()
