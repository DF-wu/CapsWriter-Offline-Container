# coding: utf-8

from __future__ import annotations

import ast
import asyncio
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
TEXT_OUTPUT_PATH = ROOT / "core" / "client" / "output" / "text_output.py"
LLM_TYPING_PATH = ROOT / "core" / "client" / "llm" / "llm_output_typing.py"


def load_type_text(system_name: str):
    tree = ast.parse(
        TEXT_OUTPUT_PATH.read_text(encoding="utf-8"),
        filename=str(TEXT_OUTPUT_PATH),
    )
    text_output = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TextOutput"
    )
    type_text = next(
        node
        for node in text_output.body
        if isinstance(node, ast.FunctionDef) and node.name == "_type_text"
    )
    probe = ast.ClassDef(
        name="ProbeTextOutput",
        bases=[],
        keywords=[],
        body=[type_text],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[probe], type_ignores=[]))
    keyboard_calls = []
    controller_calls = []

    class Controller:
        def type(self, text: str) -> None:
            controller_calls.append(text)

    namespace = {
        "keyboard": SimpleNamespace(write=keyboard_calls.append),
        "logger": SimpleNamespace(debug=lambda *_args, **_kwargs: None),
        "platform": SimpleNamespace(system=lambda: system_name),
        "_get_pynput_keyboard": lambda: SimpleNamespace(Controller=Controller),
    }
    exec(compile(module, str(TEXT_OUTPUT_PATH), "exec"), namespace)
    return namespace["ProbeTextOutput"], keyboard_calls, controller_calls


def load_llm_output_text(type_text_calls, paste_calls):
    tree = ast.parse(
        LLM_TYPING_PATH.read_text(encoding="utf-8"),
        filename=str(LLM_TYPING_PATH),
    )
    output_text = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "output_text"
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[output_text], type_ignores=[])
    )

    async def paste_text(text: str, *, restore_clipboard: bool) -> None:
        paste_calls.append((text, restore_clipboard))

    namespace = {
        "Config": SimpleNamespace(restore_clip=True),
        "logger": SimpleNamespace(debug=lambda *_args, **_kwargs: None),
        "paste_text": paste_text,
        "TextOutput": SimpleNamespace(_type_text=type_text_calls.append),
    }
    exec(compile(module, str(LLM_TYPING_PATH), "exec"), namespace)
    return namespace["output_text"]


class ClientTextOutputPortabilityTest(unittest.TestCase):
    def test_linux_typing_uses_non_root_pynput_controller(self) -> None:
        text_output, keyboard_calls, controller_calls = load_type_text("Linux")

        text_output._type_text("hello")

        self.assertEqual(controller_calls, ["hello"])
        self.assertEqual(keyboard_calls, [])

    def test_windows_typing_preserves_keyboard_write(self) -> None:
        text_output, keyboard_calls, controller_calls = load_type_text("Windows")

        text_output._type_text("hello")

        self.assertEqual(keyboard_calls, ["hello"])
        self.assertEqual(controller_calls, [])

    def test_llm_non_paste_output_uses_shared_platform_backend(self) -> None:
        type_text_calls = []
        paste_calls = []
        output_text = load_llm_output_text(type_text_calls, paste_calls)

        asyncio.run(output_text("hello", paste=False))

        self.assertEqual(type_text_calls, ["hello"])
        self.assertEqual(paste_calls, [])

    def test_llm_streaming_has_no_direct_keyboard_write_bypass(self) -> None:
        tree = ast.parse(LLM_TYPING_PATH.read_text(encoding="utf-8"))
        direct_keyboard_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "write"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "keyboard"
        ]
        shared_backend_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_type_text"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "TextOutput"
        ]

        self.assertEqual(direct_keyboard_calls, [])
        self.assertGreaterEqual(len(shared_backend_calls), 4)


if __name__ == "__main__":
    unittest.main()
