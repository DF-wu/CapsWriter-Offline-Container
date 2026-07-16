# coding: utf-8

from __future__ import annotations

import ast
import asyncio
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = ROOT / "core" / "client" / "state.py"
RESULT_PROCESSOR_PATH = (
    ROOT / "core" / "client" / "output" / "result_processor.py"
)


def load_state_surface() -> dict:
    tree = ast.parse(STATE_PATH.read_text(encoding="utf-8"), filename=str(STATE_PATH))
    helpers = {
        "websocket_state_name",
        "websocket_is_open",
        "websocket_is_closed",
    }
    body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in helpers
    ]
    client_state = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ClientState"
    )
    methods = [
        node
        for node in client_state.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"reset", "is_connected"}
    ]
    body.append(
        ast.ClassDef(
            name="ProbeState",
            bases=[],
            keywords=[],
            body=methods,
            decorator_list=[],
        )
    )
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    scheduled = []

    def schedule(coroutine, _loop):
        scheduled.append(coroutine)
        return SimpleNamespace()

    namespace = {
        "asyncio": SimpleNamespace(run_coroutine_threadsafe=schedule),
        "logger": SimpleNamespace(debug=lambda *_args, **_kwargs: None),
    }
    exec(compile(module, str(STATE_PATH), "exec"), namespace)
    namespace["scheduled"] = scheduled
    return namespace


def load_result_cleanup(websocket_is_closed):
    tree = ast.parse(
        RESULT_PROCESSOR_PATH.read_text(encoding="utf-8"),
        filename=str(RESULT_PROCESSOR_PATH),
    )
    result_processor = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ResultProcessor"
    )
    cleanup = next(
        node
        for node in result_processor.body
        if isinstance(node, ast.FunctionDef) and node.name == "_cleanup"
    )
    module = ast.fix_missing_locations(ast.Module(body=[cleanup], type_ignores=[]))
    namespace = {
        "logger": SimpleNamespace(debug=lambda *_args, **_kwargs: None),
        "websocket_is_closed": websocket_is_closed,
    }
    exec(compile(module, str(RESULT_PROCESSOR_PATH), "exec"), namespace)
    return namespace["_cleanup"]


class ModernWebSocket:
    def __init__(self, state: str) -> None:
        self.state = SimpleNamespace(name=state)
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1
        self.state = SimpleNamespace(name="CLOSED")


class ClientWebSocketStateTest(unittest.TestCase):
    def test_modern_and_legacy_connection_states_are_normalized(self) -> None:
        namespace = load_state_surface()
        is_open = namespace["websocket_is_open"]
        is_closed = namespace["websocket_is_closed"]

        self.assertTrue(is_open(ModernWebSocket("OPEN")))
        self.assertFalse(is_open(ModernWebSocket("CLOSING")))
        self.assertFalse(is_open(ModernWebSocket("CLOSED")))
        self.assertTrue(is_closed(ModernWebSocket("CLOSED")))
        self.assertTrue(is_open(SimpleNamespace(closed=False)))
        self.assertTrue(is_closed(SimpleNamespace(closed=True)))
        self.assertFalse(is_open(SimpleNamespace()))

    def test_closed_modern_connection_is_not_reported_connected(self) -> None:
        namespace = load_state_surface()
        state = namespace["ProbeState"]()
        state.websocket = ModernWebSocket("CLOSED")

        self.assertFalse(state.is_connected)

    def test_reset_closes_open_modern_connection_before_dropping_it(self) -> None:
        namespace = load_state_surface()
        state = namespace["ProbeState"]()
        websocket = ModernWebSocket("OPEN")
        state.websocket = websocket
        state.app = SimpleNamespace(loop=SimpleNamespace(is_running=lambda: True))
        state.stream = None
        state.recording = True
        state.recording_start_time = 1.0
        state.audio_files = {"task": Path("recording.wav")}

        state.reset()

        self.assertIsNone(state.websocket)
        self.assertEqual(len(namespace["scheduled"]), 1)
        asyncio.run(namespace["scheduled"][0])
        self.assertEqual(websocket.close_calls, 1)

    def test_result_cleanup_retains_open_and_clears_closed_modern_socket(self) -> None:
        namespace = load_state_surface()
        cleanup = load_result_cleanup(namespace["websocket_is_closed"])

        open_socket = ModernWebSocket("OPEN")
        open_processor = SimpleNamespace(
            state=SimpleNamespace(websocket=open_socket)
        )
        cleanup(open_processor)
        self.assertIs(open_processor.state.websocket, open_socket)

        closed_processor = SimpleNamespace(
            state=SimpleNamespace(websocket=ModernWebSocket("CLOSED"))
        )
        cleanup(closed_processor)
        self.assertIsNone(closed_processor.state.websocket)


if __name__ == "__main__":
    unittest.main()
