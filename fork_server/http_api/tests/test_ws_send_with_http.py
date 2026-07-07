# coding: utf-8

from __future__ import annotations

import ast
import asyncio
import importlib
import json
import sys
import types
import unittest
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


@dataclass
class Result:
    task_id: str
    socket_id: str
    type: str
    duration: float = 0.0
    time_start: float = 0.0
    time_submit: float = 0.0
    time_complete: float = 0.0
    text: str = ""
    text_accu: str = ""
    tokens: list[str] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    is_final: bool = False


@dataclass
class RecognitionMessage:
    task_id: str
    is_final: bool
    duration: float
    time_start: float
    time_submit: float
    time_complete: float
    text: str
    text_accu: str = ""
    tokens: list[str] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


async def immediate_to_thread(func, /, *args, **kwargs):
    return func(*args, **kwargs)


def load_ws_send_module():
    logger = SimpleNamespace(
        info=Mock(),
        debug=Mock(),
        warning=Mock(),
        error=Mock(),
    )

    core_module = types.ModuleType("core")
    server_module = types.ModuleType("core.server")
    server_module.logger = logger
    schema_module = types.ModuleType("core.server.schema")
    schema_module.Result = Result
    protocol_module = types.ModuleType("core.protocol")
    protocol_module.RecognitionMessage = RecognitionMessage
    tools_module = types.ModuleType("core.tools")
    to_thread_module = types.ModuleType("core.tools.asyncio_to_thread")
    to_thread_module.to_thread = immediate_to_thread
    state_module = types.ModuleType("core.server.state")
    state_module.console = SimpleNamespace(print=Mock())

    fake_modules = {
        "core": core_module,
        "core.server": server_module,
        "core.server.schema": schema_module,
        "core.protocol": protocol_module,
        "core.tools": tools_module,
        "core.tools.asyncio_to_thread": to_thread_module,
        "core.server.state": state_module,
    }

    module_name = "fork_server.http_api.ws_send_with_http"
    sys.modules.pop(module_name, None)
    with patch.dict(sys.modules, fake_modules):
        return importlib.import_module(module_name)


def _load_async_function(path: Path, name: str) -> ast.AsyncFunctionDef:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in module.body:
        if isinstance(node, ast.AsyncFunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")


def _is_http_router_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return (
        isinstance(test, ast.Call)
        and isinstance(test.func, ast.Attribute)
        and test.func.attr == "try_resolve"
        and isinstance(test.func.value, ast.Name)
        and test.func.value.id == "task_router"
    )


def _is_log_or_console_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    owner = node.func.value
    return isinstance(owner, ast.Name) and owner.id in {"logger", "console"}


class _LogTextNormalizer(ast.NodeTransformer):
    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, str):
            return ast.copy_location(ast.Constant(value="<log text>"), node)
        return node


class _WsSendCanonicalizer(ast.NodeTransformer):
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node = self.generic_visit(node)
        node.name = "ws_send"
        node.decorator_list = []
        node.returns = None
        return node

    def visit_If(self, node: ast.If) -> ast.AST | None:
        if _is_http_router_guard(node):
            return None
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        if _is_log_or_console_call(node):
            node.args = [_LogTextNormalizer().visit(arg) for arg in node.args]
            node.keywords = [
                ast.keyword(
                    arg=keyword.arg,
                    value=_LogTextNormalizer().visit(keyword.value),
                )
                for keyword in node.keywords
            ]
        return node


def _canonical_ws_send_ast(path: Path, name: str) -> str:
    node = _load_async_function(path, name)
    canonical = _WsSendCanonicalizer().visit(node)
    ast.fix_missing_locations(canonical)
    return ast.dump(canonical, include_attributes=False)


class FakeQueue:
    def __init__(self, items) -> None:
        self._items = list(items)

    def get(self):
        return self._items.pop(0)


class FakeWebSocket:
    def __init__(self, socket_id: str) -> None:
        self.id = socket_id
        self.sent: list[str] = []

    async def send(self, payload: str) -> None:
        self.sent.append(payload)


class WsSendWithHttpTest(unittest.TestCase):
    def tearDown(self) -> None:
        sys.modules.pop("fork_server.http_api.ws_send_with_http", None)

    def test_http_aware_ws_send_tracks_upstream_loop(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        upstream = _canonical_ws_send_ast(
            repo_root / "core/server/connection/ws_send.py",
            "ws_send",
        )
        forked = _canonical_ws_send_ast(
            repo_root / "fork_server/http_api/ws_send_with_http.py",
            "ws_send_with_http",
        )

        self.assertEqual(upstream, forked)

    def test_websocket_result_uses_upstream_result_type_field(self) -> None:
        ws_send_with_http = load_ws_send_module()
        websocket = FakeWebSocket("socket-1")
        result = Result(
            task_id="task-1",
            socket_id="socket-1",
            type="mic",
            duration=1.25,
            text="hello",
            is_final=True,
        )
        app = SimpleNamespace(
            state=SimpleNamespace(
                queue_out=FakeQueue([result, None]),
                sockets={"socket-1": websocket},
            )
        )

        with (
            patch.object(ws_send_with_http.task_router, "try_resolve", return_value=False),
            patch.object(ws_send_with_http.logger, "error") as log_error,
        ):
            asyncio.run(ws_send_with_http.ws_send_with_http(app))

        self.assertEqual(len(websocket.sent), 1)
        payload = json.loads(websocket.sent[0])
        self.assertEqual(payload["task_id"], "task-1")
        self.assertEqual(payload["text"], "hello")
        log_error.assert_not_called()

    def test_http_result_is_resolved_without_websocket_send(self) -> None:
        ws_send_with_http = load_ws_send_module()
        websocket = FakeWebSocket("socket-1")
        result = Result(
            task_id="task-1",
            socket_id="http:task-1",
            type="file",
            duration=1.25,
            text="hello",
            is_final=True,
        )
        app = SimpleNamespace(
            state=SimpleNamespace(
                queue_out=FakeQueue([result, None]),
                sockets={"socket-1": websocket},
            )
        )

        with patch.object(
            ws_send_with_http.task_router,
            "try_resolve",
            return_value=True,
        ) as try_resolve:
            asyncio.run(ws_send_with_http.ws_send_with_http(app))

        try_resolve.assert_called_once_with(result)
        self.assertEqual(websocket.sent, [])


if __name__ == "__main__":
    unittest.main()
