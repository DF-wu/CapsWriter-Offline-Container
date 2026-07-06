# coding: utf-8

from __future__ import annotations

import asyncio
import importlib
import json
import sys
import types
import unittest
from dataclasses import asdict, dataclass, field
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
