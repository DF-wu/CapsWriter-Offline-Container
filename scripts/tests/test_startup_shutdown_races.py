"""Deterministic regressions for startup cancellation and tray-thread exit."""

from __future__ import annotations

import asyncio
import importlib.util
import signal
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SERVER_DEPS_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("rich", "numpy", "websockets")
)


def _sigterm(callback) -> None:
    from core.tools.signal_handler import SignalHandler

    SignalHandler(callback)(signal.SIGTERM, None)


class _ActiveInference:
    def __init__(self) -> None:
        self.values = [0.0, 0.0]
        self.lock = threading.Lock()

    def __getitem__(self, index):
        return self.values[index]

    def __setitem__(self, index, value) -> None:
        self.values[index] = value

    def get_lock(self):
        return self.lock


class _FakeManager:
    def __init__(self, on_list=None) -> None:
        self.on_list = on_list
        self.shutdown_calls = 0

    def list(self):
        if self.on_list is not None:
            self.on_list()
        return []

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class _FakeProcess:
    def __init__(self, on_start=None) -> None:
        self.on_start = on_start
        self.pid = None
        self.start_calls = 0
        self.terminate_calls = 0
        self.join_calls = 0
        self._alive = False

    def start(self) -> None:
        self.start_calls += 1
        if self.on_start is not None:
            self.on_start()
        self.pid = 4312
        self._alive = True

    def is_alive(self) -> bool:
        if self.pid is None:
            raise AssertionError("can only test a started process")
        return self._alive

    def terminate(self) -> None:
        self.terminate_calls += 1

    def join(self, timeout=None) -> None:
        self.join_calls += 1
        self._alive = False

    def kill(self) -> None:
        self._alive = False


def _process_app():
    return SimpleNamespace(
        state=SimpleNamespace(
            queue_in=SimpleNamespace(put=lambda *_args, **_kwargs: None),
            queue_out=SimpleNamespace(get=lambda **_kwargs: True),
            sockets_id=None,
            recognize_process=None,
            recognizer_watchdog_failed=False,
            recognizer_active_inference=_ActiveInference(),
        ),
        stop=lambda: None,
        loop=None,
    )


@unittest.skipUnless(
    SERVER_DEPS_AVAILABLE,
    "server runtime dependencies not installed",
)
class ProcessStartupCancellationTest(unittest.TestCase):
    def setUp(self) -> None:
        from core.server.worker import process_manager

        self.module = process_manager
        self.manager = process_manager.ProcessManager(_process_app())
        self.stdin = SimpleNamespace(fileno=lambda: 0)

    def test_cancel_during_check_model_never_creates_manager_or_worker(self) -> None:
        calls = []

        def check_model() -> None:
            _sigterm(self.manager.stop)

        with (
            patch.object(self.module, "check_model", check_model),
            patch.object(self.module, "Manager", lambda: calls.append("manager")),
            patch.object(self.module, "Process", lambda **_kwargs: calls.append("process")),
        ):
            self.assertIsNone(self.manager.start())

        self.assertEqual(calls, [])
        self.assertFalse(self.manager.is_alive)

    def test_cancel_inside_manager_list_defers_cleanup_until_rpc_returns(self) -> None:
        fake_manager = _FakeManager(
            on_list=lambda: _sigterm(self.manager.stop)
        )
        process_calls = []
        with (
            patch.object(self.module, "check_model", lambda: None),
            patch.object(self.module, "Manager", lambda: fake_manager),
            patch.object(
                self.module,
                "Process",
                lambda **_kwargs: process_calls.append(True),
            ),
            patch.object(self.module.sys, "stdin", self.stdin),
        ):
            self.assertIsNone(self.manager.start())

        self.assertEqual(process_calls, [])
        self.assertEqual(fake_manager.shutdown_calls, 1)

    def test_cancel_while_process_object_is_built_never_calls_start(self) -> None:
        fake_manager = _FakeManager()
        process = _FakeProcess()

        def build_process(**_kwargs):
            _sigterm(self.manager.stop)
            return process

        with (
            patch.object(self.module, "check_model", lambda: None),
            patch.object(self.module, "Manager", lambda: fake_manager),
            patch.object(self.module, "Process", build_process),
            patch.object(self.module.sys, "stdin", self.stdin),
        ):
            self.assertIsNone(self.manager.start())

        self.assertEqual(process.start_calls, 0)
        self.assertEqual(fake_manager.shutdown_calls, 1)

    def test_reentrant_cancel_before_pid_assignment_reaps_started_worker(self) -> None:
        fake_manager = _FakeManager()
        process = _FakeProcess(on_start=lambda: _sigterm(self.manager.stop))
        with (
            patch.object(self.module, "check_model", lambda: None),
            patch.object(self.module, "Manager", lambda: fake_manager),
            patch.object(self.module, "Process", lambda **_kwargs: process),
            patch.object(self.module.sys, "stdin", self.stdin),
        ):
            self.assertIsNone(self.manager.start())

        self.assertEqual(process.start_calls, 1)
        self.assertFalse(process.is_alive())
        self.assertEqual(process.terminate_calls, 1)
        self.assertEqual(fake_manager.shutdown_calls, 1)


class _LifecycleComponent:
    def __init__(self, on_start=None) -> None:
        self.on_start = on_start
        self.start_calls = 0
        self.stop_calls = 0

    def start(self) -> None:
        self.start_calls += 1
        if self.on_start is not None:
            self.on_start()

    def stop(self) -> None:
        self.stop_calls += 1


@unittest.skipUnless(
    SERVER_DEPS_AVAILABLE,
    "server runtime dependencies not installed",
)
class ServerFacadeStartupCancellationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ui_package = types.ModuleType("core.server.ui")
        ui_package.__path__ = [str(ROOT / "core/server/ui")]
        with patch.dict(sys.modules, {"core.server.ui": ui_package}):
            from core.server.app import CapsWriterServer
            from fork_server.bootstrap import create_server

            fork_server = create_server()
            cls.server_types = (CapsWriterServer, type(fork_server))
            fork_server.loop.close()
            fork_server.state.queue_in.close()
            fork_server.state.queue_out.close()

    def _server(self, server_type, phase: str):
        server = server_type.__new__(server_type)
        server.is_alive = False
        server.loop = asyncio.new_event_loop()
        server.state = SimpleNamespace(
            queue_out=SimpleNamespace(put_nowait=lambda _item: None)
        )
        server.socket_manager = _LifecycleComponent()
        server.process_manager = _LifecycleComponent()
        server.tray_manager = _LifecycleComponent(
            on_start=(
                (lambda: _sigterm(server.stop)) if phase == "tray" else None
            )
        )
        server._print_banner = (
            (lambda: _sigterm(server.stop)) if phase == "banner" else lambda: None
        )
        return server

    def test_tray_and_banner_cancellation_never_start_worker(self) -> None:
        from config_server import ServerConfig as Config

        for server_type in self.server_types:
            for phase in ("tray", "banner"):
                with self.subTest(server=server_type.__name__, phase=phase):
                    server = self._server(server_type, phase)
                    try:
                        with (
                            patch("core.tools.signal_handler.register_signal"),
                            patch.object(
                                Config, "http_api_enable", False, create=True
                            ),
                        ):
                            server.start()
                        self.assertEqual(server.process_manager.start_calls, 0)
                        self.assertFalse(server.is_alive)
                    finally:
                        server.loop.close()


@unittest.skipUnless(
    SERVER_DEPS_AVAILABLE,
    "server runtime dependencies not installed",
)
class ServerProxyTeardownTest(unittest.TestCase):
    def test_ws_recv_finally_tolerates_captured_proxy_after_manager_shutdown(self):
        from core.server.connection.ws_recv import _ws_recv_admitted

        class CapturedProxy:
            def __init__(self) -> None:
                self.items = []
                self.dead = False

            def append(self, item) -> None:
                self.items.append(item)

            def __contains__(self, item) -> bool:
                if self.dead:
                    raise BrokenPipeError("manager stopped")
                return item in self.items

            def remove(self, item) -> None:
                self.items.remove(item)

        proxy = CapturedProxy()

        class WebSocket:
            id = "captured-proxy"
            remote_address = ("127.0.0.1", 6016)

            def __aiter__(self):
                return self

            async def __anext__(self):
                proxy.dead = True
                raise StopAsyncIteration

        state = SimpleNamespace(sockets={}, sockets_id=proxy)
        asyncio.run(_ws_recv_admitted(WebSocket(), SimpleNamespace(state=state)))

        self.assertEqual(state.sockets, {})
        self.assertEqual(proxy.items, ["captured-proxy"])


def _module(name: str, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    return module


class _Log:
    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


def _load_client_lifecycle_types():
    logger = _Log()
    console = SimpleNamespace(print=lambda *_args, **_kwargs: None)
    client_package = _module("core.client", logger=logger)
    client_package.__path__ = [str(ROOT / "core/client")]
    output_package = _module("core.client.output", logger=logger)
    output_package.__path__ = [str(ROOT / "core/client/output")]
    manager_package = _module(
        "core.client.manager",
        logger=logger,
        TrayManager=None,
        MicRunner=None,
        FileRunner=None,
    )
    manager_package.__path__ = [str(ROOT / "core/client/manager")]

    class _Placeholder:
        def __init__(self, *_args, **_kwargs):
            pass

    config = SimpleNamespace(log_level="INFO", udp_control=False)
    stubs = {
        "config_client": _module(
            "config_client", ClientConfig=config, __version__="test"
        ),
        "core.client": client_package,
        "core.client.output": output_package,
        "core.client.state": _module(
            "core.client.state",
            ClientState=_Placeholder,
            console=console,
            websocket_is_closed=lambda websocket: bool(
                getattr(websocket, "closed", False)
            ),
        ),
        "core.protocol": _module("core.protocol", RecognitionMessage=object),
        "core.client.output.text_output": _module(
            "core.client.output.text_output", TextOutput=_Placeholder
        ),
        "core.tools.window_detector": _module(
            "core.tools.window_detector", get_active_window_info=lambda: {}
        ),
        "keyboard": _module("keyboard", press_and_release=lambda *_args: None),
        "core.client.udp": _module("core.client.udp"),
        "core.client.udp.udp_broadcaster": _module(
            "core.client.udp.udp_broadcaster", broadcast_output_udp=lambda *_args: None
        ),
        "core.client.audio": _module("core.client.audio"),
        "core.client.audio.file_manager": _module(
            "core.client.audio.file_manager", AudioFileManager=_Placeholder
        ),
        "core.client.llm": _module("core.client.llm"),
        "core.client.llm.llm_write_md": _module(
            "core.client.llm.llm_write_md", write_llm_md=lambda *_args: None
        ),
        "core.tools.zhconv": _module(
            "core.tools.zhconv", convert=lambda text, _locale: text
        ),
        "core.client.connection": _module(
            "core.client.connection", WebSocketManager=_Placeholder
        ),
        "core.client.manager": manager_package,
        "core.client.ui": _module(
            "core.client.ui",
            TipsDisplay=SimpleNamespace(show_mic_tips=lambda: None),
        ),
        "core.client.audio.stream": _module(
            "core.client.audio.stream", AudioStreamManager=_Placeholder
        ),
        "core.client.shortcut": _module("core.client.shortcut"),
        "core.client.shortcut.shortcut_manager": _module(
            "core.client.shortcut.shortcut_manager", ShortcutManager=_Placeholder
        ),
        "core.client.shortcut.shortcut_config": _module(
            "core.client.shortcut.shortcut_config", Shortcut=_Placeholder
        ),
        "core.client.udp.udp_control": _module(
            "core.client.udp.udp_control", UDPController=_Placeholder
        ),
        "core.client.hotword": _module("core.client.hotword"),
        "core.client.hotword.manager": _module(
            "core.client.hotword.manager", HotwordManager=_Placeholder
        ),
        "core.client.llm.llm_handler": _module(
            "core.client.llm.llm_handler", LLMHandler=_Placeholder
        ),
        "core.client.diary": _module("core.client.diary"),
        "core.client.diary.diary_writer": _module(
            "core.client.diary.diary_writer", DiaryWriter=_Placeholder
        ),
        "core.tools.signal_handler": _module(
            "core.tools.signal_handler", register_signal=lambda *_args: None
        ),
        "core.tools.empty_working_set": _module(
            "core.tools.empty_working_set",
            empty_current_working_set=lambda: None,
        ),
    }

    def load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    with patch.dict(sys.modules, stubs):
        result_module = load(
            "core.client.output.result_processor",
            ROOT / "core/client/output/result_processor.py",
        )
        output_package.ResultProcessor = result_module.ResultProcessor
        mic_module = load(
            "core.client.manager.mic_runner",
            ROOT / "core/client/manager/mic_runner.py",
        )
        manager_package.MicRunner = mic_module.MicRunner
        manager_package.TrayManager = _Placeholder
        manager_package.FileRunner = _Placeholder
        app_module = load("core.client.app", ROOT / "core/client/app.py")
    runtime_modules = {
        "core.client": client_package,
        "core.client.output": output_package,
        "core.client.output.result_processor": result_module,
    }
    return (
        app_module.CapsWriterClient,
        result_module.ResultProcessor,
        runtime_modules,
    )


class _CountingLoop(asyncio.SelectorEventLoop):
    def __init__(self) -> None:
        super().__init__()
        self.writes = 0
        self.selects = 0
        self.condition = threading.Condition()
        original_select = self._selector.select

        def counted_select(timeout=None):
            with self.condition:
                self.selects += 1
                self.condition.notify_all()
            return original_select(timeout)

        self._selector.select = counted_select

    def _write_to_self(self) -> None:
        self.writes += 1
        super()._write_to_self()

    def wait_for_select(self, target: int, timeout: float = 1.0) -> None:
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.selects < target:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError(
                        f"loop reached {self.selects} selects, expected {target}"
                    )
                self.condition.wait(remaining)


class _ClientState:
    def __init__(self) -> None:
        self.websocket = None
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1
        self.websocket = None


class _ClientSocket:
    closed = False


class _ClientWebSocketManager:
    def __init__(self, app, *, block_connect=False, gate_message=False) -> None:
        self.app = app
        self.block_connect = block_connect
        self.gate_message = gate_message
        self.connect_calls = 0
        self.close_calls = 0
        self.connect_started = threading.Event()
        self.receive_started = threading.Event()
        self.message_release = None

    async def connect(self) -> bool:
        self.connect_calls += 1
        self.connect_started.set()
        if self.block_connect:
            await asyncio.Event().wait()
        self.app.state.websocket = _ClientSocket()
        return True

    async def receive(self):
        self.receive_started.set()
        if self.gate_message:
            self.message_release = asyncio.Event()
            await self.message_release.wait()
            return object()
        await asyncio.Event().wait()

    async def close(self) -> None:
        self.close_calls += 1
        self.app.state.websocket = None


class _NoopStop:
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class _ConsumeFirstWakeup:
    def __init__(self, loop, processor, prior_selects) -> None:
        self.loop = loop
        self.processor = processor
        self.prior_selects = prior_selects

    def stop(self) -> None:
        deadline = time.monotonic() + 1
        while not self.processor._exit_event.is_set():
            if time.monotonic() >= deadline:
                raise AssertionError("processor exit wakeup was not consumed")
            time.sleep(0.001)
        self.loop.wait_for_select(self.prior_selects + 1)


class ClientTrayExitRaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        (
            cls.client_type,
            cls.processor_type,
            cls.runtime_modules,
        ) = _load_client_lifecycle_types()

    def _client(self, loop, **ws_options):
        client = self.client_type.__new__(self.client_type)
        client.loop = loop
        client.state = _ClientState()
        client.ws = _ClientWebSocketManager(client, **ws_options)
        client._result_processor = None
        client._runner_task = None
        client._shutdown_future = None
        client._shutdown_lock = threading.Lock()
        client._stop_requested = threading.Event()
        client.shortcut = client.stream = client.tray = _NoopStop()
        client.hotword = client.llm = _NoopStop()
        client.udp = _NoopStop()
        return client

    def _start_client_thread(self, client, argv):
        result = {}

        def run_client() -> None:
            asyncio.set_event_loop(client.loop)
            try:
                with (
                    patch.object(sys, "argv", argv),
                    patch.dict(sys.modules, self.runtime_modules),
                ):
                    client.start()
            except BaseException as error:
                result["error"] = error

        thread = threading.Thread(target=run_client)
        thread.start()
        return thread, result

    def test_connected_tray_exit_consumes_old_wakeup_without_reconnect(self) -> None:
        loop = _CountingLoop()
        client = self._client(loop)
        thread, result = self._start_client_thread(client, ["start_client.py"])
        self.assertTrue(client.ws.receive_started.wait(1))
        prior_selects = loop.selects
        client.udp = _ConsumeFirstWakeup(
            loop, client._result_processor, prior_selects
        )

        client.stop()  # pystray invokes this from outside the asyncio thread
        thread.join(2)
        if thread.is_alive():
            loop.call_soon_threadsafe(loop.stop)
            thread.join(1)

        try:
            self.assertFalse(thread.is_alive(), "tray exit did not stop the loop")
            self.assertEqual(client.ws.connect_calls, 1)
            self.assertEqual(client.ws.close_calls, 1)
            self.assertEqual(client.state.reset_calls, 1)
            self.assertGreaterEqual(loop.writes, 2)
            self.assertFalse(asyncio.all_tasks(loop))
            self.assertNotIn("error", result)
            self.assertTrue(client._shutdown_future.done())
        finally:
            loop.close()

    def test_tray_exit_cancels_connect_in_progress(self) -> None:
        loop = _CountingLoop()
        client = self._client(loop, block_connect=True)
        thread, result = self._start_client_thread(client, ["start_client.py"])
        self.assertTrue(client.ws.connect_started.wait(1))

        client.stop()
        thread.join(2)
        try:
            self.assertFalse(thread.is_alive(), "connect cancellation hung shutdown")
            self.assertEqual(client.ws.connect_calls, 1)
            self.assertEqual(client.ws.close_calls, 1)
            self.assertEqual(client.state.reset_calls, 1)
            self.assertFalse(asyncio.all_tasks(loop))
            self.assertNotIn("error", result)
            self.assertTrue(client._shutdown_future.done())
        finally:
            loop.close()

    def test_tray_exit_cancels_long_message_handler(self) -> None:
        loop = _CountingLoop()
        client = self._client(loop, gate_message=True)
        thread, result = self._start_client_thread(client, ["start_client.py"])
        self.assertTrue(client.ws.receive_started.wait(1))
        deadline = time.monotonic() + 1
        while client._result_processor is None:
            if time.monotonic() >= deadline:
                self.fail("result processor was not published")
            time.sleep(0.001)

        handler_started = threading.Event()

        async def long_handler(_message) -> None:
            handler_started.set()
            await asyncio.Event().wait()

        client._result_processor._handle_message = long_handler
        loop.call_soon_threadsafe(client.ws.message_release.set)
        self.assertTrue(handler_started.wait(1))

        client.stop()
        thread.join(2)
        try:
            self.assertFalse(thread.is_alive(), "message handler cancellation hung")
            self.assertEqual(client.ws.connect_calls, 1)
            self.assertEqual(client.ws.close_calls, 1)
            self.assertEqual(client.state.reset_calls, 1)
            self.assertFalse(asyncio.all_tasks(loop))
            self.assertNotIn("error", result)
            self.assertTrue(client._shutdown_future.done())
        finally:
            loop.close()

    def test_file_mode_without_result_processor_still_stops_loop(self) -> None:
        loop = _CountingLoop()
        client = self._client(loop)
        started = threading.Event()

        class BlockingFileRunner:
            def __init__(self, _app, _files) -> None:
                pass

            async def run(self) -> None:
                started.set()
                await asyncio.Event().wait()

        globals_ = self.client_type.start.__globals__
        original_file_runner = globals_["FileRunner"]
        globals_["FileRunner"] = BlockingFileRunner
        try:
            thread, result = self._start_client_thread(
                client, ["start_client.py", "recording.wav"]
            )
            self.assertTrue(started.wait(1))
            client.stop()
            thread.join(2)
            self.assertFalse(thread.is_alive(), "file-mode stop did not wake the loop")
            self.assertEqual(client.ws.close_calls, 1)
            self.assertEqual(client.state.reset_calls, 1)
            self.assertFalse(asyncio.all_tasks(loop))
            self.assertNotIn("error", result)
            self.assertTrue(client._shutdown_future.done())
        finally:
            globals_["FileRunner"] = original_file_runner
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.close()


if __name__ == "__main__":
    unittest.main()
