# coding: utf-8

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fork_server.bootstrap import (
    _run_fork_services,
    _stop_http_enabled_server,
)


SERVER_DEPS_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("rich", "uvicorn")
)


class _LoopFacade:
    def __init__(self, loop) -> None:
        self._loop = loop
        self.stop = MagicMock()

    def call_soon_threadsafe(self, callback) -> None:
        self._loop.call_soon_threadsafe(callback)


class _SocketManager:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.stop_calls = 0

    async def start(self) -> None:
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.cancelled.set()

    def stop(self) -> None:
        self.stop_calls += 1


def _server(loop) -> SimpleNamespace:
    return SimpleNamespace(
        is_alive=True,
        _http_stop_requested=False,
        _http_server=None,
        _http_shutdown_event=asyncio.Event(),
        loop=_LoopFacade(loop),
        socket_manager=_SocketManager(),
        process_manager=SimpleNamespace(stop=MagicMock()),
        tray_manager=SimpleNamespace(stop=MagicMock()),
    )


def _patched_core_server():
    core_module = types.ModuleType("core")
    core_module.__path__ = []
    server_module = types.ModuleType("core.server")
    server_module.logger = SimpleNamespace(info=MagicMock())
    server_module.console = SimpleNamespace(print=MagicMock())
    return patch.dict(
        sys.modules,
        {"core": core_module, "core.server": server_module},
    )


class ForkServerLifecycleTest(unittest.TestCase):
    def test_explicit_stop_waits_for_http_teardown_without_stopping_loop(self) -> None:
        async def scenario() -> None:
            server = _server(asyncio.get_running_loop())
            http_started = asyncio.Event()
            http_torn_down = asyncio.Event()
            handle = SimpleNamespace(should_exit=False, force_exit=False)

            async def run_http(cw_server) -> None:
                cw_server._http_server = handle
                http_started.set()
                try:
                    while not handle.should_exit:
                        await asyncio.sleep(0)
                finally:
                    http_torn_down.set()
                    cw_server._http_server = None

            runner = asyncio.create_task(_run_fork_services(server, run_http))
            await asyncio.wait_for(server.socket_manager.started.wait(), timeout=0.5)
            await asyncio.wait_for(http_started.wait(), timeout=0.5)
            _stop_http_enabled_server(server)
            await asyncio.wait_for(runner, timeout=0.5)

            self.assertTrue(http_torn_down.is_set())
            self.assertTrue(server.socket_manager.cancelled.is_set())
            self.assertTrue(handle.should_exit)
            server.loop.stop.assert_not_called()
            server.process_manager.stop.assert_called_once_with()
            server.tray_manager.stop.assert_called_once_with()

        with _patched_core_server():
            asyncio.run(scenario())

    def test_http_task_exit_stops_and_reaps_websocket_service(self) -> None:
        async def scenario() -> None:
            server = _server(asyncio.get_running_loop())
            http_started = asyncio.Event()
            simulate_sigterm = asyncio.Event()

            async def run_http(cw_server) -> None:
                handle = SimpleNamespace(should_exit=False, force_exit=False)
                cw_server._http_server = handle
                http_started.set()
                try:
                    await simulate_sigterm.wait()
                finally:
                    cw_server._http_server = None

            runner = asyncio.create_task(_run_fork_services(server, run_http))
            await asyncio.wait_for(server.socket_manager.started.wait(), timeout=0.5)
            await asyncio.wait_for(http_started.wait(), timeout=0.5)
            simulate_sigterm.set()
            await asyncio.wait_for(runner, timeout=0.5)

            self.assertFalse(server.is_alive)
            self.assertTrue(server.socket_manager.cancelled.is_set())
            server.process_manager.stop.assert_called_once_with()
            server.tray_manager.stop.assert_called_once_with()
            server.loop.stop.assert_not_called()

        with _patched_core_server():
            asyncio.run(scenario())

    def test_service_runner_cancellation_still_reaps_both_services(self) -> None:
        async def scenario() -> None:
            server = _server(asyncio.get_running_loop())
            http_started = asyncio.Event()
            http_torn_down = asyncio.Event()
            handle = SimpleNamespace(should_exit=False, force_exit=False)

            async def run_http(cw_server) -> None:
                cw_server._http_server = handle
                http_started.set()
                try:
                    while not handle.should_exit:
                        await asyncio.sleep(0)
                finally:
                    http_torn_down.set()
                    cw_server._http_server = None

            runner = asyncio.create_task(_run_fork_services(server, run_http))
            await asyncio.wait_for(server.socket_manager.started.wait(), timeout=0.5)
            await asyncio.wait_for(http_started.wait(), timeout=0.5)
            runner.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(runner, timeout=0.5)

            self.assertTrue(handle.should_exit)
            self.assertTrue(http_torn_down.is_set())
            self.assertTrue(server.socket_manager.cancelled.is_set())
            server.process_manager.stop.assert_called_once_with()
            server.tray_manager.stop.assert_called_once_with()

        with _patched_core_server():
            asyncio.run(scenario())

    def test_repeated_service_runner_cancellation_still_reaps_both_services(self) -> None:
        async def scenario() -> None:
            server = _server(asyncio.get_running_loop())
            http_started = asyncio.Event()
            http_cleanup_started = asyncio.Event()
            release_http_cleanup = asyncio.Event()
            http_torn_down = asyncio.Event()
            handle = SimpleNamespace(should_exit=False, force_exit=False)

            async def run_http(cw_server) -> None:
                cw_server._http_server = handle
                http_started.set()
                try:
                    while not handle.should_exit:
                        await asyncio.sleep(0)
                finally:
                    http_cleanup_started.set()
                    await release_http_cleanup.wait()
                    http_torn_down.set()
                    cw_server._http_server = None

            runner = asyncio.create_task(_run_fork_services(server, run_http))
            await asyncio.wait_for(server.socket_manager.started.wait(), timeout=0.5)
            await asyncio.wait_for(http_started.wait(), timeout=0.5)
            runner.cancel()
            await asyncio.wait_for(http_cleanup_started.wait(), timeout=0.5)

            try:
                for _ in range(3):
                    runner.cancel()
                    await asyncio.sleep(0)
                    self.assertFalse(runner.done())
            finally:
                release_http_cleanup.set()

            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(runner, timeout=0.5)

            self.assertTrue(handle.should_exit)
            self.assertTrue(http_torn_down.is_set())
            self.assertTrue(server.socket_manager.cancelled.is_set())
            server.process_manager.stop.assert_called_once_with()
            server.tray_manager.stop.assert_called_once_with()

        with _patched_core_server():
            asyncio.run(scenario())

    @unittest.skipUnless(
        SERVER_DEPS_AVAILABLE,
        "server HTTP runtime dependencies are not installed",
    )
    def test_uvicorn_uses_finite_graceful_shutdown_timeout(self) -> None:
        from fork_server.http_api import serve

        observed = {}

        class FakeConfig:
            def __init__(self, app, **kwargs) -> None:
                observed["app"] = app
                observed["config"] = kwargs
                self.host = kwargs["host"]
                self.port = kwargs["port"]

        class FakeServer:
            def __init__(self, config) -> None:
                self.config = config
                self.should_exit = False
                self.force_exit = False

            async def serve(self) -> None:
                observed["served"] = True

        fake_uvicorn = types.ModuleType("uvicorn")
        fake_uvicorn.Config = FakeConfig
        fake_uvicorn.Server = FakeServer

        async def scenario() -> None:
            cw_server = SimpleNamespace(
                state=SimpleNamespace(),
                _http_stop_requested=False,
                _http_server=None,
            )
            with (
                patch.dict(sys.modules, {"uvicorn": fake_uvicorn}),
                patch.object(serve, "create_app", return_value=object()),
                patch.object(serve.task_router, "bind"),
                patch.object(serve.shutil, "which", return_value=None),
            ):
                await serve.run_http_server(cw_server)
            self.assertIsNone(cw_server._http_server)

        asyncio.run(scenario())
        self.assertTrue(observed["served"])
        self.assertEqual(observed["config"]["timeout_graceful_shutdown"], 5)


if __name__ == "__main__":
    unittest.main()
