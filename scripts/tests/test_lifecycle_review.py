"""Regression coverage for service signals and unattended startup failures."""

from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SERVER_DEPS_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("rich", "websockets", "numpy")
)


def load_signal_module():
    spec = importlib.util.spec_from_file_location(
        "review_signal_handler", ROOT / "core/tools/signal_handler.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SignalHandlerTest(unittest.TestCase):
    def test_interactive_sigint_requires_two_presses_even_just_after_registration(self):
        module = load_signal_module()
        stopped = []
        with patch.object(sys, "stdin", TtyInput()):
            handler = module.SignalHandler(lambda: stopped.append(True))
            handler(signal.SIGINT, None)
            self.assertEqual(stopped, [])
            handler(signal.SIGINT, None)
            self.assertEqual(stopped, [True])
            handler(signal.SIGINT, None)
            self.assertEqual(stopped, [True])

    def test_noninteractive_sigint_stops_immediately(self):
        module = load_signal_module()
        stopped = []
        with patch.object(sys, "stdin", io.StringIO()):
            handler = module.SignalHandler(lambda: stopped.append(True))
            # An old first-press timestamp must not delay unattended shutdown.
            handler.last_time = 0
            handler(signal.SIGINT, None)
        self.assertEqual(stopped, [True])

    def test_expired_sigint_confirmation_requires_another_press(self):
        module = load_signal_module()
        stopped = []
        with (
            patch.object(sys, "stdin", TtyInput()),
            patch.object(module.time, "monotonic", side_effect=[100, 102, 102.5]),
        ):
            handler = module.SignalHandler(lambda: stopped.append(True))
            handler(signal.SIGINT, None)
            handler(signal.SIGINT, None)
            self.assertEqual(stopped, [])
            handler(signal.SIGINT, None)
            self.assertEqual(stopped, [True])

    def test_sigterm_is_immediate_and_callback_is_not_reentered(self):
        module = load_signal_module()
        stopped = []

        def stop():
            stopped.append(True)
            handler(signal.SIGTERM, None)

        with patch.object(sys, "stdin", TtyInput()):
            handler = module.SignalHandler(stop)
            handler.last_time = 0
            handler(signal.SIGTERM, None)
            self.assertEqual(stopped, [True])
            handler(signal.SIGTERM, None)
        self.assertEqual(stopped, [True])

    def test_sigterm_does_not_depend_on_a_live_stdin(self):
        module = load_signal_module()
        stopped = []
        closed_input = io.StringIO()
        closed_input.close()
        with patch.object(sys, "stdin", closed_input):
            module.SignalHandler(lambda: stopped.append(True))(signal.SIGTERM, None)
        self.assertEqual(stopped, [True])

    def test_sigterm_cleans_up_once_when_stdout_is_closed(self):
        output = io.StringIO()
        output.close()
        self.assert_sigterm_cleanup_with_stdout(output)

    def test_sigterm_cleans_up_once_when_output_pipe_reader_has_closed(self):
        read_fd, write_fd = os.pipe()
        os.close(read_fd)
        with os.fdopen(write_fd, "wb", buffering=0) as raw_output:
            with io.TextIOWrapper(
                raw_output, encoding="utf-8", write_through=True
            ) as output:
                self.assert_sigterm_cleanup_with_stdout(output)

    def assert_sigterm_cleanup_with_stdout(self, output):
        module = load_signal_module()
        stopped = []

        def stop():
            stopped.append(True)
            handler(signal.SIGTERM, None)

        handler = module.SignalHandler(stop)
        with patch.object(sys, "stdout", output):
            handler(signal.SIGTERM, None)
            handler(signal.SIGTERM, None)
        self.assertEqual(stopped, [True])


class TtyInput(io.StringIO):
    def isatty(self):
        return True


class ErrorPauseTest(unittest.TestCase):
    def test_enabled_terminal_pause_consumes_enter(self):
        pause = runpy.run_path(str(ROOT / "core/tools/interactive.py"))["pause_on_error"]
        stdin = TtyInput("\n")
        with patch.dict(os.environ, {}, clear=True), patch.object(sys, "stdin", stdin):
            pause("Press Enter")
        self.assertEqual(stdin.tell(), 1)

    def test_disabled_aliases_skip_terminal_input(self):
        pause = runpy.run_path(str(ROOT / "core/tools/interactive.py"))["pause_on_error"]
        for value in ("false", "0", "no", "off", " FALSE "):
            with self.subTest(value=value):
                stdin = TtyInput("\n")
                with (
                    patch.dict(os.environ, {"CAPSWRITER_INTERACTIVE_ERRORS": value}),
                    patch.object(sys, "stdin", stdin),
                ):
                    pause("Press Enter")
                self.assertEqual(stdin.tell(), 0)


@unittest.skipUnless(SERVER_DEPS_AVAILABLE, "server runtime dependencies not installed")
class StartupFailureTest(unittest.TestCase):
    def test_actual_failure_paths_do_not_wait_for_unattended_input(self):
        for failure in ("occupied_port", "missing_model", "unsupported_model"):
            for mode in ("disabled", "non_tty", "eof"):
                with self.subTest(failure=failure, mode=mode):
                    self.run_failure(failure, mode)

    def run_failure(self, failure, mode):
        code = textwrap.dedent("""
            import asyncio
            import io
            import os
            from pathlib import Path
            import socket
            import sys
            import tempfile
            from types import SimpleNamespace
            from config_server import ServerConfig as Config, ModelPaths

            class TtyInput(io.StringIO):
                def isatty(self):
                    return True
                def readline(self, *args):
                    if sys.argv[2] == "disabled":
                        raise AssertionError("disabled error pause read stdin")
                    return super().readline(*args)

            if sys.argv[2] != "non_tty":
                sys.stdin = TtyInput()
            if sys.argv[1] == "occupied_port":
                from core.server.connection.server_manager import SocketManager
                with socket.socket() as occupied:
                    occupied.bind(("127.0.0.1", 0))
                    occupied.listen()
                    Config.addr, Config.port = occupied.getsockname()
                    manager = SocketManager(SimpleNamespace())
                    asyncio.run(manager.start())
                    assert not manager._is_running
            else:
                from core.server.worker.check_model import check_model
                with tempfile.TemporaryDirectory() as temp:
                    Config.model_type = (
                        "paraformer" if sys.argv[1] == "missing_model" else "invalid"
                    )
                    ModelPaths.paraformer_dir = Path(temp)
                    ModelPaths.paraformer_model = Path(temp) / "missing.onnx"
                    ModelPaths.paraformer_tokens = Path(temp) / "missing.txt"
                    check_model()
        """)
        env = dict(os.environ)
        env.pop("CAPSWRITER_INTERACTIVE_ERRORS", None)
        if mode == "disabled":
            env["CAPSWRITER_INTERACTIVE_ERRORS"] = "false"
        with subprocess.Popen(
            [sys.executable, "-c", code, failure, mode],
            cwd=ROOT, env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ) as process:
            try:
                # Keep the input pipe open: unconditional input() would block.
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                self.fail(f"{failure} blocked waiting for {mode} input")
            stdout, stderr = process.communicate()
        self.assertEqual(process.returncode, 0 if failure == "occupied_port" else 1,
                         stdout + stderr)
        self.assertNotIn("Traceback", stderr)


@unittest.skipUnless(os.name == "posix" and SERVER_DEPS_AVAILABLE,
                     "requires POSIX process signals and server runtime dependencies")
class ServerSignalSubprocessTest(unittest.TestCase):
    def test_sigterm_reaps_worker_and_releases_websocket_listener(self):
        code = textwrap.dedent("""
            import asyncio
            import json
            import multiprocessing
            from pathlib import Path
            import signal
            import sys
            import types
            from config_server import ServerConfig as Config
            from fork_server.bootstrap import create_server

            # The headless fixture needs no desktop toast/Tk dependencies.
            # Keep the real tray manager; it honors Config.enable_tray=False.
            ui_package = types.ModuleType("core.server.ui")
            ui_package.__path__ = [str(Path.cwd() / "core/server/ui")]
            sys.modules["core.server.ui"] = ui_package

            def worker(queue):
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                queue.get()

            Config.addr = "127.0.0.1"
            Config.port = 0
            Config.enable_tray = False
            Config.http_api_enable = False
            server = create_server()
            manager = server.process_manager

            def start_worker_without_models():
                manager.is_alive = True
                manager._process = multiprocessing.get_context("fork").Process(
                    target=worker, args=(server.state.queue_in,), daemon=True
                )
                manager._process.start()

            manager.start = start_worker_without_models
            ready = Path(sys.argv[1])

            def report_ready():
                listener = server.socket_manager._server
                if listener is None:
                    server.loop.call_later(0.02, report_ready)
                    return
                temporary = ready.with_suffix(".tmp")
                temporary.write_text(json.dumps({
                    "port": listener.sockets[0].getsockname()[1],
                    "worker_pid": manager._process.pid,
                }))
                temporary.replace(ready)

            server.loop.call_soon(report_ready)
            server.start()
            assert not manager._process.is_alive(), "recognizer child survived stop"
            server.loop.run_until_complete(server.socket_manager._server.wait_closed())
            server.loop.close()
            ready.with_suffix(".stopped").write_text("cleaned")
        """)
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryFile() as output:
            ready = Path(temp) / "ready.json"
            process = subprocess.Popen(
                [sys.executable, "-c", code, str(ready)], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=output, stderr=output,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 15
                while (
                    not ready.exists()
                    and process.poll() is None
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.02)
                output.seek(0)
                self.assertTrue(ready.exists(), output.read().decode(errors="replace"))
                data = json.loads(ready.read_text())
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    output.seek(0)
                    self.fail("SIGTERM cleanup timed out: " + output.read().decode(errors="replace"))
                output.seek(0)
                self.assertEqual(process.returncode, 0, output.read().decode(errors="replace"))
                self.assertTrue(ready.with_suffix(".stopped").exists())
                with self.assertRaises(ProcessLookupError):
                    os.kill(data["worker_pid"], 0)
                with socket.socket() as listener:
                    listener.bind(("127.0.0.1", data["port"]))
            finally:
                # The failing regression must not leave its orphan child behind.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
