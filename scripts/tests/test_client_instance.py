"""跨程序驗證互斥、崩潰後重新啟動，以及入口在載入硬體前拒絕多開。"""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import start_client
from fork_client.instance import ClientInstance, RESTART_WAIT_ARGUMENT, report_status


ROOT = Path(__file__).resolve().parents[2]


class ClientInstanceTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "client-instance.lock"

    def worker(self, mode="hold", *args):
        process = subprocess.Popen(
            [sys._base_executable, "-m", "scripts.tests.test_client_instance", "--worker", mode, str(self.path), *args],
            cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", env={
                **os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": os.pathsep.join(sys.path),
            },
        )

        def cleanup():
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=5)

        self.addCleanup(cleanup)
        return process

    def line(self, process):
        result = queue.Queue()
        threading.Thread(target=lambda: result.put(process.stdout.readline()), daemon=True).start()
        return result.get(timeout=5).strip()

    def test_two_processes_cannot_hold_lock_and_owner_pid_is_readable(self):
        owner = self.worker()
        self.assertEqual(self.line(owner), "acquired")
        contender = ClientInstance(self.path)
        self.assertFalse(contender.acquire())
        self.assertEqual(contender.owner()["pid"], owner.pid)
        self.assertIsNone(owner.poll())

    def test_crashed_owner_does_not_leave_a_stale_lock(self):
        owner = self.worker()
        self.assertEqual(self.line(owner), "acquired")
        owner.kill()
        owner.wait(timeout=5)
        replacement = ClientInstance(self.path)
        try:
            self.assertTrue(replacement.acquire())
            self.assertEqual(replacement.owner()["pid"], os.getpid())
        finally:
            replacement.release()
        self.assertTrue(self.path.exists(), "持久鎖檔不能刪除，以免破壞其他程序的鎖")

    def test_restart_waits_for_old_client_before_starting_runtime(self):
        owner = self.worker()
        self.assertEqual(self.line(owner), "acquired")
        replacement = self.worker("entry", RESTART_WAIT_ARGUMENT)
        self.assertEqual(self.line(replacement), "entry")
        self.assertIsNone(replacement.poll())
        self.assertEqual(ClientInstance(self.path).owner()["pid"], owner.pid)
        owner.stdin.write("\n")
        owner.stdin.flush()
        owner.wait(timeout=5)
        self.assertEqual(self.line(replacement), "runtime:[]")
        self.assertEqual(ClientInstance(self.path).owner()["pid"], replacement.pid)
        replacement.stdin.write("\n")
        replacement.stdin.flush()
        self.assertEqual(replacement.wait(timeout=5), 0)

    def test_duplicate_entrypoint_never_initializes_runtime(self):
        owner = self.worker("entry")
        self.assertEqual(self.line(owner), "entry")
        self.assertEqual(self.line(owner), "runtime:[]")
        duplicate = self.worker("entry")
        output, error = duplicate.communicate(timeout=5)
        self.assertEqual(duplicate.returncode, 0, error)
        self.assertIn(f"PID={owner.pid}", output)
        self.assertNotIn("runtime:", output)

    def test_file_mode_and_settings_can_run_while_microphone_client_is_active(self):
        owner = self.worker()
        self.assertEqual(self.line(owner), "acquired")
        environment = {"CAPSWRITER_CLIENT_SETTINGS": str(self.path.with_name("settings.json"))}
        with patch.dict(os.environ, environment), patch.object(start_client, "_run_client", return_value=0) as run:
            self.assertEqual(start_client.main(["audio.wav"]), 0)
            run.assert_called_once_with(["audio.wav"])
        ui = type("SettingsUI", (), {"run_settings": staticmethod(lambda: True)})
        with patch.dict(sys.modules, {"fork_client.settings_ui": ui, "core.client": None}):
            self.assertEqual(start_client.main(["--settings"]), 0)

    def test_lock_is_released_when_first_run_is_cancelled_or_runtime_raises(self):
        environment = {"CAPSWRITER_CLIENT_SETTINGS": str(self.path.with_name("settings.json"))}
        with patch.dict(os.environ, environment):
            with patch.object(start_client, "_run_client", return_value=0):
                self.assertEqual(start_client.main([]), 0)
            with patch.object(start_client, "_run_client", side_effect=RuntimeError("startup failed")):
                with self.assertRaisesRegex(RuntimeError, "startup failed"):
                    start_client.main([])
            probe = ClientInstance(self.path)
            try:
                self.assertTrue(probe.acquire())
            finally:
                probe.release()

    def test_status_reports_live_pid_and_ignores_stale_pid_after_crash(self):
        owner = self.worker()
        self.assertEqual(self.line(owner), "acquired")
        environment = {"CAPSWRITER_CLIENT_SETTINGS": str(self.path.with_name("settings.json"))}
        output = io.StringIO()
        with patch.dict(os.environ, environment), redirect_stdout(output):
            self.assertEqual(report_status(), 0)
        self.assertIn(f"PID={owner.pid}", output.getvalue())
        owner.kill()
        owner.wait(timeout=5)
        with patch.dict(os.environ, environment), redirect_stdout(io.StringIO()):
            self.assertEqual(start_client.main(["--status"]), 1)


def worker_main():
    mode, path, *arguments = sys.argv[2:]
    if mode == "entry":
        os.environ["CAPSWRITER_CLIENT_SETTINGS"] = str(Path(path).with_name("settings.json"))

        def runtime(args):
            print(f"runtime:{json.dumps(args)}", flush=True)
            sys.stdin.readline()
            return 0

        start_client._run_client = runtime
        print("entry", flush=True)
        return start_client.main(arguments)
    instance = ClientInstance(Path(path))
    try:
        print("acquired" if instance.acquire() else "busy", flush=True)
        sys.stdin.readline()
    finally:
        instance.release()
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        raise SystemExit(worker_main())
    unittest.main()
