# coding: utf-8

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
TRAY_PATH = ROOT / "core" / "ui" / "tray.py"
TRAY_MANAGER_PATH = ROOT / "core" / "client" / "manager" / "tray_manager.py"


class FakeSubprocess:
    DEVNULL = object()
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    DETACHED_PROCESS = 0x00000008

    def __init__(self) -> None:
        self.calls = []

    def Popen(self, cmd, **kwargs):
        call = {"cmd": list(cmd), "kwargs": dict(kwargs)}
        self.calls.append(call)
        return SimpleNamespace(**call)


class FakeOS:
    def __init__(self, name: str) -> None:
        self.name = name
        self.startfile_calls = []

    def startfile(self, target: str) -> None:
        self.startfile_calls.append(target)


def load_functions(
    path: Path,
    names: set[str],
    *,
    os_name: str,
    platform: str = "linux",
):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    fake_os = FakeOS(os_name)
    fake_subprocess = FakeSubprocess()
    namespace = {
        "os": fake_os,
        "subprocess": fake_subprocess,
        "sys": SimpleNamespace(platform=platform),
    }
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace, fake_os, fake_subprocess


def find_method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"{class_name}.{method_name} was not found")


def is_name_call(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    )


def is_attr_call(node: ast.AST, value_name: str, attr_name: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr_name
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == value_name
    )


class TrayProcessLaunchTest(unittest.TestCase):
    def test_detached_process_helpers_suppress_stdio_and_detach_session(self) -> None:
        for path in (TRAY_PATH, TRAY_MANAGER_PATH):
            for os_name in ("posix", "nt"):
                with self.subTest(path=path, os_name=os_name):
                    namespace, _fake_os, fake_subprocess = load_functions(
                        path,
                        {"_detached_popen_kwargs", "_launch_detached_process"},
                        os_name=os_name,
                    )

                    proc = namespace["_launch_detached_process"](["python", "app.py"])

                    self.assertEqual(proc.cmd, ["python", "app.py"])
                    self.assertEqual(len(fake_subprocess.calls), 1)
                    kwargs = fake_subprocess.calls[0]["kwargs"]
                    self.assertIs(kwargs["stdin"], fake_subprocess.DEVNULL)
                    self.assertIs(kwargs["stdout"], fake_subprocess.DEVNULL)
                    self.assertIs(kwargs["stderr"], fake_subprocess.DEVNULL)
                    self.assertTrue(kwargs["close_fds"])
                    if os_name == "nt":
                        self.assertNotIn("start_new_session", kwargs)
                        self.assertEqual(
                            kwargs["creationflags"],
                            (
                                fake_subprocess.CREATE_NEW_PROCESS_GROUP
                                | fake_subprocess.DETACHED_PROCESS
                            ),
                        )
                    else:
                        self.assertTrue(kwargs["start_new_session"])
                        self.assertNotIn("creationflags", kwargs)

    def test_default_opener_uses_detached_launcher_on_posix_platforms(self) -> None:
        for platform, expected_cmd in (
            ("darwin", ["open", "/tmp/hot.txt"]),
            ("linux", ["xdg-open", "/tmp/hot.txt"]),
        ):
            with self.subTest(platform=platform):
                namespace, _fake_os, fake_subprocess = load_functions(
                    TRAY_MANAGER_PATH,
                    {
                        "_detached_popen_kwargs",
                        "_launch_detached_process",
                        "_open_with_default_app",
                    },
                    os_name="posix",
                    platform=platform,
                )

                namespace["_open_with_default_app"]("/tmp/hot.txt")

                self.assertEqual(fake_subprocess.calls[0]["cmd"], expected_cmd)
                self.assertTrue(
                    fake_subprocess.calls[0]["kwargs"]["start_new_session"]
                )

    def test_default_opener_uses_startfile_on_windows(self) -> None:
        namespace, fake_os, fake_subprocess = load_functions(
            TRAY_MANAGER_PATH,
            {
                "_detached_popen_kwargs",
                "_launch_detached_process",
                "_open_with_default_app",
            },
            os_name="nt",
            platform="win32",
        )

        namespace["_open_with_default_app"]("C:/CapsWriter/hot.txt")

        self.assertEqual(fake_os.startfile_calls, ["C:/CapsWriter/hot.txt"])
        self.assertEqual(fake_subprocess.calls, [])

    def test_tray_callbacks_use_launch_helpers_source_guard(self) -> None:
        restart = find_method(TRAY_PATH, "_TraySystem", "on_restart")
        restart_calls = list(ast.walk(restart))
        self.assertTrue(
            any(is_name_call(node, "_launch_detached_process") for node in restart_calls)
        )
        self.assertFalse(
            any(is_attr_call(node, "subprocess", "Popen") for node in restart_calls)
        )

        hotword = find_method(TRAY_MANAGER_PATH, "TrayManager", "_add_hotword")
        hotword_calls = list(ast.walk(hotword))
        self.assertTrue(
            any(is_name_call(node, "_open_with_default_app") for node in hotword_calls)
        )
        self.assertFalse(
            any(is_attr_call(node, "subprocess", "Popen") for node in hotword_calls)
        )
        self.assertFalse(
            any(is_attr_call(node, "os", "startfile") for node in hotword_calls)
        )


if __name__ == "__main__":
    unittest.main()
