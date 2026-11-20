# coding: utf-8

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import user_config


ROOT = Path(__file__).resolve().parents[2]

CONFIG_CLIENT = """\
from pathlib import Path

class ClientConfig:
    addr = '127.0.0.1'
    port = '6016'
    paste_apps = ['WeiXin.exe']
    enter_apps = [('happ.exe', 0.5)]
    save_audio = True
    audio_dir = Path('audio')
"""
CONFIG_SERVER = """\
from pathlib import Path

class ServerConfig:
    model_type = 'qwen_asr'
    hotwords_path = Path() / 'hot-server.txt'

class ModelPaths:
    fun_asr_nano_gguf_hotwords = Path() / 'hot-server.txt'
"""


def load(path: Path, name: str = "_test_config"):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def install(*, server: bool = True):
    """A fake program folder with its own user/ directory."""

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "app"
        (root / "LLM").mkdir(parents=True)
        (root / "config_client.py").write_text(CONFIG_CLIENT, encoding="utf-8")
        if server:
            (root / "config_server.py").write_text(CONFIG_SERVER, encoding="utf-8")
            (root / "hot-server.txt").write_text("server default\n", encoding="utf-8")
        (root / "hot.txt").write_text("default hot\n", encoding="utf-8")
        (root / "hot-rule.txt").write_text("default rule\n", encoding="utf-8")
        (root / "LLM" / "__init__.py").write_text("", encoding="utf-8")
        (root / "LLM" / "default.py").write_text("name = ''\n", encoding="utf-8")
        (root / "LLM" / "翻译.py").write_text("name = '翻译'\n", encoding="utf-8")
        with patch.object(user_config, "program_root", return_value=root), \
                patch.dict(os.environ, {user_config.USER_DIR_ENV: ""}):
            yield root


class UserDirTest(unittest.TestCase):
    def test_user_dir_defaults_to_program_folder_and_honors_override(self) -> None:
        with install() as root:
            self.assertEqual(user_config.user_dir(), root / "user")
            with patch.dict(os.environ, {user_config.USER_DIR_ENV: str(root / "elsewhere")}):
                self.assertEqual(user_config.user_dir(), (root / "elsewhere").resolve())

    def test_data_file_prefers_user_copy(self) -> None:
        with install() as root:
            self.assertEqual(user_config.data_file("hot.txt"), root / "hot.txt")
            (root / "user").mkdir()
            (root / "user" / "hot.txt").write_text("mine\n", encoding="utf-8")
            self.assertEqual(user_config.data_file("hot.txt"), root / "user" / "hot.txt")

    def test_roles_merge_with_user_copies_first(self) -> None:
        with install() as root:
            self.assertEqual(user_config.role_dirs(), [root / "LLM"])
            user_roles = root / "user" / "LLM"
            user_roles.mkdir(parents=True)
            (user_roles / "default.py").write_text("name = 'mine'\n", encoding="utf-8")
            (user_roles / "custom.py").write_text("name = 'custom'\n", encoding="utf-8")

            self.assertEqual(user_config.role_dirs(), [user_roles, root / "LLM"])
            self.assertEqual(
                user_config.role_files(),
                [user_roles / "custom.py", user_roles / "default.py", root / "LLM" / "翻译.py"],
            )
            package_path = [str(root / "LLM")]
            user_config.extend_role_package(package_path)
            user_config.extend_role_package(package_path)
            self.assertEqual(package_path, [str(user_roles), str(root / "LLM")])


class OverlayTest(unittest.TestCase):
    def test_overlay_applies_known_attributes_and_warns_about_unknown_names(self) -> None:
        with install() as root:
            (root / "user").mkdir()
            (root / "user" / "config_client.py").write_text(
                "class ClientConfig:\n"
                "    addr = '192.0.2.10'\n"
                "    adr = 'typo'\n"
                "class Missing:\n"
                "    value = 1\n",
                encoding="utf-8",
            )
            namespace = vars(load(root / "config_client.py"))
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                applied = user_config.apply_overlay(namespace, "config_client.py")

            self.assertEqual(applied, ("ClientConfig.addr",))
            self.assertEqual(user_config.applied_overrides("config_client.py"), applied)
            self.assertEqual(namespace["ClientConfig"].addr, "192.0.2.10")
            self.assertFalse(hasattr(namespace["ClientConfig"], "adr"))
            self.assertIn("ClientConfig.adr", stderr.getvalue())
            self.assertIn("Missing", stderr.getvalue())

    def test_overlay_without_user_file_is_a_no_op(self) -> None:
        with install() as root:
            namespace = vars(load(root / "config_client.py"))
            self.assertEqual(user_config.apply_overlay(namespace, "config_client.py"), ())
            self.assertEqual(namespace["ClientConfig"].addr, "127.0.0.1")

    def test_broken_user_file_fails_loudly(self) -> None:
        with install() as root:
            (root / "user").mkdir()
            (root / "user" / "config_client.py").write_text("class ClientConfig(:\n", encoding="utf-8")
            namespace = vars(load(root / "config_client.py"))
            with self.assertRaisesRegex(user_config.UserConfigError, "SyntaxError"):
                user_config.apply_overlay(namespace, "config_client.py")

    def test_server_data_paths_follow_user_hotwords_unless_overridden(self) -> None:
        data_paths = {
            "ServerConfig.hotwords_path": "hot-server.txt",
            "ModelPaths.fun_asr_nano_gguf_hotwords": "hot-server.txt",
        }
        with install() as root:
            namespace = vars(load(root / "config_server.py"))
            user_config.apply_overlay(namespace, "config_server.py", data_paths)
            self.assertEqual(namespace["ServerConfig"].hotwords_path, Path("hot-server.txt"))

            (root / "user").mkdir()
            user_hot = root / "user" / "hot-server.txt"
            user_hot.write_text("mine\n", encoding="utf-8")
            namespace = vars(load(root / "config_server.py"))
            user_config.apply_overlay(namespace, "config_server.py", data_paths)
            self.assertEqual(namespace["ServerConfig"].hotwords_path, user_hot)
            self.assertEqual(namespace["ModelPaths"].fun_asr_nano_gguf_hotwords, user_hot)

            (root / "user" / "config_server.py").write_text(
                "from pathlib import Path\n"
                "class ServerConfig:\n"
                "    hotwords_path = Path('elsewhere.txt')\n",
                encoding="utf-8",
            )
            namespace = vars(load(root / "config_server.py"))
            user_config.apply_overlay(namespace, "config_server.py", data_paths)
            self.assertEqual(namespace["ServerConfig"].hotwords_path, Path("elsewhere.txt"))

    def test_real_config_modules_apply_user_overlay_on_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            user = Path(tmp)
            (user / "config_client.py").write_text(
                "class ClientConfig:\n    addr = '192.0.2.20'\n", encoding="utf-8"
            )
            (user / "config_server.py").write_text(
                "class ServerConfig:\n    enable_tray = False\n", encoding="utf-8"
            )
            (user / "hot-server.txt").write_text("mine\n", encoding="utf-8")
            with patch.dict(os.environ, {user_config.USER_DIR_ENV: tmp}):
                client = load(ROOT / "config_client.py", "_test_real_config_client")
                server = load(ROOT / "config_server.py", "_test_real_config_server")

            self.assertEqual(client.ClientConfig.addr, "192.0.2.20")
            self.assertIs(server.ServerConfig.enable_tray, False)
            self.assertEqual(server.ServerConfig.hotwords_path, (user / "hot-server.txt").resolve())
            self.assertEqual(
                server.ModelPaths.fun_asr_nano_gguf_hotwords,
                (user / "hot-server.txt").resolve(),
            )


class SeedTest(unittest.TestCase):
    def test_client_seed_creates_templates_and_copies_without_overwriting(self) -> None:
        with install() as root:
            created = user_config.seed()
            user = root / "user"

            self.assertIn(user / "README.txt", created)
            self.assertEqual((user / "hot.txt").read_text(encoding="utf-8"), "default hot\n")
            self.assertEqual((user / "hot-rule.txt").read_text(encoding="utf-8"), "default rule\n")
            self.assertTrue((user / "LLM" / "default.py").is_file())
            self.assertTrue((user / "LLM" / "翻译.py").is_file())
            self.assertFalse((user / "LLM" / "__init__.py").exists())
            self.assertFalse((user / "config_server.py").exists())

            # The template is valid Python that overrides nothing.
            namespace = vars(load(root / "config_client.py"))
            self.assertEqual(user_config.apply_overlay(namespace, "config_client.py"), ())
            self.assertIn("class ClientConfig:", (user / "config_client.py").read_text(encoding="utf-8"))

            (user / "hot.txt").write_text("mine\n", encoding="utf-8")
            (user / "LLM" / "default.py").unlink()
            self.assertEqual(user_config.seed(), [])
            self.assertEqual((user / "hot.txt").read_text(encoding="utf-8"), "mine\n")
            # Roles are seeded only into a new user/LLM, never re-added.
            self.assertFalse((user / "LLM" / "default.py").exists())

    def test_server_seed_adds_server_template_and_hotwords(self) -> None:
        with install() as root:
            user_config.seed(server=True)
            user = root / "user"
            self.assertIn("class ServerConfig:", (user / "config_server.py").read_text(encoding="utf-8"))
            self.assertEqual((user / "hot-server.txt").read_text(encoding="utf-8"), "server default\n")
            self.assertFalse((user / "LLM").exists())

    def test_client_only_install_gets_no_server_template(self) -> None:
        with install(server=False) as root:
            user_config.seed(server=True)
            self.assertFalse((root / "user" / "config_server.py").exists())
            self.assertFalse((root / "user" / "hot-server.txt").exists())

    def test_unwritable_user_dir_only_warns(self) -> None:
        with install(), patch.object(Path, "mkdir", side_effect=PermissionError("denied")):
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(user_config.seed(), [])
            self.assertIn("denied", stderr.getvalue())


class ImportLegacyTest(unittest.TestCase):
    def make_old_install(self, parent: Path) -> Path:
        old = parent / "old"
        (old / "LLM").mkdir(parents=True)
        (old / "config_client.py").write_text(
            CONFIG_CLIENT.replace("'127.0.0.1'", "'192.168.1.20'")
            .replace("['WeiXin.exe']", "['WeiXin.exe', 'Code.exe']")
            .replace("Path('audio')", "Path('D:/recordings')")
            + "    removed_setting = 1\n",
            encoding="utf-8",
        )
        (old / "config_server.py").write_text(
            CONFIG_SERVER.replace("'qwen_asr'", "'sensevoice'"), encoding="utf-8"
        )
        (old / "hot.txt").write_text("old custom hot\n", encoding="utf-8")
        (old / "hot-rule.txt").write_text("default rule\n", encoding="utf-8")
        (old / "LLM" / "default.py").write_text("name = ''\n", encoding="utf-8")
        (old / "LLM" / "大助理.py").write_text("api_key = 'secret'\n", encoding="utf-8")
        return old

    def test_import_writes_only_changed_settings_and_customized_files(self) -> None:
        with install() as root:
            old = self.make_old_install(root.parent)
            user_config.seed()  # A first run already seeded untouched defaults.

            report = user_config.import_legacy(old)
            user = root / "user"

            namespace = vars(load(root / "config_client.py"))
            applied = user_config.apply_overlay(namespace, "config_client.py")
            self.assertEqual(
                set(applied),
                {"ClientConfig.addr", "ClientConfig.paste_apps", "ClientConfig.audio_dir"},
            )
            self.assertEqual(namespace["ClientConfig"].addr, "192.168.1.20")
            self.assertEqual(namespace["ClientConfig"].audio_dir, Path("D:/recordings"))
            self.assertNotIn("removed_setting", (user / "config_client.py").read_text(encoding="utf-8"))
            self.assertIn("model_type = 'sensevoice'", (user / "config_server.py").read_text(encoding="utf-8"))

            self.assertEqual((user / "hot.txt").read_text(encoding="utf-8"), "old custom hot\n")
            self.assertEqual((user / "hot-rule.txt").read_text(encoding="utf-8"), "default rule\n")
            self.assertEqual((user / "LLM" / "大助理.py").read_text(encoding="utf-8"), "api_key = 'secret'\n")
            self.assertTrue(any("大助理.py" in line for line in report))
            self.assertFalse(any("default.py" in line for line in report))

    def test_import_keeps_existing_user_customizations(self) -> None:
        with install() as root:
            old = self.make_old_install(root.parent)
            user = root / "user"
            user.mkdir()
            (user / "hot.txt").write_text("already mine\n", encoding="utf-8")
            (user / "config_client.py").write_text("class ClientConfig:\n    port = '7000'\n", encoding="utf-8")

            report = user_config.import_legacy(old)

            self.assertEqual((user / "hot.txt").read_text(encoding="utf-8"), "already mine\n")
            self.assertEqual(
                (user / "config_client.py").read_text(encoding="utf-8"),
                "class ClientConfig:\n    port = '7000'\n",
            )
            self.assertTrue(any(line.startswith("保留") and "hot.txt" in line for line in report))
            self.assertTrue(any(line.startswith("保留") and "config_client.py" in line for line in report))

    def test_client_only_install_skips_server_settings(self) -> None:
        with install(server=False) as root:
            old = self.make_old_install(root.parent)
            (old / "hot-server.txt").write_text("old server hot\n", encoding="utf-8")

            user_config.import_legacy(old)

            self.assertTrue((root / "user" / "config_client.py").is_file())
            self.assertFalse((root / "user" / "config_server.py").exists())
            self.assertFalse((root / "user" / "hot-server.txt").exists())

    def test_import_rejects_current_or_unrelated_folder(self) -> None:
        with install() as root:
            with self.assertRaisesRegex(user_config.UserConfigError, "舊版"):
                user_config.import_legacy(root)
            with self.assertRaisesRegex(user_config.UserConfigError, "config_client.py"):
                user_config.import_legacy(root.parent)

    def test_unrepresentable_values_are_reported_not_written(self) -> None:
        self.assertIsNone(user_config._render_value(object()))
        self.assertIsNone(user_config._render_value([Path("a")]))
        self.assertEqual(user_config._render_value(Path("models/x")), "Path('models/x')")
        self.assertEqual(user_config._render_value([("a.exe", 0.5)]), "[('a.exe', 0.5)]")


class ClientEntrypointTest(unittest.TestCase):
    def test_import_config_argument_reports_and_returns_status(self) -> None:
        import start_client

        stdout = io.StringIO()
        with patch("user_config.import_legacy", return_value=["匯入 x"]) as run, \
                patch("sys.stdout", stdout):
            self.assertEqual(start_client.main(["--import-config", "D:/old"]), 0)
        run.assert_called_once_with("D:/old")
        self.assertIn("匯入 x", stdout.getvalue())

        stderr = io.StringIO()
        with patch(
            "user_config.import_legacy",
            side_effect=user_config.UserConfigError("bad"),
        ), patch("sys.stderr", stderr):
            self.assertEqual(start_client.main(["--import-config", "D:/old"]), 2)
        self.assertIn("bad", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
