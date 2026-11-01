"""Render the real Compose model without a daemon, GPU, or deployment mutation."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from fork_server.settings import FIELDS, SettingsStore

ROOT = Path(__file__).resolve().parents[2]


def compose_command():
    if shutil.which("docker-compose"):
        return ["docker-compose"]
    if shutil.which("docker"):
        probe = subprocess.run(["docker", "compose", "version"], capture_output=True, timeout=15)
        if probe.returncode == 0:
            return ["docker", "compose"]
    return None


def enable_block(source, name):
    start = f"    # BEGIN {name}\n"
    end = f"    # END {name}\n"
    before, separator, rest = source.partition(start)
    if not separator:
        raise AssertionError(f"Missing hardware block: {name}")
    block, separator, after = rest.partition(end)
    if not separator:
        raise AssertionError(f"Missing end marker: {name}")
    return before + "".join(line[2:] if line.startswith("# ") else line
                            for line in block.splitlines(keepends=True)) + after


@unittest.skipUnless(compose_command(), "Docker Compose CLI is required; no daemon needed")
class ComposeRenderTest(unittest.TestCase):
    def render(self, env=None, block=None, profile=None):
        source = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        for name in block or []:
            source = enable_block(source, name)
        clean = {k: v for k, v in os.environ.items()
                 if not k.startswith(("CAPSWRITER_", "COMPOSE_", "DOCKER_"))}
        clean.update(env or {})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "compose.yml"
            path.write_text(source, encoding="utf-8")
            cmd = compose_command() + ["--project-directory", str(ROOT), "--env-file", os.devnull,
                                       "-f", str(path)]
            if profile:
                cmd += ["--profile", profile]
            result = subprocess.run(cmd + ["config", "--format", "json"],
                                    env=clean, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)

    def test_default_is_one_cpu_safe_nonroot_server(self):
        model = self.render()
        self.assertEqual(set(model["services"]), {"capswriter-server"})
        server = model["services"]["capswriter-server"]
        self.assertNotIn("devices", server)
        self.assertNotIn("deploy", server)
        self.assertNotIn("user", server)
        self.assertNotIn("cap_add", server)
        self.assertEqual(server["cap_drop"], ["ALL"])
        self.assertEqual(server["environment"]["CAPSWRITER_TESLA_P4_OPTIMIZATION"], "false")
        self.assertEqual(server["environment"]["CAPSWRITER_SETTINGS_ENABLE"], "false")
        self.assertTrue(all(p["host_ip"] == "127.0.0.1" for p in server["ports"]))
        mounts = {v["target"]: v for v in server["volumes"]}
        self.assertEqual(mounts["/app/models"]["source"], "capswriter-server-models")
        self.assertEqual(mounts["/app/logs"]["source"], "capswriter-server-logs")
        self.assertEqual(mounts["/app/settings"]["source"], "capswriter-server-settings")
        self.assertTrue(mounts["/app/hot-server.txt"]["read_only"])
        self.assertFalse(mounts["/app/hot-server.txt"]["bind"]["create_host_path"])

    def test_web_profile_adds_web_without_forcing_server_api_or_public_key(self):
        model = self.render(profile="web")
        self.assertEqual(set(model["services"]), {"capswriter-server", "capswriter-web"})
        web = model["services"]["capswriter-web"]
        self.assertEqual(web["environment"]["CAPSWRITER_WEB_ALLOW_PUBLIC_API_KEY"], "false")
        self.assertEqual(web["ports"][0]["host_ip"], "127.0.0.1")
        self.assertEqual(model["services"]["capswriter-server"]["environment"]["CAPSWRITER_HTTP_API_ENABLE"], "false")

    def test_model_bind_and_p4_env_are_selected_in_same_file(self):
        server = self.render({"CAPSWRITER_MODEL_TYPE": "fun_asr_nano",
                              "CAPSWRITER_MODELS_SOURCE": "./models",
                              "CAPSWRITER_TESLA_P4_OPTIMIZATION": "true"})["services"]["capswriter-server"]
        self.assertEqual(server["environment"]["CAPSWRITER_MODEL_TYPE"], "fun_asr_nano")
        self.assertEqual(server["environment"]["CAPSWRITER_TESLA_P4_OPTIMIZATION"], "true")
        mount = next(v for v in server["volumes"] if v["target"] == "/app/models")
        self.assertEqual(mount["type"], "bind")
        self.assertEqual(mount["source"], str(ROOT / "models"))

    def test_blank_daily_values_reach_settings_store_unlocked(self):
        settings = {f.env: "" for f in FIELDS}
        settings.update({"CAPSWRITER_SETTINGS_ENABLE": "true",
                         "CAPSWRITER_HTTP_API_ENABLE": "true",
                         "CAPSWRITER_HTTP_API_KEY": "compose-test-only"})
        env = self.render(settings)["services"]["capswriter-server"]["environment"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server-settings.json"
            path.write_text(json.dumps({"version": 1, "values": {"model_type": "fun_asr_nano"}}))
            env["CAPSWRITER_SETTINGS_PATH"] = str(path)
            store = SettingsStore(env)
            self.assertEqual(store.environment()["CAPSWRITER_MODEL_TYPE"], "fun_asr_nano")
            self.assertTrue(all(store.sources[f.key] != "environment" for f in FIELDS))

    def test_nvidia_block_reserves_gpu_without_root(self):
        server = self.render(block=["NVIDIA"])["services"]["capswriter-server"]
        device = server["deploy"]["resources"]["reservations"]["devices"][0]
        self.assertEqual(device, {"driver": "nvidia", "count": -1, "capabilities": ["gpu"]})
        self.assertNotIn("user", server)

    def test_dri_block_uses_operator_group_ids(self):
        server = self.render({"CAPSWRITER_DRI_RENDER_GID": "123", "CAPSWRITER_DRI_VIDEO_GID": "456"},
                             block=["DRI"])["services"]["capswriter-server"]
        self.assertEqual(server["devices"][0]["source"], "/dev/dri")
        self.assertEqual(server["group_add"], ["123", "456"])

    def test_boost_privileges_require_explicit_block_and_do_not_enable_boost(self):
        server = self.render(block=["NVIDIA", "NVIDIA_BOOST"])["services"]["capswriter-server"]
        self.assertEqual(server["user"], "0:0")
        self.assertEqual(set(server["cap_add"]), {"SYS_ADMIN", "DAC_OVERRIDE", "CHOWN", "FOWNER"})
        self.assertEqual(server["stop_grace_period"], "2m30s")
        self.assertEqual(server["environment"]["CAPSWRITER_GPU_BOOST_ENABLED"], "false")
