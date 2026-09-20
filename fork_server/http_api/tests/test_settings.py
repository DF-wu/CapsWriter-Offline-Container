"""Persistence and startup contracts for restart-only server settings."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

from fork_server.settings import ConfigError, RevisionConflict, SettingsStore
from fork_server.settings_bootstrap import bootstrap_environment


class SettingsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "settings.json"
        self.env = {
            "CAPSWRITER_SETTINGS_ENABLE": "true",
            "CAPSWRITER_SETTINGS_PATH": str(self.path),
            "CAPSWRITER_HTTP_API_KEY": "private-test-key",
        }

    def save(self, values):
        self.path.write_text(json.dumps({"version": 1, "values": values}))

    def fields(self, snapshot):
        return {field["key"]: field for field in snapshot["fields"]}

    def test_save_preserves_unedited_values_and_only_applies_after_restart(self):
        self.save({"model_type": "fun_asr_nano", "format_spell": False})
        store = SettingsStore(self.env)
        result = store.update({"format_num": False}, store.snapshot()["revision"])
        fields = self.fields(result)
        self.assertTrue(result["restart_required"])
        self.assertTrue(fields["format_num"]["value"])
        self.assertFalse(fields["format_num"]["next_value"])
        self.assertEqual(fields["format_num"]["source"], "default")
        self.assertEqual(fields["format_num"]["next_source"], "saved")
        self.assertEqual(json.loads(self.path.read_text())["values"], {
            "model_type": "fun_asr_nano", "format_spell": False, "format_num": False,
        })
        restarted = SettingsStore(self.env)
        self.assertFalse(restarted.effective["format_num"])
        self.assertFalse(restarted.snapshot()["restart_required"])

    def test_environment_has_priority_and_secrets_are_never_in_snapshot(self):
        self.save({"format_num": False})
        env = {**self.env, "CAPSWRITER_FORMAT_NUM": "true", "DATABASE_PASSWORD": "secret"}
        store = SettingsStore(env)
        result = store.update({"format_num": False}, store.snapshot()["revision"])
        field = self.fields(result)["format_num"]
        self.assertTrue(field["value"])
        self.assertTrue(field["next_value"])
        self.assertTrue(field["overridden_by_environment"])
        self.assertEqual(field["source"], "environment")
        self.assertFalse(result["restart_required"])
        serialized = json.dumps(result)
        for secret in ("private-test-key", "DATABASE_PASSWORD", str(self.path)):
            self.assertNotIn(secret, serialized)
        self.assertEqual(env["CAPSWRITER_FORMAT_NUM"], "true")

    def test_null_restores_default_and_tracks_pending_reset(self):
        self.save({"num_threads": 4})
        store = SettingsStore(self.env)
        result = store.update({"num_threads": None}, store.snapshot()["revision"])
        field = self.fields(result)["num_threads"]
        self.assertEqual(field["value"], 4)
        self.assertIsNone(field["next_value"])
        self.assertEqual(field["next_source"], "default")
        self.assertTrue(field["restart_required"])
        self.assertNotIn("num_threads", json.loads(self.path.read_text())["values"])

    def test_revision_rejects_stale_concurrent_writer_without_losing_changes(self):
        stores = [SettingsStore(self.env), SettingsStore(self.env)]
        revision = stores[0].snapshot()["revision"]

        def write(index):
            try:
                stores[index].update({"max_upload_mb": 20 + index}, revision)
                return "saved"
            except RevisionConflict:
                return "conflict"

        with ThreadPoolExecutor(max_workers=2) as executor:
            self.assertCountEqual(executor.map(write, range(2)), ["saved", "conflict"])
        self.assertIn(json.loads(self.path.read_text())["values"]["max_upload_mb"], (20, 21))

    def test_failed_atomic_replace_preserves_previous_file(self):
        self.save({"format_num": False})
        original = self.path.read_bytes()
        store = SettingsStore(self.env)
        with patch("fork_server.settings.os.replace", side_effect=PermissionError):
            with self.assertRaises(PermissionError):
                store.update({"format_num": True}, store.snapshot()["revision"])
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.glob(".capswriter-settings-*")), [])

    def test_rejects_invalid_changes_without_writing(self):
        store = SettingsStore(self.env)
        for values in (
            [], {"http_api_key": "secret"}, {"format_num": "false"},
            {"max_upload_mb": True}, {"max_upload_mb": 1.5},
            {"max_upload_mb": 1025}, {"max_pending_requests": -1},
            {"max_audio_seconds": float("nan")}, {"max_audio_seconds": float("inf")},
            {"max_audio_seconds": 10 ** 400}, {"model_type": "sensevoice"},
        ):
            with self.subTest(values=values), self.assertRaises(ConfigError):
                store.update(values, store.snapshot()["revision"])
        self.assertFalse(self.path.exists())

    def test_invalid_files_fail_closed(self):
        for content in (
            "x", '{"version": true, "values": {}}',
            '{"version": 1, "values": {"max_audio_seconds": NaN}}',
            '{"version": 1, "values": {"unknown": 1}}',
            "[" * 2000 + "]" * 2000, " " * 65537,
        ):
            self.path.write_text(content)
            with self.subTest(content=content[:80]), self.assertRaises(ConfigError):
                SettingsStore(self.env)

    def test_management_requires_path_and_auth_but_disabled_keeps_saved_values(self):
        for missing in ("CAPSWRITER_SETTINGS_PATH", "CAPSWRITER_HTTP_API_KEY"):
            env = {key: value for key, value in self.env.items() if key != missing}
            with self.subTest(missing=missing), self.assertRaises(ConfigError):
                SettingsStore(env)
        self.save({"format_num": False})
        store = SettingsStore({**self.env, "CAPSWRITER_SETTINGS_ENABLE": "false"})
        self.assertFalse(store.effective["format_num"])
        with self.assertRaises(ConfigError):
            store.update({}, store.snapshot()["revision"])

    def test_secret_file_auth_and_blank_env_override(self):
        key_file = self.path.parent / "api-key"
        key_file.write_text("file-key\n")
        self.save({"max_upload_mb": 12})
        env = {**self.env, "CAPSWRITER_HTTP_API_KEY": "",
               "CAPSWRITER_HTTP_API_KEY_FILE": str(key_file),
               "CAPSWRITER_HTTP_API_MAX_UPLOAD_MB": " "}
        self.assertEqual(SettingsStore(env).effective["max_upload_mb"], 12)

    def test_bootstrap_preserves_saved_provenance_and_legacy_gpu_override(self):
        self.save({"model_type": "fun_asr_nano", "format_num": False, "inference_hardware": "gpu"})
        original = {**self.env, "CAPSWRITER_GPU_MODE": "cpu"}
        bootstrapped = bootstrap_environment(original)
        self.assertEqual(bootstrapped["CAPSWRITER_MODEL_TYPE"], "fun_asr_nano")
        self.assertEqual(bootstrapped["CAPSWRITER_INFERENCE_HARDWARE"], "cpu")
        store = SettingsStore(bootstrapped)
        self.assertEqual(store.sources["model_type"], "saved")
        self.assertEqual(store.sources["inference_hardware"], "environment")
        self.assertFalse(store.snapshot()["restart_required"])
        self.assertNotIn("CAPSWRITER_MODEL_TYPE", original)

    def test_injected_marker_cannot_remove_api_auth(self):
        store = SettingsStore({**self.env, "CAPSWRITER_SETTINGS_INJECTED_KEYS": "CAPSWRITER_HTTP_API_KEY"})
        self.assertEqual(store.environment()["CAPSWRITER_HTTP_API_KEY"], "private-test-key")

    def test_native_preserves_config_defaults_and_only_offers_supported_controls(self):
        store = SettingsStore(self.env, {"model_type": "sensevoice", "format_num": False}, profile="native")
        self.assertEqual(store.effective["model_type"], "sensevoice")
        self.assertFalse(store.effective["format_num"])
        self.assertNotIn("qwen_preset", self.fields(store.snapshot()))
        self.assertNotIn("inference_hardware", self.fields(store.snapshot()))
        store.update({"model_type": "paraformer"}, store.snapshot()["revision"])
        with self.assertRaises(ConfigError):
            store.update({"inference_hardware": "cpu"}, store.snapshot()["revision"])

    def test_native_startup_preserves_gpu_and_python_config_defaults(self):
        from config_server import FunASRNanoGGUFArgs, Qwen3ASRGGUFArgs
        from start_server_universal import configure_http_api

        class NativeConfig:
            model_type = "sensevoice"
            format_num = False
            format_spell = False

        with patch.object(Qwen3ASRGGUFArgs, "onnx_provider", "DML"), \
                patch.object(Qwen3ASRGGUFArgs, "llm_use_gpu", True), \
                patch.object(Qwen3ASRGGUFArgs, "n_threads", None, create=True), \
                patch.object(Qwen3ASRGGUFArgs, "n_threads_batch", None, create=True), \
                patch.object(FunASRNanoGGUFArgs, "n_threads", 7):
            configure_http_api({}, NativeConfig)
            self.assertEqual(NativeConfig.model_type, "sensevoice")
            self.assertFalse(NativeConfig.format_num)
            self.assertEqual(Qwen3ASRGGUFArgs.onnx_provider, "DML")
            self.assertTrue(Qwen3ASRGGUFArgs.llm_use_gpu)
            self.assertEqual(FunASRNanoGGUFArgs.n_threads, 7)
            self.assertIsNone(Qwen3ASRGGUFArgs.n_threads)
            self.assertIsNone(Qwen3ASRGGUFArgs.n_threads_batch)
            self.save({"model_type": "paraformer", "format_num": True, "num_threads": 2})
            configure_http_api(self.env, NativeConfig)
            self.assertEqual(NativeConfig.model_type, "paraformer")
            self.assertTrue(NativeConfig.format_num)
            self.assertFalse(NativeConfig.format_spell)
            self.assertEqual(FunASRNanoGGUFArgs.n_threads, 2)
            self.assertEqual(Qwen3ASRGGUFArgs.n_threads, 2)
            self.assertEqual(Qwen3ASRGGUFArgs.n_threads_batch, 2)
            self.assertEqual(Qwen3ASRGGUFArgs.onnx_provider, "DML")

    @unittest.skipIf(os.name == "nt", "Docker shell entrypoint is POSIX only")
    def test_entrypoint_bootstraps_saved_settings_and_preserves_command_arguments(self):
        self.save({"model_type": "fun_asr_nano", "inference_hardware": "cpu"})
        root = Path(__file__).resolve().parents[3]
        env = {**os.environ, **self.env}
        for name in ("CAPSWRITER_SETTINGS_BOOTSTRAPPED", "CAPSWRITER_SETTINGS_INJECTED_KEYS",
                     "CAPSWRITER_MODEL_TYPE", "CAPSWRITER_INFERENCE_HARDWARE", "CAPSWRITER_GPU_MODE"):
            env.pop(name, None)
        result = subprocess.run([
            "sh", str(root / "docker/server/entrypoint.sh"), sys.executable, "-c",
            'import json,os,sys; print(json.dumps([os.environ["CAPSWRITER_MODEL_TYPE"], '
            'os.environ["CAPSWRITER_QWEN_USE_CUDA"],sys.argv[1]]))', "argument with spaces",
        ], env=env, cwd=root, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout.splitlines()[-1]),
                         ["fun_asr_nano", "false", "argument with spaces"])


if __name__ == "__main__":
    unittest.main()
