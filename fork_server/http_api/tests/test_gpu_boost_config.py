"""GPU clock controls must reach the worker as validated NVIDIA arguments."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fork_server.gpu_boost_config import apply_gpu_boost_config, parse_gpu_boost_config
from fork_server.http_api.runtime_config import ConfigError


class GpuBoostConfigTest(unittest.TestCase):
    def test_native_tensorrt_remains_eligible_for_nvidia_boost(self):
        from config_server import SenseVoiceArgs
        from fork_server.gpu_boost_config import configure_gpu_boost_backend

        for provider in ("TRT", "TENSORRT"):
            with self.subTest(provider=provider), patch.object(SenseVoiceArgs, "onnx_provider", provider):
                target = SimpleNamespace(model_type="sensevoice")
                configure_gpu_boost_backend(target, {})
                self.assertTrue(target.gpu_boost_backend_active)

    def test_default_does_not_enable_clock_changes(self):
        config = parse_gpu_boost_config({})
        self.assertFalse(config.enabled)
        self.assertEqual(config.memory_clock_mhz, 0)

    def test_explicit_gpu_and_frequency_become_literal_arguments(self):
        target = SimpleNamespace()
        environment = {
            "CAPSWRITER_GPU_BOOST_ENABLED": "true",
            "CAPSWRITER_GPU_BOOST_DEVICE": "1",
            "CAPSWRITER_GPU_BOOST_MEMORY_CLOCK_MHZ": "6000",
            "CAPSWRITER_GPU_BOOST_IDLE_SECONDS": "2.5",
        }
        before = dict(environment)
        apply_gpu_boost_config(target, environment)
        self.assertTrue(target.gpu_boost_enabled)
        self.assertEqual(target.gpu_boost_cmd, ["nvidia-smi", "-i", "1", "-lmc", "6000"])
        self.assertEqual(target.gpu_unboost_cmd, ["nvidia-smi", "-i", "1", "-rmc"])
        self.assertEqual(target.gpu_unboost_timeout, 2.5)
        self.assertEqual(environment, before)

    def test_enabling_requires_an_explicit_frequency(self):
        for value in (None, "", "0"):
            environment = {"CAPSWRITER_GPU_BOOST_ENABLED": "true"}
            if value is not None:
                environment["CAPSWRITER_GPU_BOOST_MEMORY_CLOCK_MHZ"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ConfigError, "MEMORY_CLOCK_MHZ"):
                parse_gpu_boost_config(environment)

    def test_invalid_controls_are_rejected_before_command_construction(self):
        for name, value in (
            ("CAPSWRITER_GPU_BOOST_ENABLED", "maybe"),
            ("CAPSWRITER_GPU_BOOST_DEVICE", "0; touch /tmp/should-not-exist"),
            ("CAPSWRITER_GPU_BOOST_DEVICE", "-1"),
            ("CAPSWRITER_GPU_BOOST_DEVICE", "1024"),
            ("CAPSWRITER_GPU_BOOST_MEMORY_CLOCK_MHZ", "6000 && true"),
            ("CAPSWRITER_GPU_BOOST_MEMORY_CLOCK_MHZ", "-1"),
            ("CAPSWRITER_GPU_BOOST_IDLE_SECONDS", "nan"),
            ("CAPSWRITER_GPU_BOOST_IDLE_SECONDS", "0"),
            ("CAPSWRITER_GPU_BOOST_TIMEOUT", "inf"),
            ("CAPSWRITER_GPU_BOOST_TIMEOUT", "0"),
        ):
            with self.subTest(name=name, value=value), self.assertRaisesRegex(ConfigError, name):
                parse_gpu_boost_config({name: value})

    def test_configuration_does_not_execute_hardware_commands(self):
        with patch("subprocess.Popen") as popen, patch("subprocess.run") as run:
            apply_gpu_boost_config(SimpleNamespace(), {
                "CAPSWRITER_GPU_BOOST_ENABLED": "true",
                "CAPSWRITER_GPU_BOOST_MEMORY_CLOCK_MHZ": "6000",
            })
        popen.assert_not_called()
        run.assert_not_called()

    def test_native_entrypoint_accepts_explicit_boost_environment(self):
        from start_server_universal import configure_http_api

        target = type("NativeConfig", (), {
            "model_type": "qwen_asr", "format_num": True, "format_spell": True,
            "gpu_boost_enabled": False,
        })
        configure_http_api({
            "CAPSWRITER_GPU_BOOST_ENABLED": "true",
            "CAPSWRITER_GPU_BOOST_MEMORY_CLOCK_MHZ": "6000",
        }, target)
        self.assertTrue(target.gpu_boost_enabled)
        self.assertEqual(target.gpu_boost_cmd, ["nvidia-smi", "-i", "0", "-lmc", "6000"])

    def test_native_entrypoint_preserves_python_clock_commands_without_override(self):
        from start_server_universal import configure_http_api

        target = type("NativeConfig", (), {
            "model_type": "qwen_asr", "format_num": True, "format_spell": True,
            "gpu_boost_enabled": True, "gpu_boost_cmd": "custom-boost",
        })
        configure_http_api({"CAPSWRITER_GPU_BOOST_TIMEOUT": "3"}, target)
        self.assertTrue(target.gpu_boost_enabled)
        self.assertEqual(target.gpu_boost_cmd, "custom-boost")


if __name__ == "__main__":
    unittest.main()
