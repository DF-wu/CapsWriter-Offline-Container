# coding: utf-8

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import fork_server.env_config as env_config
from config_server import (
    FunASRNanoGGUFArgs,
    ParaformerArgs,
    Qwen3ASRGGUFArgs,
    SenseVoiceArgs,
    ServerConfig,
)
from fork_server.http_api.runtime_config import ConfigError


CLASS_ATTRS = {
    ServerConfig: (
        "model_type",
        "addr",
        "port",
        "log_level",
        "enable_tray",
        "hotwords_path",
        "max_websocket_connections",
        "max_websocket_task_seconds",
        "http_api_enable",
        "http_api_bind",
        "http_api_port",
        "http_api_key",
        "http_api_max_upload_mb",
        "http_api_max_audio_seconds",
        "http_api_task_timeout",
        "http_api_max_concurrent_requests",
        "http_api_max_pending_requests",
        "http_api_cors_origins",
        "http_api_allow_insecure_bind",
        "http_api_log_transcripts",
    ),
    Qwen3ASRGGUFArgs: (
        "model_dir",
        "onnx_provider",
        "llm_use_gpu",
        "chunk_size",
        "n_ctx",
        "memory_num",
        "dml_pad_to",
        "n_batch",
        "n_ubatch",
        "flash_attn",
        "offload_kqv",
    ),
    FunASRNanoGGUFArgs: (
        "encoder_onnx_path",
        "ctc_onnx_path",
        "decoder_gguf_path",
        "tokens_path",
        "onnx_provider",
        "llm_use_gpu",
        "enable_ctc",
        "n_predict",
        "dml_pad_to",
        "max_hotwords",
        "similar_threshold",
        "n_threads",
    ),
    ParaformerArgs: ("paraformer", "tokens"),
    SenseVoiceArgs: ("encoder_path", "decoder_path", "tokenizer_path"),
}


def snapshot_class_attrs() -> dict[type, dict[str, tuple[bool, object]]]:
    snapshot: dict[type, dict[str, tuple[bool, object]]] = {}
    for cls, attrs in CLASS_ATTRS.items():
        snapshot[cls] = {}
        for attr in attrs:
            if hasattr(cls, attr):
                snapshot[cls][attr] = (True, getattr(cls, attr))
            else:
                snapshot[cls][attr] = (False, None)
    return snapshot


def restore_class_attrs(snapshot: dict[type, dict[str, tuple[bool, object]]]) -> None:
    for cls, attrs in snapshot.items():
        for attr, (existed, value) in attrs.items():
            if existed:
                setattr(cls, attr, value)
            elif hasattr(cls, attr):
                delattr(cls, attr)


class EnvConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self._snapshot = snapshot_class_attrs()

    def tearDown(self) -> None:
        restore_class_attrs(self._snapshot)

    def apply_env(self, env: dict[str, str]) -> None:
        with patch.dict(os.environ, env, clear=True):
            env_config.apply()

    def assert_invalid(self, name: str, value: str) -> None:
        with self.subTest(name=name, value=value):
            with self.assertRaisesRegex(ConfigError, name):
                self.apply_env({name: value})

    def test_rejects_invalid_boolean_env_values(self) -> None:
        for name in (
            "CAPSWRITER_ENABLE_TRAY",
            "CAPSWRITER_QWEN_USE_CUDA",
            "CAPSWRITER_QWEN_VULKAN_ENABLE",
            "CAPSWRITER_QWEN_LLAMA_FLASH_ATTN",
            "CAPSWRITER_QWEN_LLAMA_OFFLOAD_KQV",
            "CAPSWRITER_FUNASR_USE_CUDA",
            "CAPSWRITER_FUNASR_VULKAN_ENABLE",
            "CAPSWRITER_FUNASR_ENABLE_CTC",
        ):
            self.assert_invalid(name, "maybe")

    def test_rejects_invalid_choice_env_values(self) -> None:
        for name, value in (
            ("CAPSWRITER_MODEL_TYPE", "qwen"),
            ("CAPSWRITER_LOG_LEVEL", "TRACE"),
            ("CAPSWRITER_QWEN_PRESET", "defaults"),
        ):
            self.assert_invalid(name, value)

    def test_rejects_invalid_integer_env_values(self) -> None:
        for name, value in (
            ("CAPSWRITER_SERVER_PORT", "abc"),
            ("CAPSWRITER_SERVER_PORT", "0"),
            ("CAPSWRITER_SERVER_PORT", "65536"),
            ("CAPSWRITER_SERVER_MAX_WEBSOCKET_CONNECTIONS", "abc"),
            ("CAPSWRITER_SERVER_MAX_WEBSOCKET_CONNECTIONS", "0"),
            ("CAPSWRITER_SERVER_MAX_WEBSOCKET_CONNECTIONS", "1025"),
            ("CAPSWRITER_SERVER_MAX_WEBSOCKET_TASK_SECONDS", "abc"),
            ("CAPSWRITER_SERVER_MAX_WEBSOCKET_TASK_SECONDS", "0"),
            ("CAPSWRITER_SERVER_MAX_WEBSOCKET_TASK_SECONDS", "86401"),
            ("CAPSWRITER_SERVER_MAX_WEBSOCKET_TASK_SECONDS", "nan"),
            ("CAPSWRITER_QWEN_N_CTX", "abc"),
            ("CAPSWRITER_QWEN_N_CTX", "0"),
            ("CAPSWRITER_QWEN_LLAMA_N_BATCH", "0"),
            ("CAPSWRITER_QWEN_LLAMA_N_UBATCH", "abc"),
            ("CAPSWRITER_FUNASR_N_PREDICT", "0"),
            ("CAPSWRITER_NUM_THREADS", "0"),
        ):
            self.assert_invalid(name, value)

    def test_rejects_invalid_float_env_values(self) -> None:
        for name, value in (
            ("CAPSWRITER_QWEN_CHUNK_SIZE", "abc"),
            ("CAPSWRITER_QWEN_CHUNK_SIZE", "0"),
            ("CAPSWRITER_FUNASR_SIMILAR_THRESHOLD", "abc"),
            ("CAPSWRITER_FUNASR_SIMILAR_THRESHOLD", "-0.1"),
            ("CAPSWRITER_FUNASR_SIMILAR_THRESHOLD", "1.1"),
        ):
            self.assert_invalid(name, value)

    def test_blank_env_values_keep_optional_overrides_unset(self) -> None:
        self.apply_env({"CAPSWRITER_QWEN_LLAMA_N_BATCH": " "})

        existed, value = self._snapshot[Qwen3ASRGGUFArgs]["n_batch"]
        if existed:
            self.assertEqual(Qwen3ASRGGUFArgs.n_batch, value)
        else:
            self.assertFalse(hasattr(Qwen3ASRGGUFArgs, "n_batch"))

    def test_applies_valid_server_and_model_env_values(self) -> None:
        self.apply_env(
            {
                "CAPSWRITER_MODEL_TYPE": " Fun_ASR_Nano ",
                "CAPSWRITER_SERVER_PORT": "16016",
                "CAPSWRITER_SERVER_MAX_WEBSOCKET_CONNECTIONS": "13",
                "CAPSWRITER_SERVER_MAX_WEBSOCKET_TASK_SECONDS": "7200.5",
                "CAPSWRITER_LOG_LEVEL": "warning",
                "CAPSWRITER_ENABLE_TRAY": "yes",
                "CAPSWRITER_QWEN_PRESET": "cpu_only",
                "CAPSWRITER_QWEN_CHUNK_SIZE": "42.5",
                "CAPSWRITER_QWEN_N_CTX": "4096",
                "CAPSWRITER_QWEN_MEMORY_NUM": "0",
                "CAPSWRITER_QWEN_PAD_TO": "0",
                "CAPSWRITER_QWEN_LLAMA_N_BATCH": "1024",
                "CAPSWRITER_QWEN_LLAMA_N_UBATCH": "512",
                "CAPSWRITER_QWEN_LLAMA_FLASH_ATTN": "on",
                "CAPSWRITER_QWEN_LLAMA_OFFLOAD_KQV": "off",
                "CAPSWRITER_FUNASR_USE_CUDA": "true",
                "CAPSWRITER_FUNASR_VULKAN_ENABLE": "false",
                "CAPSWRITER_FUNASR_ENABLE_CTC": "no",
                "CAPSWRITER_FUNASR_N_PREDICT": "128",
                "CAPSWRITER_FUNASR_PAD_TO": "0",
                "CAPSWRITER_FUNASR_MAX_HOTWORDS": "0",
                "CAPSWRITER_FUNASR_SIMILAR_THRESHOLD": "1",
                "CAPSWRITER_NUM_THREADS": "2",
            }
        )

        self.assertEqual(ServerConfig.model_type, "fun_asr_nano")
        self.assertEqual(ServerConfig.port, "16016")
        self.assertEqual(ServerConfig.max_websocket_connections, 13)
        self.assertEqual(ServerConfig.max_websocket_task_seconds, 7200.5)
        self.assertEqual(ServerConfig.log_level, "WARNING")
        self.assertTrue(ServerConfig.enable_tray)
        self.assertEqual(Qwen3ASRGGUFArgs.onnx_provider, "CPU")
        self.assertFalse(Qwen3ASRGGUFArgs.llm_use_gpu)
        self.assertEqual(Qwen3ASRGGUFArgs.chunk_size, 42.5)
        self.assertEqual(Qwen3ASRGGUFArgs.n_ctx, 4096)
        self.assertEqual(Qwen3ASRGGUFArgs.memory_num, 0)
        self.assertEqual(Qwen3ASRGGUFArgs.dml_pad_to, 0)
        self.assertEqual(Qwen3ASRGGUFArgs.n_batch, 1024)
        self.assertEqual(Qwen3ASRGGUFArgs.n_ubatch, 512)
        self.assertTrue(Qwen3ASRGGUFArgs.flash_attn)
        self.assertFalse(Qwen3ASRGGUFArgs.offload_kqv)
        self.assertEqual(FunASRNanoGGUFArgs.onnx_provider, "CUDA")
        self.assertFalse(FunASRNanoGGUFArgs.llm_use_gpu)
        self.assertFalse(FunASRNanoGGUFArgs.enable_ctc)
        self.assertEqual(FunASRNanoGGUFArgs.n_predict, 128)
        self.assertEqual(FunASRNanoGGUFArgs.dml_pad_to, 0)
        self.assertEqual(FunASRNanoGGUFArgs.max_hotwords, 0)
        self.assertEqual(FunASRNanoGGUFArgs.similar_threshold, 1.0)
        self.assertEqual(FunASRNanoGGUFArgs.n_threads, 2)


if __name__ == "__main__":
    unittest.main()
