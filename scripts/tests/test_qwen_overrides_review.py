"""Exercise env -> factory -> Qwen config -> native context without model assets."""

from __future__ import annotations

import ast
import ctypes
import dataclasses
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import config_server
from fork_server import env_config
from fork_server.http_api.runtime_config import ConfigError


ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = ROOT / "core/server/engines"


def load_classes(path, names, **dependencies):
    """Keep production classes intact; omit imports and native-library init()."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    tree.body = [node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name in names]
    namespace = {"__name__": __name__, **dependencies}
    exec(compile(tree, str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


class QwenOverridesReviewTest(unittest.TestCase):
    def setUp(self):
        # apply() mutates config classes; restore added attrs as well as defaults.
        self.snapshots = {cls: dict(vars(cls)) for cls in vars(config_server).values()
                          if isinstance(cls, type)}
        self.addCleanup(self.restore_config)
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def restore_config(self):
        for cls, before in self.snapshots.items():
            for name in set(vars(cls)) - set(before):
                delattr(cls, name)
            for name, value in before.items():
                if not name.startswith("__"):
                    setattr(cls, name, value)

    def create_engine(self, environment, binding_path="llama/llama.py", *, native_startup=False):
        os.environ.update(environment)
        if native_startup:
            from start_server_universal import configure_http_api
            configure_http_api(os.environ, config_server.ServerConfig)
        else:
            env_config.apply()
        schema = load_classes(
            ENGINE_ROOT / "qwen_asr_gguf/inference/schema.py",
            {"AlignerConfig", "ASREngineConfig"}, dataclass=dataclasses.dataclass,
        )
        captured = []

        def init_context(model, params):
            captured.append(params)
            return 1

        native = load_classes(
            ENGINE_ROOT / binding_path,
            {"llama_context_params", "LlamaContext"},
            ctypes=ctypes, os=os, llama_init_from_model=init_context,
            llama_free=lambda ptr: None,
        )
        native.LlamaContext.__init__.__globals__["llama_context_default_params"] = native.llama_context_params
        model = SimpleNamespace(ptr=1, token_to_id=lambda text: 0)
        llama = SimpleNamespace(
            LlamaContext=native.LlamaContext,
            LlamaModel=Mock(return_value=model),
            get_token_embeddings_gguf=Mock(return_value=None),
        )
        engine_module = load_classes(
            ENGINE_ROOT / "qwen_asr_gguf/inference/asr.py", {"QwenASREngine"},
            os=os, llama=llama, QwenAudioEncoder=Mock(),
        )
        factory = load_classes(ENGINE_ROOT / "factory.py", {"EngineFactory"}).EngineFactory
        # Preserve the real factory's public-attribute collection and dataclass
        # construction; replace only lazy importing the asset-loading adapter.
        factory._ASR_LOADERS["qwen_asr"] = lambda: (
            engine_module.QwenASREngine, schema.ASREngineConfig,
            config_server.Qwen3ASRGGUFArgs,
        )
        with patch.object(os, "cpu_count", return_value=8):
            engine = factory.create_asr_engine("qwen_asr")
        self.assertEqual(len(captured), 1)
        return engine, captured[0]

    def test_advertised_overrides_reach_native_context(self):
        engine, params = self.create_engine({
            "CAPSWRITER_QWEN_LLAMA_N_BATCH": "1024",
            "CAPSWRITER_QWEN_LLAMA_N_UBATCH": "128",
            "CAPSWRITER_QWEN_LLAMA_FLASH_ATTN": "false",
            "CAPSWRITER_QWEN_LLAMA_OFFLOAD_KQV": "false",
            "CAPSWRITER_QWEN_N_CTX": "8192",
            "CAPSWRITER_NUM_THREADS": "3",
        })
        self.assertEqual(params.n_ctx, 8192)
        self.assertEqual(params.n_batch, 1024)
        self.assertEqual(params.n_ubatch, 128)
        self.assertEqual(params.flash_attn_type, 0)
        self.assertFalse(params.offload_kqv)
        self.assertEqual((params.n_threads, params.n_threads_batch), (3, 3))
        self.assertEqual(config_server.FunASRNanoGGUFArgs.n_threads, 3)
        self.assertEqual(engine.config.n_batch, 1024)

    def test_defaults_preserve_existing_context_behavior(self):
        _, params = self.create_engine({})
        self.assertEqual((params.n_ctx, params.n_batch, params.n_ubatch), (2048, 4096, 512))
        self.assertEqual(params.flash_attn_type, 1)
        self.assertTrue(params.offload_kqv)
        self.assertEqual((params.n_threads, params.n_threads_batch), (4, 8))

    def test_shared_thread_limit_reaches_qwen_without_other_overrides(self):
        _, params = self.create_engine({"CAPSWRITER_NUM_THREADS": "2"})
        self.assertEqual((params.n_threads, params.n_threads_batch), (2, 2))

    def test_native_startup_applies_shared_thread_limit_to_qwen(self):
        _, params = self.create_engine({"CAPSWRITER_NUM_THREADS": "3"}, native_startup=True)
        self.assertEqual((params.n_threads, params.n_threads_batch), (3, 3))
        self.assertEqual(config_server.FunASRNanoGGUFArgs.n_threads, 3)

    def test_native_startup_preserves_automatic_thread_defaults(self):
        _, params = self.create_engine({}, native_startup=True)
        self.assertEqual((params.n_threads, params.n_threads_batch), (4, 8))

    def test_boolean_true_uses_b7798_flash_attention_enum(self):
        # Both possible Qwen bindings must preserve the b7798 ctypes enum ABI.
        for binding in ("llama/llama.py", "qwen_asr_gguf/inference/llama.py"):
            with self.subTest(binding=binding):
                _, params = self.create_engine({
                    "CAPSWRITER_QWEN_LLAMA_FLASH_ATTN": "true",
                    "CAPSWRITER_QWEN_LLAMA_OFFLOAD_KQV": "true",
                }, binding)
                self.assertEqual(params.flash_attn_type, 1)
                self.assertTrue(params.offload_kqv)

    def test_rejects_unsupported_values_before_engine_creation(self):
        for name, value in (
            ("CAPSWRITER_QWEN_LLAMA_N_BATCH", "0"),
            ("CAPSWRITER_QWEN_LLAMA_N_UBATCH", "-1"),
            ("CAPSWRITER_QWEN_LLAMA_N_UBATCH", "1.5"),
            ("CAPSWRITER_QWEN_LLAMA_FLASH_ATTN", "auto"),
            ("CAPSWRITER_QWEN_LLAMA_OFFLOAD_KQV", "maybe"),
            ("CAPSWRITER_NUM_THREADS", "0"),
        ):
            with self.subTest(name=name, value=value):
                with patch.dict(os.environ, {name: value}, clear=True):
                    with self.assertRaisesRegex(ConfigError, name):
                        env_config.apply()


if __name__ == "__main__":
    unittest.main()
