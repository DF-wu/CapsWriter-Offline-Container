# coding: utf-8
"""以外部 ORT session 替身验证加载后资格，不接触 GPU 或模型文件。"""

import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fork_server.gpu_boost_config import configure_gpu_boost_backend
from scripts.tests.test_gpu_boost import load_gpu_boost_namespace


ROOT = Path(__file__).resolve().parents[2]


def load_class(relative_path, name, namespace):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == name)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def ort_session(provider):
    session = Mock()
    session.get_providers.return_value = [provider, "CPUExecutionProvider"] if provider != "CPUExecutionProvider" else [provider]
    session.get_inputs.return_value = [SimpleNamespace(type="tensor(float)")]
    session.get_modelmeta.return_value = SimpleNamespace(custom_metadata_map={"lid_dict": "{}", "textnorm_dict": "{}", "emo_dict": "{}"})
    return session


def sensevoice(encoder_provider, decoder_provider, requested="CUDA"):
    # 运行真实的编码器／解码器构造器，仅替换外部 ORT 边界。
    ort = SimpleNamespace(
        get_available_providers=lambda: ["CUDAExecutionProvider", "TensorrtExecutionProvider", "CPUExecutionProvider"],
        InferenceSession=Mock(side_effect=[ort_session(encoder_provider), ort_session(decoder_provider)]),
        SessionOptions=Mock(), GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL=1),
    )
    namespace = {"ort": ort, "Path": Path, "json": json,
                 "np": SimpleNamespace(float16="float16", float32="float32")}
    encoder = load_class("core/server/engines/sensevoice_onnx/inference/encoder.py", "SenseVoiceEncoder", namespace)
    decoder = load_class("core/server/engines/sensevoice_onnx/inference/decoder.py", "SenseVoiceDecoder", namespace)
    return SimpleNamespace(engine=SimpleNamespace(
        encoder=encoder("encoder.onnx", requested, dml_pad_to=0),
        decoder=decoder("decoder.onnx", requested, dml_pad_to=0),
    ))


def gguf_engine(model_type, providers, llm_use_gpu):
    config = SimpleNamespace(onnx_provider="CUDA", llm_use_gpu=llm_use_gpu)
    if model_type == "qwen_asr":
        return SimpleNamespace(config=config, engine=SimpleNamespace(
            encoder=SimpleNamespace(sess_fe=ort_session(providers[0]), sess_be=ort_session(providers[1])),
            model=object(),
        ))
    return SimpleNamespace(config=config, models=SimpleNamespace(
        encoder=SimpleNamespace(sess=ort_session(providers[0])),
        ctc_decoder=SimpleNamespace(sess=ort_session(providers[1])), model=object(),
    ))


class LoadedGpuBoostTest(unittest.TestCase):
    def config(self, model_type):
        return SimpleNamespace(
            model_type=model_type, gpu_boost_enabled=True, gpu_boost_backend_active=True,
            gpu_boost_cmd=["nvidia-smi", "-i", "0", "-lmc", "6000"],
            gpu_unboost_cmd=["nvidia-smi", "-i", "0", "-rmc"], gpu_unboost_timeout=1.0,
        )

    def initialize_worker(self, config, recognizer):
        # 保留真实 initialize 的加载／资格更新／ready 顺序。
        cls = load_class("core/server/worker/worker.py", "RecognizerWorker", {
            "Queue": object, "ListProxy": object, "logger": Mock(), "system": lambda: "Linux", "os": os,
        })
        worker = cls.__new__(cls)
        worker._setup_environment = Mock()
        worker.loader = SimpleNamespace(load=Mock(), recognizer=recognizer, punc_model=None, aligner=None)
        ready_eligibility = []
        worker.handler = SimpleNamespace(set_engine=Mock(), queue_out=SimpleNamespace(
            put=lambda _: ready_eligibility.append(config.gpu_boost_backend_active)))
        with patch("config_server.ServerConfig", config), patch.dict(os.environ, {}, clear=True):
            worker.initialize()
        return ready_eligibility

    def test_sensevoice_ort_cpu_fallback_disables_boost_before_worker_ready(self):
        for requested in ("CUDA", "TRT", "TENSORRT"):
            with self.subTest(requested=requested):
                config = self.config("sensevoice")
                recognizer = sensevoice("CPUExecutionProvider", "CPUExecutionProvider", requested=requested)
                self.assertEqual(self.initialize_worker(config, recognizer), [False])
                gpu = load_gpu_boost_namespace(config)
                state = SimpleNamespace(gpu_boosted=False, gpu_last_active=0.0)
                manager = gpu["GpuBoostManager"](state)
                with patch.object(gpu["subprocess"], "Popen") as popen:
                    manager.handle_command(SimpleNamespace(command="gpu_boost"))
                    manager.handle_command(SimpleNamespace(type="mic"))
                    manager.handle_command(SimpleNamespace(type="file"))
                    manager.shutdown()
                popen.assert_not_called()
                self.assertFalse(state.gpu_boosted)

    def test_sensevoice_keeps_either_gpu_session_eligible(self):
        for provider in ("CUDAExecutionProvider", "DmlExecutionProvider", "TensorrtExecutionProvider"):
            for providers in ((provider, "CPUExecutionProvider"), ("CPUExecutionProvider", provider)):
                with self.subTest(providers=providers):
                    config = self.config("sensevoice")
                    configure_gpu_boost_backend(config, {}, recognizer=sensevoice(*providers))
                    self.assertTrue(config.gpu_boost_backend_active)

    def test_gguf_models_use_loaded_sessions_and_keep_gpu_decoder_choice(self):
        for model_type in ("qwen_asr", "fun_asr_nano"):
            for providers, llm_gpu, expected in (
                (("CPUExecutionProvider", "CPUExecutionProvider"), False, False),
                (("CUDAExecutionProvider", "CPUExecutionProvider"), False, True),
                (("CPUExecutionProvider", "CUDAExecutionProvider"), False, True),
                (("CPUExecutionProvider", "CPUExecutionProvider"), True, True),
            ):
                with self.subTest(model_type=model_type, providers=providers, llm_gpu=llm_gpu):
                    config = self.config(model_type)
                    configure_gpu_boost_backend(config, {}, recognizer=gguf_engine(model_type, providers, llm_gpu))
                    self.assertEqual(config.gpu_boost_backend_active, expected)

    def test_forced_cpu_overrides_all_loaded_gpu_backends(self):
        for model_type, recognizer in (
            ("sensevoice", sensevoice("CUDAExecutionProvider", "CUDAExecutionProvider")),
            ("qwen_asr", gguf_engine("qwen_asr", ("CUDAExecutionProvider", "CUDAExecutionProvider"), True)),
            ("fun_asr_nano", gguf_engine("fun_asr_nano", ("CUDAExecutionProvider", "CUDAExecutionProvider"), True)),
            ("paraformer", SimpleNamespace(config=SimpleNamespace(provider="cuda"))),
        ):
            with self.subTest(model_type=model_type):
                config = self.config(model_type)
                configure_gpu_boost_backend(config, {"CAPSWRITER_INFERENCE_HARDWARE": " CPU "}, recognizer=recognizer)
                self.assertFalse(config.gpu_boost_backend_active)

    def test_sherpa_uses_loaded_config_when_sessions_are_not_exposed(self):
        for provider, expected in (("cpu", False), ("cuda", True), ("coreml", False)):
            with self.subTest(provider=provider):
                config = self.config("paraformer")
                configure_gpu_boost_backend(config, {}, recognizer=SimpleNamespace(config=SimpleNamespace(provider=provider)))
                self.assertEqual(config.gpu_boost_backend_active, expected)

    def test_runtime_ineligibility_preserves_owned_reset(self):
        for partial in (False, True):
            for cleanup in ("idle", "shutdown"):
                with self.subTest(partial=partial, cleanup=cleanup):
                    config = self.config("sensevoice")
                    gpu = load_gpu_boost_namespace(config)
                    state = SimpleNamespace(gpu_boosted=False, gpu_last_active=0.0)
                    manager = gpu["GpuBoostManager"](state)
                    clock = SimpleNamespace(monotonic=Mock(return_value=100.0))
                    with patch.dict(gpu, {"time": clock}), patch.object(manager, "_check_admin", return_value=True), patch.object(gpu["subprocess"], "Popen") as popen:
                        popen.return_value.wait.return_value = 7 if partial else 0
                        manager.ensure_boosted()
                        configure_gpu_boost_backend(config, {}, recognizer=sensevoice("CPUExecutionProvider", "CPUExecutionProvider"))
                        manager.ensure_boosted()
                        self.assertEqual(popen.call_count, 1)
                        clock.monotonic.return_value = 102.0
                        popen.return_value.wait.return_value = 0
                        manager.check_idle() if cleanup == "idle" else manager.shutdown()
                        self.assertEqual(popen.call_args.args[0], config.gpu_unboost_cmd)
                        self.assertEqual(popen.call_count, 2)
                        self.assertFalse(state.gpu_boosted)
                        manager.shutdown()
                        self.assertEqual(popen.call_count, 2)


if __name__ == "__main__":
    unittest.main()
