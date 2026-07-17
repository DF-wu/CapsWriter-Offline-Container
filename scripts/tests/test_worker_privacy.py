# coding: utf-8

from __future__ import annotations

import ast
import importlib.util
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple
import unittest


ROOT = Path(__file__).resolve().parents[2]
PRIVACY_PATH = ROOT / "core" / "server" / "privacy.py"
PIPELINE_PATH = ROOT / "core" / "server" / "worker" / "pipeline.py"
PROMPT_BUILDER_PATH = (
    ROOT
    / "core"
    / "server"
    / "engines"
    / "fun_asr_gguf"
    / "inference"
    / "prompt_builder.py"
)
ALIGNER_PATHS = (
    ROOT
    / "core"
    / "server"
    / "engines"
    / "force_aligner_gguf"
    / "inference"
    / "aligner.py",
    ROOT
    / "core"
    / "server"
    / "engines"
    / "qwen_asr_gguf"
    / "inference"
    / "aligner.py",
)


class FakeArray:
    """Minimal ndarray surface needed by the extracted prompt-builder test."""

    def __getitem__(self, _indices):
        return self

    def astype(self, _dtype):
        return self


class FakeNumpy:
    """Keep the dependency-light repository gate independent of NumPy."""

    ndarray = FakeArray
    float32 = object()

    @staticmethod
    def zeros(_shape):
        return FakeArray()


np = FakeNumpy()


class Capture:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def _record(self, *args, **_kwargs) -> None:
        self.messages.append(" ".join(str(arg) for arg in args))

    debug = _record
    error = _record
    info = _record
    print = _record
    warning = _record


def load_privacy_module():
    spec = importlib.util.spec_from_file_location("capswriter_test_privacy", PRIVACY_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load privacy module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def class_node(path: Path, name: str) -> ast.ClassDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")


def load_pipeline_class(privacy, logger: Capture, console: Capture):
    def session_key(socket_id, task_id):
        return str(socket_id), str(task_id)

    class WorkerState:
        def __init__(self) -> None:
            self.sessions = {}
            self.gpu_boosted = False
            self.gpu_last_active = 0.0

        def get_session(self, task_id, socket_id, source):
            key = session_key(socket_id, task_id)
            if key not in self.sessions:
                result = SimpleNamespace(
                    task_id=task_id,
                    socket_id=socket_id,
                    type=source,
                    duration=1.0,
                    time_start=0.0,
                    time_submit=0.0,
                    time_complete=0.0,
                    text="",
                    text_accu="",
                    tokens=[],
                    timestamps=[],
                    is_final=False,
                )
                self.sessions[key] = SimpleNamespace(result=result)
            return self.sessions[key]

    class TextFormatter:
        def __init__(self, _punc_model) -> None:
            pass

        @staticmethod
        def format(text: str) -> str:
            return f"{text}!"

    class EngineCapabilities:
        TIMESTAMPS = object()

    namespace = {
        "Config": SimpleNamespace(gpu_boost_enabled=False),
        "EngineCapabilities": EngineCapabilities,
        "Result": object,
        "Task": object,
        "TextFormatter": TextFormatter,
        "WorkerState": WorkerState,
        "console": console,
        "logger": logger,
        "merge_by_text": lambda previous, segment: previous + segment,
        "merge_tokens_by_sequence_matcher": (
            lambda prev_tokens, prev_timestamps, **_kwargs: (
                prev_tokens,
                prev_timestamps,
            )
        ),
        "process_audio_task": lambda _task, _result: [0.0],
        "process_tokens_safely": lambda values: list(values),
        "re": re,
        "session_key": session_key,
        "sync_tokens_from_text": lambda tokens, timestamps, _text: (
            tokens,
            timestamps,
        ),
        "time": time,
        "tokens_to_text": lambda tokens: "".join(tokens),
        "transcript_logging": privacy.transcript_logging,
        "transcript_logging_enabled": privacy.transcript_logging_enabled,
    }
    module = ast.Module(body=[class_node(PIPELINE_PATH, "TaskPipeline")], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(PIPELINE_PATH), "exec"), namespace)
    return namespace["TaskPipeline"], EngineCapabilities


class WorkerPrivacyTest(unittest.TestCase):
    def test_worker_pipeline_redacts_text_without_changing_result(self) -> None:
        secret = "private meeting transcript"
        privacy = load_privacy_module()
        logger = Capture()
        console = Capture()
        pipeline_class, capabilities = load_pipeline_class(privacy, logger, console)

        class Stream:
            def __init__(self) -> None:
                self.result = SimpleNamespace(text=secret, tokens=[], timestamps=[])

            def accept_waveform(self, _sample_rate, _samples) -> None:
                pass

        recognizer = SimpleNamespace(
            capabilities=[capabilities.TIMESTAMPS],
            create_stream=Stream,
            decode_stream=lambda *_args, **_kwargs: None,
        )
        pipeline = pipeline_class(recognizer)
        task = SimpleNamespace(
            task_id="private-task",
            socket_id="http:private-task",
            type="file",
            language="english",
            context="private prompt",
            samplerate=16000,
            offset=0.0,
            overlap=0.0,
            is_final=True,
            time_start=time.time(),
            time_submit=time.time(),
            log_transcript=False,
        )

        result = pipeline.process(task)
        emitted = "\n".join(logger.messages + console.messages)

        self.assertIn(secret, result.text)
        self.assertNotIn(secret, emitted)
        self.assertIn("redacted", emitted)
        self.assertTrue(privacy.transcript_logging_enabled())

    def test_websocket_default_preserves_existing_worker_output(self) -> None:
        secret = "visible websocket transcript"
        privacy = load_privacy_module()
        logger = Capture()
        console = Capture()
        pipeline_class, capabilities = load_pipeline_class(privacy, logger, console)

        class Stream:
            def __init__(self) -> None:
                self.result = SimpleNamespace(text=secret, tokens=[], timestamps=[])

            def accept_waveform(self, _sample_rate, _samples) -> None:
                pass

        pipeline = pipeline_class(
            SimpleNamespace(
                capabilities=[capabilities.TIMESTAMPS],
                create_stream=Stream,
                decode_stream=lambda *_args, **_kwargs: None,
            )
        )
        task = SimpleNamespace(
            task_id="ws-task",
            socket_id="ws",
            type="mic",
            language="auto",
            context="",
            samplerate=16000,
            offset=0.0,
            overlap=0.0,
            is_final=True,
            time_start=time.time(),
            time_submit=time.time(),
        )

        pipeline.process(task)

        self.assertIn(secret, "\n".join(logger.messages + console.messages))

    def test_private_pipeline_exception_detail_is_not_logged(self) -> None:
        secret = "private prompt embedded in engine error"
        privacy = load_privacy_module()
        logger = Capture()
        console = Capture()
        pipeline_class, capabilities = load_pipeline_class(privacy, logger, console)

        class Stream:
            def __init__(self) -> None:
                self.result = SimpleNamespace(text="", tokens=[], timestamps=[])

            def accept_waveform(self, _sample_rate, _samples) -> None:
                pass

        def fail_decode(*_args, **_kwargs):
            raise RuntimeError(secret)

        pipeline = pipeline_class(
            SimpleNamespace(
                capabilities=[capabilities.TIMESTAMPS],
                create_stream=Stream,
                decode_stream=fail_decode,
            )
        )
        task = SimpleNamespace(
            task_id="private-error",
            socket_id="http:private-error",
            type="file",
            language="english",
            context="private context",
            samplerate=16000,
            offset=0.0,
            overlap=0.0,
            is_final=True,
            time_start=time.time(),
            time_submit=time.time(),
            log_transcript=False,
        )

        with self.assertRaisesRegex(RuntimeError, secret):
            pipeline.process(task)

        emitted = "\n".join(logger.messages + console.messages)
        self.assertNotIn(secret, emitted)
        self.assertIn("redacted", emitted)

    def test_fun_asr_prompt_logging_obeys_task_context(self) -> None:
        secret = "private FunASR context"
        detected_hotword = "private detected hotword"
        privacy = load_privacy_module()
        logger = Capture()
        llama = SimpleNamespace(text_to_tokens=lambda _vocab, _text: [0])
        namespace = {
            "List": List,
            "Optional": Optional,
            "Tuple": Tuple,
            "llama": llama,
            "logger": logger,
            "np": np,
            "transcript_logging_enabled": privacy.transcript_logging_enabled,
        }
        module = ast.Module(
            body=[class_node(PROMPT_BUILDER_PATH, "PromptBuilder")],
            type_ignores=[],
        )
        ast.fix_missing_locations(module)
        exec(compile(module, str(PROMPT_BUILDER_PATH), "exec"), namespace)
        builder = namespace["PromptBuilder"]("vocab", np.zeros((1, 2)))

        with privacy.transcript_logging(False):
            *_unused, prompt = builder.build_prompt(
                hotwords=[detected_hotword],
                context=secret,
            )

        self.assertIn(secret, prompt)
        self.assertIn(detected_hotword, prompt)
        emitted = "\n".join(logger.messages)
        self.assertNotIn(secret, emitted)
        self.assertNotIn(detected_hotword, emitted)
        self.assertTrue(any("redacted" in message for message in logger.messages))

    def test_forced_aligner_warnings_are_privacy_gated(self) -> None:
        for path in ALIGNER_PATHS:
            with self.subTest(path=path.relative_to(ROOT)):
                source = path.read_text(encoding="utf-8")
                self.assertIn("transcript_logging_enabled", source)
                self.assertIn(
                    'item.text if transcript_logging_enabled() else "[redacted]"',
                    source,
                )


if __name__ == "__main__":
    unittest.main()
