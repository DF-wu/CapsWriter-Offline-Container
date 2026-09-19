"""Behavioral regressions for the September 2026 upstream integration."""
from __future__ import annotations

import ast
import asyncio
import base64
import ctypes
import difflib
import importlib.util
import logging
import re
import sys
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]


def load_definitions(relative_path, names=None, **namespace):
    """Load pure definitions without starting desktop/backend import side effects."""
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    body = [node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and (names is None or node.name in names)]
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, str(path), "exec", dont_inherit=False), namespace)
    return namespace


class UpstreamRefreshTest(unittest.TestCase):
    def test_overlap_inside_token_preserves_space_and_history(self):
        namespace = load_definitions(
            "core/server/merger/token_merger.py", difflib=difflib,
            logger=logging.getLogger(__name__),
            Punctuation=SimpleNamespace(ALL=",.!?，。！？"),
        )
        merge = namespace["merge_tokens_by_sequence_matcher"]
        tail = ['hairpiece', '?', ' ', 'Wait', ',', ' ', 'does', ' ',
                'he', ' ', 'eat', ' ', 'chalk', '?']
        new = ['Wait', ',', ' ', 'does', ' ', 'he', ' ', 'eat', ' ',
               'chalk', '? ', 'Just', ' ', 'once']
        for history in ([], ['prior'] * 70):
            with self.subTest(history=len(history)):
                previous = history + tail
                previous_ts = [float(i) for i in range(len(previous))]
                incoming_ts = [i * 0.1 for i in range(len(new))]
                tokens, times = merge(previous, previous_ts, new, incoming_ts, 60, 4)
                self.assertEqual(''.join(tokens), ''.join(history + tail) + ' Just once')
                self.assertEqual(tokens[:len(history)], history)
                self.assertEqual(times[:len(history)], previous_ts[:len(history)])
                self.assertEqual(len(tokens), len(times))
                self.assertEqual(times[tokens.index('Just')], 61.1)

    def test_threshold_crossing_sends_cached_and_current_audio(self):
        from scripts.tests.test_client_recorder_cleanup import (
            FakeArray, FakeState, FakeWebSocketManager, load_recorder_class,
        )

        class Chunk(FakeArray):
            def __init__(self, payload):
                self.payload = payload

            def tobytes(self):
                return self.payload

        recorder_class = load_recorder_class()
        namespace = recorder_class.record_and_send.__globals__
        namespace['Config'].save_audio = False
        namespace['Config'].threshold = 2.0
        namespace['np'] = SimpleNamespace(
            concatenate=lambda items: Chunk(b''.join(item.payload for item in items)),
            mean=lambda data, axis: data,
        )

        async def scenario():
            state = FakeState()
            websocket = FakeWebSocketManager()
            for item in (
                {'type': 'begin', 'time': 1.0},
                {'type': 'data', 'time': 2.0, 'data': Chunk(b'before')},
                {'type': 'data', 'time': 3.0, 'data': Chunk(b'crossing')},
                {'type': 'finish', 'time': 4.0},
            ):
                await state.queue_in.put(item)
            recorder = recorder_class(SimpleNamespace(state=state, ws=websocket))
            await recorder.record_and_send()
            await asyncio.wait_for(state.queue_in.join(), timeout=0.1)
            return state, websocket

        state, websocket = asyncio.run(scenario())
        self.assertEqual(state.released_chunks, 2)
        self.assertEqual(len(websocket.messages), 2)
        self.assertEqual(base64.b64decode(websocket.messages[0].data), b'beforecrossing')
        self.assertTrue(websocket.messages[1].is_final)

    def test_numeric_normalization_keeps_verbs_and_rate_terms(self):
        name = "_upstream_refresh_itn"
        path = ROOT / "core/tools/chinese_itn/__init__.py"
        spec = importlib.util.spec_from_file_location(
            name, path, submodule_search_locations=[str(path.parent)],
        )
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules):
            sys.modules[name] = module
            spec.loader.exec_module(module)
            for before, after in (
                ('点一下按钮', '点一下按钮'), ('点击三次', '点击3次'),
                ('万一失败', '万一失败'), ('手续费万三', '手续费万三'),
                ('上百万', '上百万'), ('一万三', '13000'), ('三点一四', '3.14'),
            ):
                with self.subTest(before=before):
                    self.assertEqual(module.chinese_to_num(before), after)

    def test_smart_split_keeps_abbreviations_decimals_and_short_phrases(self):
        handler = load_definitions(
            "core/client/transcribe/result_handler.py", {"ResultHandler"}, re=re,
        )["ResultHandler"]
        source = "Dr. Jones uses 3.14 daily, yes. Philip H. works here."
        lines = handler.smart_split(source).splitlines()
        self.assertEqual([line.strip() for line in lines],
                         ["Dr. Jones uses 3.14 daily, yes.", "Philip H. works here."])
        self.assertEqual(
            [line.strip() for line in handler.smart_split(
                "We have four words, these are four words."
            ).splitlines()], ["We have four words,", "these are four words."])

    def test_subtitles_use_next_start_gap_and_cap_long_silence(self):
        constants = SimpleNamespace(Punctuation=SimpleNamespace(ALL=",.!?，。！？"))
        namespace = load_definitions(
            "core/tools/srt_from_txt.py", {"lines_match_words"},
            re=re, difflib=difflib, timedelta=timedelta,
            srt=SimpleNamespace(Subtitle=lambda **kwargs: SimpleNamespace(**kwargs)),
        )
        with patch.dict(sys.modules, {"core.constants": constants}):
            for second_start, expected_end in ((1.0, 0.9), (10.0, 1.0)):
                with self.subTest(second_start=second_start):
                    subtitles = namespace["lines_match_words"](
                        ['Hello', 'World'],
                        [{'word': 'Hello', 'start': 0.0}, {'word': 'World', 'start': second_start}],
                    )
                    self.assertEqual(subtitles[0].end.total_seconds(), expected_end)
                    self.assertGreater(subtitles[1].end, subtitles[1].start)

    def test_llm_struct_layout_matches_shared_b7798_asr_runtime(self):
        names = {"llama_model_params", "llama_context_params"}
        llm = load_definitions("core/server/engines/llama/llama.py", names, ctypes=ctypes)
        for engine in ('qwen_asr_gguf', 'fun_asr_gguf', 'force_aligner_gguf'):
            asr = load_definitions(f"core/server/engines/{engine}/inference/llama.py", names, ctypes=ctypes)
            for name in names:
                with self.subTest(engine=engine, structure=name):
                    self.assertEqual(ctypes.sizeof(llm[name]), ctypes.sizeof(asr[name]))
                    self.assertEqual([f[0] for f in llm[name]._fields_], [f[0] for f in asr[name]._fields_])
                    for field, *_ in asr[name]._fields_:
                        self.assertEqual(getattr(llm[name], field).offset, getattr(asr[name], field).offset)

    def test_penalty_sampler_calls_b7798_four_argument_abi(self):
        penalties = Mock(return_value=2)
        namespace = load_definitions(
            "core/server/engines/llama/llama.py", {"LlamaSampler"},
            llama_sampler_chain_default_params=Mock(), llama_sampler_chain_init=Mock(return_value=1),
            llama_sampler_chain_add=Mock(), llama_sampler_init_penalties=penalties,
            llama_sampler_init_greedy=Mock(), llama_sampler_free=Mock(),
        )
        sampler = namespace['LlamaSampler'](
            temperature=0, repeat_penalty=1.1, frequency_penalty=0.2,
            presence_penalty=0.3, penalty_last_n=64, n_vocab=1000, seed=1,
        )
        penalties.assert_called_once_with(64, 1.1, 0.2, 0.3)
        del sampler


if __name__ == '__main__':
    unittest.main()
