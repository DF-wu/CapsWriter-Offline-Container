"""Client entrypoint and output regressions reported during PR review."""
from __future__ import annotations

import ast
import asyncio
import logging
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.tests.test_upstream_refresh import load_definitions


class ClientReviewTests(unittest.TestCase):
    def test_requested_files_select_file_mode_without_dropping_missing_paths(self):
        for include_valid in (False, True):
            with self.subTest(include_valid=include_valid), tempfile.TemporaryDirectory() as tmp:
                missing = Path(tmp) / "missing.wav"
                valid = Path(tmp) / "present.wav"
                valid.write_bytes(b"audio fixture")
                requested = [valid, missing] if include_valid else [missing]
                dispatched = []

                class FileRunner:
                    def __init__(self, app, files):
                        self.files = files

                    async def run(self):
                        dispatched.extend(self.files)

                def mic_runner(app):
                    self.fail("Explicit file arguments must not start microphone mode")

                client_type = load_definitions(
                    "core/client/app.py", {"CapsWriterClient"},
                    Path=Path, os=os, sys=sys, register_signal=lambda *_: None,
                    FileRunner=FileRunner, MicRunner=mic_runner,
                )["CapsWriterClient"]
                client = object.__new__(client_type)
                client.loop = asyncio.new_event_loop()
                try:
                    with patch.object(sys, "argv", ["start_client.py", *map(str, requested)]):
                        client.start()
                finally:
                    client.loop.close()
                self.assertEqual(dispatched, requested)

    def test_missing_media_reports_error_before_connecting(self):
        transcriber = load_definitions(
            "core/client/transcribe/file_transcriber.py", {"FileTranscriber"},
            logger=logging.getLogger("missing-media-review"),
        )["FileTranscriber"]
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.wav"
            instance = transcriber(SimpleNamespace(), missing)
            with self.assertLogs("missing-media-review", level="ERROR") as logs:
                self.assertFalse(asyncio.run(instance.check()))
            self.assertIn(str(missing), logs.output[0])

    def test_auto_enter_matches_mixed_case_foreground_executable(self):
        async def run_case(process_name):
            keys, outputs = [], []
            async def output(text, *, paste):
                outputs.append((text, paste))

            config = SimpleNamespace(
                traditional_convert=False, hot=False, paste=False,
                paste_apps=["CODE.EXE"], enter_apps=[("code.exe", 0)],
                llm_enabled=False, save_audio=False,
            )
            console = SimpleNamespace(print=lambda *_: None, line=lambda: None)
            state = SimpleNamespace(set_output_text=lambda *_: None)
            correction = SimpleNamespace(text="hello", matches=[], similars=[])
            hotword = SimpleNamespace(
                get_phoneme_corrector=lambda: SimpleNamespace(correct=lambda *_a, **_k: correction),
                get_rule_corrector=lambda: SimpleNamespace(substitute=lambda text: text),
            )
            processor_type = load_definitions(
                "core/client/output/result_processor.py", {"ResultProcessor", "_auto_enter"},
                asyncio=asyncio, time=time, Config=config, console=console,
                logger=logging.getLogger(__name__),
                TextOutput=SimpleNamespace(strip_punc=lambda text: text),
                get_active_window_info=lambda: {"process_name": process_name},
                keyboard=SimpleNamespace(press_and_release=keys.append),
                broadcast_output_udp=lambda *_: None,
            )["ResultProcessor"]
            # Load the real async helper too (the shared loader selects classes
            # and synchronous definitions only).
            helper_path = Path(__file__).resolve().parents[2] / "core/client/output/result_processor.py"
            helper = next(node for node in ast.parse(helper_path.read_text(encoding="utf-8")).body
                          if isinstance(node, ast.AsyncFunctionDef) and node.name == "_auto_enter")
            exec(compile(ast.Module(body=[helper], type_ignores=[]), str(helper_path), "exec"),
                 processor_type._handle_message.__globals__)
            processor = processor_type(SimpleNamespace(
                state=state, hotword=hotword, output=SimpleNamespace(output=output),
            ))
            await processor._handle_message(SimpleNamespace(
                text="hello", time_complete=2, time_submit=1, is_final=True,
            ))
            # Drain the real auto-enter task, including its zero-delay scheduling.
            pending = asyncio.all_tasks() - {asyncio.current_task()}
            if pending:
                await asyncio.gather(*pending)
            return keys, outputs

        self.assertEqual(asyncio.run(run_case("Code.exe")), (["enter"], [("hello", True)]))
        self.assertEqual(asyncio.run(run_case("other.exe")), ([], [("hello", False)]))


if __name__ == "__main__":
    unittest.main()
