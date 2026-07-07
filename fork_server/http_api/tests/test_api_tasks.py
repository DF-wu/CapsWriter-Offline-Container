# coding: utf-8

from __future__ import annotations

import unittest

from fork_server.http_api import transcription_tasks as tasks


class HttpApiTaskSubmissionTest(unittest.TestCase):
    def test_prompt_context_is_normalized_and_bounded(self) -> None:
        prompt = "  alpha\r\nbeta\r" + ("x" * tasks.MAX_PROMPT_CONTEXT_CHARS)
        context = tasks.normalize_prompt_context(prompt)
        self.assertTrue(context.startswith("alpha\nbeta\n"))
        self.assertNotIn("\r", context)
        self.assertEqual(len(context), tasks.MAX_PROMPT_CONTEXT_CHARS)

    def test_language_hint_accepts_openai_style_aliases(self) -> None:
        cases = {
            None: "auto",
            "": "auto",
            "zh": "chinese",
            "zh_CN": "chinese",
            "pt-BR": "pt-br",
            "EN": "english",
            "ja": "japanese",
            "yue": "cantonese",
            "spanish": "spanish",
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(tasks.normalize_language_hint(value), expected)

    def test_language_hint_rejects_oversized_or_unsafe_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "at most"):
            tasks.normalize_language_hint("x" * (tasks.MAX_LANGUAGE_HINT_CHARS + 1))
        for value in ("en\nwarning", "en us", "zh/../../"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "letters"):
                tasks.normalize_language_hint(value)

    def test_short_audio_spec_uses_upstream_task_field_names(self) -> None:
        pcm = b"\0" * tasks.seconds_to_bytes(1.0)
        specs = list(
            tasks.iter_transcription_task_specs(
                task_id="task-short",
                socket_id="http:task-short",
                pcm=pcm,
                time_start=123.0,
                context="meeting terms",
                language="chinese",
            )
        )

        self.assertEqual(len(specs), 1)
        spec = specs[0]
        self.assertEqual(spec["type"], "file")
        self.assertNotIn("source", spec)
        self.assertEqual(spec["data"], pcm)
        self.assertEqual(spec["context"], "meeting terms")
        self.assertEqual(spec["language"], "chinese")
        self.assertTrue(spec["is_final"])

    def test_long_audio_specs_propagate_hints_to_all_segments(self) -> None:
        pcm = b"\1" * tasks.seconds_to_bytes(3.0)

        specs = list(
            tasks.iter_transcription_task_specs(
                task_id="task-long",
                socket_id="http:task-long",
                pcm=pcm,
                time_start=123.0,
                context="CapsWriter, FunASR",
                language="english",
                seg_duration=1.0,
                seg_overlap=0.5,
            )
        )

        self.assertEqual([spec["offset"] for spec in specs], [0.0, 1.0, 2.0])
        self.assertEqual(
            [spec["is_final"] for spec in specs], [False, False, True]
        )
        for spec in specs:
            self.assertEqual(spec["type"], "file")
            self.assertEqual(spec["context"], "CapsWriter, FunASR")
            self.assertEqual(spec["language"], "english")
            self.assertEqual(spec["socket_id"], "http:task-long")


if __name__ == "__main__":
    unittest.main()
