"""Regression coverage for subtitle splitting and word-start-only timing."""
from __future__ import annotations

import difflib
import importlib.util
import re
import sys
import unittest
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from scripts.tests.test_upstream_refresh import load_definitions


class SubtitleReviewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        constants = SimpleNamespace(**load_definitions("core/constants.py"))
        constant_import = patch.dict(sys.modules, {"core.constants": constants})
        constant_import.start()
        cls.addClassCleanup(constant_import.stop)
        cls.handler = load_definitions(
            "core/client/transcribe/result_handler.py", {"ResultHandler"}, re=re,
        )["ResultHandler"]
        cls.match_words = staticmethod(load_definitions(
            "core/tools/srt_from_txt.py", {"lines_match_words"},
            re=re, difflib=difflib, timedelta=timedelta,
            srt=SimpleNamespace(Subtitle=lambda **kwargs: SimpleNamespace(**kwargs)),
        )["lines_match_words"])

    def test_chinese_comma_splits_without_whitespace(self):
        self.assertEqual(
            self.handler.smart_split("今天我們測試字幕分行，明天繼續測試其他功能。"),
            "今天我們測試字幕分行，\n明天繼續測試其他功能。",
        )

    def test_english_comma_still_requires_whitespace(self):
        for source, expected in (
            ("We have four words,these are four words.",
             ["We have four words,these are four words."]),
            ("We have four words, these are four words.",
             ["We have four words,", "these are four words."]),
        ):
            with self.subTest(source=source):
                self.assertEqual(
                    [line.strip() for line in self.handler.smart_split(source).splitlines()],
                    expected,
                )

    def test_cjk_threshold_ignores_punctuation_and_counts_kana(self):
        for source, expected in (
            ("甲乙丙，丁戊己庚。", "甲乙丙，丁戊己庚。"),
            ("甲乙丙丁，戊己庚。", "甲乙丙丁，戊己庚。"),
            ("甲乙丙丁，戊己庚辛。", "甲乙丙丁，\n戊己庚辛。"),
            ("あいうえ，カキクケ。", "あいうえ，\nカキクケ。"),
            ("あいう，カキクケ。", "あいう，カキクケ。"),
            ("㐀㐁㐂㐃，甲乙丙丁。", "㐀㐁㐂㐃，\n甲乙丙丁。"),
            ("甲乙ASR，丙丁戊己。", "甲乙ASR，丙丁戊己。"),
        ):
            with self.subTest(source=source):
                self.assertEqual(self.handler.smart_split(source), expected)

    def test_count_units_excludes_punctuation_only_words(self):
        for source, expected in (
            ("，。！？ … —", 0), ("甲乙丙，", 3),
            ("あいうえカキクケ。", 8), ("㐀㐁㐂㐃", 4),
            ("甲乙 ASR test!", 4), ("can't re-enter", 2),
        ):
            with self.subTest(source=source):
                self.assertEqual(self.handler.count_units(source), expected)

    def assert_subtitles(self, lines, words, expected):
        subtitles = self.match_words(lines, words)
        self.assertEqual(
            [(subtitle.index, subtitle.content,
              subtitle.start.total_seconds(), subtitle.end.total_seconds())
             for subtitle in subtitles],
            expected,
        )

    def test_dense_line_end_cannot_precede_its_last_token(self):
        self.assert_subtitles(
            ["甲乙", "丙"],
            [{"word": "甲", "start": 0.0}, {"word": "乙", "start": 0.95},
             {"word": "丙", "start": 1.0}],
            [(1, "甲乙", 0.0, 0.95), (2, "丙", 1.0, 1.2)],
        )

    def test_multitoken_line_keeps_gap_or_caps_silence(self):
        for next_start, expected_end in ((1.0, 0.9), (10.0, 1.4)):
            with self.subTest(next_start=next_start):
                self.assert_subtitles(
                    ["甲乙", "丙丁"],
                    [{"word": "甲", "start": 0.0}, {"word": "乙", "start": 0.4},
                     {"word": "丙", "start": next_start},
                     {"word": "丁", "start": next_start + 0.2}],
                    [(1, "甲乙", 0.0, expected_end),
                     (2, "丙丁", next_start, next_start + 0.4)],
                )

    def test_final_line_estimates_only_last_token_tail(self):
        line = "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉"
        self.assert_subtitles(
            [line],
            [{"word": char, "start": index * 0.2} for index, char in enumerate(line)],
            [(1, line, 0.0, 4.0)],
        )

    def test_final_english_token_preserves_word_boundaries(self):
        self.assert_subtitles(
            ["We have four words."],
            [{"word": "We", "start": 0.0}, {"word": "have", "start": 0.3},
             {"word": "four words", "start": 0.6}],
            [(1, "We have four words.", 0.0, 1.3)],
        )

    def test_final_kana_token_uses_character_duration(self):
        self.assert_subtitles(
            ["あいうえ"],
            [{"word": "あい", "start": 0.0}, {"word": "うえ", "start": 0.4}],
            [(1, "あいうえ", 0.0, 0.8)],
        )


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("srt", "typer", "rich", "colorama")),
    "real subtitle file integration requires srt, typer, rich, colorama",
)
class SubtitleFileReviewTest(unittest.TestCase):
    def test_chinese_split_generates_complete_srt_file(self):
        from core.tools.srt_from_txt import generate_srt_file

        handler = load_definitions(
            "core/client/transcribe/result_handler.py", {"ResultHandler"}, re=re,
        )["ResultHandler"]
        lines = handler.smart_split("甲乙丙丁，戊己庚辛。").splitlines()
        words = [{"word": char, "start": index * 0.2}
                 for index, char in enumerate("甲乙丙丁戊己庚辛")]
        with TemporaryDirectory() as directory:
            output = Path(directory) / "transcript.srt"
            generate_srt_file(words, lines, output)
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                "1\n00:00:00,000 --> 00:00:00,700\n甲乙丙丁\n\n"
                "2\n00:00:00,800 --> 00:00:01,600\n戊己庚辛\n\n",
            )


if __name__ == "__main__":
    unittest.main()
