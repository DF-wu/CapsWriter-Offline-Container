from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from client.tui.i18n import CATALOGS, Translator, normalize_locale
from client.tui.storage import atomic_write_text, safe_output_stem, suggested_output_path


class I18nTest(unittest.TestCase):
    def test_catalogs_have_identical_keys(self) -> None:
        self.assertEqual(set(CATALOGS["en"]), set(CATALOGS["zh-Hant"]))

    def test_locale_aliases_and_rejection(self) -> None:
        self.assertEqual(normalize_locale("en_US"), "en")
        self.assertEqual(normalize_locale("zh-TW"), "zh-Hant")
        self.assertEqual(normalize_locale("traditional"), "zh-Hant")
        with self.assertRaisesRegex(ValueError, "choose en or zh-Hant"):
            normalize_locale("fr")

    def test_translator_formats_status(self) -> None:
        self.assertIn("meeting.wav", Translator("en")("status_transcribing", name="meeting.wav"))
        self.assertIn("處理中", Translator("zh-Hant")("status_transcribing", name="meeting.wav"))


class StorageTest(unittest.TestCase):
    def test_safe_stem_handles_windows_names_characters_and_length(self) -> None:
        self.assertEqual(safe_output_stem("CON"), "CON_audio")
        self.assertEqual(safe_output_stem('bad:name?.wav'), "bad_name_.wav")
        long = safe_output_stem("a" * 300)
        self.assertLessEqual(len(long), 120)
        self.assertEqual(long, safe_output_stem("a" * 300))

    def test_suggested_output_extension_matches_format(self) -> None:
        source = Path("/tmp/meeting.wav")
        self.assertEqual(suggested_output_path(source, "text"), Path("/tmp/meeting.txt"))
        self.assertEqual(suggested_output_path(source, "verbose_json"), Path("/tmp/meeting.json"))
        with self.assertRaises(ValueError):
            suggested_output_path(source, "xml")
        text_source = Path("/tmp/meeting.txt")
        self.assertEqual(
            suggested_output_path(text_source, "text"),
            Path("/tmp/meeting.transcript.txt"),
        )

    def test_atomic_write_creates_and_replaces_utf8_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory, "nested", "transcript.txt")
            self.assertEqual(atomic_write_text(target, "第一版"), target)
            self.assertEqual(target.read_text(encoding="utf-8"), "第一版")
            atomic_write_text(target, "second")
            self.assertEqual(target.read_text(encoding="utf-8"), "second")
            self.assertEqual(list(target.parent.glob(".*.tmp")), [])

    def test_atomic_write_preserves_old_file_and_cleans_temp_on_replace_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory, "transcript.txt")
            target.write_text("old", encoding="utf-8")
            with mock.patch("client.tui.storage.os.replace", side_effect=OSError("blocked")):
                with self.assertRaisesRegex(OSError, "blocked"):
                    atomic_write_text(target, "new")
            self.assertEqual(target.read_text(encoding="utf-8"), "old")
            self.assertEqual(list(target.parent.glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
