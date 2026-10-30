"""Shortcut recording behavior without a display, global hook, or microphone."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fork_client.shortcut_capture import ShortcutCapture, tk_key_name
from fork_client.shortcut_keys import normalize_shortcut


def event(keysym, keycode, char=""):
    return SimpleNamespace(keysym=keysym, keycode=keycode, char=char)


class ShortcutCaptureTest(unittest.TestCase):
    def test_capture_issue_examples_and_ignore_repeated_presses(self):
        for keys in (("grave", "1"), ("F1", "F2", "F3"), ("F13",), ("F14",)):
            with self.subTest(keys=keys):
                capture = ShortcutCapture()
                events = [event(key, i + 10) for i, key in enumerate(keys)]
                for key in events:
                    capture.press(key)
                    capture.press(key)
                for key in events[:-1]:
                    self.assertIsNone(capture.release(key))
                self.assertEqual(capture.release(events[-1]), normalize_shortcut("+".join("`" if key == "grave" else key for key in keys)))

    def test_release_uses_keycode_even_when_modifier_changes_character(self):
        capture = ShortcutCapture()
        capture.press(event("Control_L", 37))
        capture.press(event("Shift_L", 50))
        capture.press(event("A", 38, "\x01"))
        self.assertIsNone(capture.release(event("Shift_L", 50)))
        self.assertIsNone(capture.release(event("Control_L", 37)))
        self.assertEqual(capture.release(event("a", 38, "a")), normalize_shortcut("ctrl_l+shift_l+a"))

    def test_physical_modifier_sides_are_not_collapsed(self):
        for left, right, expected in (("Control_L", "Control_R", "ctrl_l+ctrl_r"), ("Shift_L", "Shift_R", "shift_l+shift_r"), ("Super_L", "Super_R", "cmd_l+cmd_r")):
            with self.subTest(left=left):
                capture = ShortcutCapture()
                capture.press(event(left, 10))
                capture.press(event(right, 11))
                self.assertIsNone(capture.release(event(left, 10)))
                self.assertEqual(capture.release(event(right, 11)), normalize_shortcut(expected))
        self.assertEqual(tk_key_name(event("Shift_R", 11)), "shift_r")

    def test_capture_freezes_simultaneous_chord_at_first_release(self):
        capture = ShortcutCapture()
        capture.press(event("F1", 10))
        capture.press(event("F2", 11))
        self.assertIsNone(capture.release(event("F1", 10)))
        capture.press(event("F3", 12))
        self.assertIsNone(capture.release(event("F2", 11)))
        self.assertEqual(capture.release(event("F3", 12)), "f1+f2")

    def test_named_keys_and_unknown_codes(self):
        for keysym, expected in (("space", "space"), ("plus", "plus"), ("KP_Add", "numpad_add"), ("KP_End", "numpad1"), ("Prior", "page_up"), ("Return", "enter"), ("asciitilde", "~")):
            with self.subTest(keysym=keysym):
                self.assertEqual(tk_key_name(event(keysym, 42)), expected)
        # An X11 keycode is not a portable vk_N; never silently save one.
        with self.assertRaises(ValueError):
            tk_key_name(event("??", 191))
        self.assertEqual(tk_key_name(event("Cyrillic_ya", 42, "я")), "я")
        self.assertIsNone(ShortcutCapture().release(event("F13", 191)))

    def test_symbols_do_not_assume_us_keyboard_positions(self):
        for keysym, char in (("ampersand", "&"), ("quotedbl", '"'), ("parenleft", "("), ("underscore", "_")):
            with self.subTest(keysym=keysym):
                capture = ShortcutCapture()
                key = event(keysym, 10, char)
                capture.press(key)
                self.assertEqual(capture.release(key), char)

    def test_active_layout_supplies_unshifted_character(self):
        for keysym, shifted, base in (("exclam", "!", "1"), ("ampersand", "&", "&"), ("2", "2", "é"), ("question", "?", "ß")):
            with self.subTest(keysym=keysym), patch(
                "fork_client.shortcut_capture.unshifted_character", return_value=base,
            ):
                self.assertEqual(tk_key_name(event(keysym, 10, shifted)), base)


if __name__ == "__main__":
    unittest.main()
