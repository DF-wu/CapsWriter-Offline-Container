"""Shortcut parsing and real manager dispatch, without a desktop dependency."""
from __future__ import annotations

from itertools import permutations
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from threading import Event

from fork_client.shortcut_keys import normalize_shortcut, parse_shortcut
from scripts.tests.test_desktop_portability import (
    ROOT, FakeEmulator, FakeEventHandler, FakeShortcut, PLATFORM_SUPPORT,
    load_module, loaded_shortcut_manager,
)


class ParsingTest(unittest.TestCase):
    def test_canonical_names(self):
        for raw, expected in {
            ' ` + 1 ': '1+`', 'F3+F1+f2': 'f1+f2+f3',
            '<CTRL> + <ALT> + <Shift> + A': 'a+alt+ctrl+shift',
            'control+F24': 'ctrl+f24', 'Caps Lock': 'caps_lock',
            'plus+f13': 'f13+plus', 'numpad_add+NUMPAD0': 'numpad0+numpad_add',
            'vk_000124+f14': 'f14+vk_124', ' ': 'space', '+': 'plus', ' + ': 'plus',
            'LEFT CTRL+right shift': 'ctrl_l+shift_r',
            '<esc>': 'esc', 'x1': 'x1',
        }.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_shortcut(raw), expected)
                self.assertEqual(parse_shortcut(raw), tuple(expected.split('+')))
                self.assertEqual(normalize_shortcut(expected), expected)

    def test_invalid_and_duplicate_tokens(self):
        for raw in ('', '\t', 'a+', '+a', 'a++b', 'F25', 'f0',
                    'vk_0', 'vk_65536', 'vk_-1', 'vk_1.0', 'vk_0x20',
                    'a+A', 'ctrl+control', 'alt_l+left_alt', 'vk_01+vk_1',
                    '<ctrl', 'ctrl>', '<<ctrl>>', '<>', '<+>', 'x1+a',
                    'numpad10', 'not_a_key', None, 42):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                normalize_shortcut(raw)

    def test_every_function_key_and_vk_boundary(self):
        for number in range(1, 25):
            self.assertEqual(parse_shortcut(f'<F{number}>'), (f'f{number}',))
        self.assertEqual(parse_shortcut('vk_65535'), ('vk_65535',))


class RuntimeTest(unittest.TestCase):
    def manager(self, module, *bindings):
        manager = module.ShortcutManager.__new__(module.ShortcutManager)
        manager.tasks = {
            normalize_shortcut(key): SimpleNamespace(shortcut=FakeShortcut(
                normalize_shortcut(key), 'keyboard')) for key in bindings
        }
        manager._emulator = FakeEmulator()
        manager._event_handler = FakeEventHandler()
        manager._restoring_keys = set()
        manager.keyboard_listener = SimpleNamespace(suppress_event=Mock())
        return manager

    @staticmethod
    def events(manager):
        return [(kind, name) for kind, name, _ in manager._event_handler.events]

    @staticmethod
    def key(name):
        return SimpleNamespace(char=name) if len(name) == 1 else SimpleNamespace(name=name)

    def test_all_press_and_release_orders_rearm_only_after_all_up(self):
        with loaded_shortcut_manager() as module:
            for order in permutations(('f1', 'f2', 'f3')):
                for release_order in permutations(order):
                    with self.subTest(order=order, release=release_order):
                        manager = self.manager(module, 'f1+f2+f3')
                        down, up = manager.create_portable_keyboard_callbacks()
                        for name in order:
                            down(self.key(name))
                            down(self.key(name))  # OS autorepeat
                        self.assertEqual(self.events(manager), [('down', 'f1+f2+f3')])
                        up(self.key(release_order[0]))
                        down(self.key(release_order[0]))  # partial release cannot retrigger
                        self.assertEqual(self.events(manager), [
                            ('down', 'f1+f2+f3'), ('up', 'f1+f2+f3')])
                        for name in release_order:
                            up(self.key(name))
                        for name in order:
                            down(self.key(name))
                        self.assertEqual(self.events(manager)[-1], ('down', 'f1+f2+f3'))
                        self.assertEqual(len(self.events(manager)), 3)

    def test_sequence_without_overlap_never_triggers(self):
        with loaded_shortcut_manager() as module:
            manager = self.manager(module, '`+1')
            down, up = manager.create_portable_keyboard_callbacks()
            down(self.key('`')); up(self.key('`'))
            down(self.key('1')); up(self.key('1'))
            self.assertEqual(self.events(manager), [])
            down(self.key('1')); down(self.key('`'))
            self.assertEqual(self.events(manager), [('down', '1+`')])

    def test_win32_shifted_character_keeps_recorded_unshifted_name(self):
        with loaded_shortcut_manager() as module:
            manager = self.manager(module, 'shift_l+1')
            with patch.object(module.KeyMapper, 'vk_to_name', create=True,
                              side_effect=['shift_l', '!', 'shift_l', '1'] * 2), patch.object(
                module.KeyMapper, 'scan_to_name', create=True, return_value='1'
            ) as scan_name:
                hook = manager.create_keyboard_filter()
                for _ in range(2):
                    for message, vk, scan in ((1, 0xA0, 42), (1, 49, 2), (2, 0xA0, 42), (2, 49, 2)):
                        hook(message, SimpleNamespace(vkCode=vk, scanCode=scan, flags=0))
                self.assertEqual(scan_name.call_count, 4)
            self.assertEqual(self.events(manager), [
                ('down', '1+shift_l'), ('up', '1+shift_l'),
                ('down', '1+shift_l'), ('up', '1+shift_l'),
            ])
            self.assertEqual(manager._physical_keys, {})

    def test_modifiers_distinct_physical_keys_and_sides(self):
        with loaded_shortcut_manager() as module:
            manager = self.manager(module, 'ctrl+a', 'left_ctrl+a', 'ctrl_r+a',
                                   'ctrl+ctrl_r+a')
            down, up = manager.create_portable_keyboard_callbacks()
            down(self.key('ctrl_r')); down(self.key('a'))
            self.assertEqual(self.events(manager), [('down', 'a+ctrl'), ('down', 'a+ctrl_r')])
            down(self.key('ctrl_l'))
            self.assertEqual(self.events(manager)[-2:], [
                ('down', 'a+ctrl_l'), ('down', 'a+ctrl+ctrl_r')])
            up(self.key('ctrl_r'))
            self.assertIn(('up', 'a+ctrl_r'), self.events(manager))
            self.assertNotIn(('up', 'a+ctrl_l'), self.events(manager))

    def test_explicit_left_modifiers_do_not_match_right_side(self):
        with loaded_shortcut_manager() as module:
            for modifier in ('ctrl', 'alt'):
                manager = self.manager(module, modifier + '+a', modifier + '_l+a',
                                       modifier + '_r+a')
                down, up = manager.create_portable_keyboard_callbacks()
                down(self.key(modifier + '_r')); down(self.key('a'))
                self.assertEqual(self.events(manager), [
                    ('down', f'a+{modifier}'), ('down', f'a+{modifier}_r')])
                up(self.key('a')); up(self.key(modifier + '_r'))
                down(self.key(modifier + '_l')); down(self.key('a'))
                self.assertEqual(self.events(manager)[-2:], [
                    ('down', f'a+{modifier}'), ('down', f'a+{modifier}_l')])

    def test_x11_raw_adapter_tracks_hardware_across_shift_and_layout(self):
        class RawListener:
            def __init__(self, **kwargs):
                self.on_press = kwargs['on_press']
                self.on_release = kwargs['on_release']
            def _handle_message(self, display, event, injected):
                callback = self.on_press if event.type == 2 else self.on_release
                return callback(event.key, injected)

        keyboard = SimpleNamespace(
            Listener=RawListener, Key=(), KeyCode=SimpleNamespace(
                from_char=lambda char, vk: SimpleNamespace(char=char, vk=vk),
                from_vk=lambda vk: SimpleNamespace(vk=vk),
            ),
        )
        with loaded_shortcut_manager() as module:
            # French-style top row: Shift+é emits 2. A US punctuation fold
            # cannot recover the base name; the raw adapter consults the layout.
            for base, shifted, code in (('a', 'A', 38), ('é', '2', 11), ('ß', '?', 20)):
                manager = self.manager(module, f'{base}+shift')
                down, up = manager.create_portable_keyboard_callbacks()
                listener = PLATFORM_SUPPORT.create_x11_keyboard_listener(keyboard, down, up)
                display = SimpleNamespace(keycode_to_keysym=Mock(
                    side_effect=lambda detail, index: 0xFFE1 if detail == 50 else ord(base)
                ))
                def send(detail, key, pressed=True):
                    listener._handle_message(display, SimpleNamespace(
                        type=2 if pressed else 3, detail=detail, state=1 << 13, key=key,
                    ), False)
                for _ in range(2):
                    send(50, self.key('shift'))
                    send(code, SimpleNamespace(char=shifted, vk=ord(shifted)))
                    send(50, self.key('shift'), False)
                    send(code, SimpleNamespace(char=base, vk=ord(base)), False)
                    self.assertEqual(manager._physical_keys, {})
                binding = normalize_shortcut(f'{base}+shift')
                self.assertEqual(self.events(manager), [('down', binding), ('up', binding)] * 2)
                display.keycode_to_keysym.assert_any_call(code, 2)
                # An emulated Controller callback must not inherit stale raw
                # context after _handle_message returns.
                listener.on_press(self.key('f1'), True)
                self.assertEqual(manager._physical_keys, {})

    def test_generic_alt_and_shift_accept_either_side(self):
        with loaded_shortcut_manager() as module:
            for alt, shift in (('alt_l', 'shift'), ('alt_r', 'shift_r'), ('alt_gr', 'shift_r')):
                manager = self.manager(module, 'alt+shift+f24')
                down, up = manager.create_portable_keyboard_callbacks()
                for name in (alt, shift, 'f24'):
                    down(self.key(name))
                self.assertEqual(self.events(manager), [('down', 'alt+f24+shift')])
                up(self.key(shift))
                self.assertEqual(self.events(manager)[-1], ('up', 'alt+f24+shift'))

    def test_injected_emulated_and_restoring_keys_do_not_complete_chord(self):
        with loaded_shortcut_manager() as module:
            manager = self.manager(module, 'a+f13')
            down, up = manager.create_portable_keyboard_callbacks()
            down(self.key('a'))
            down(self.key('f13'), injected=True)
            manager._emulator.emulating.add('f13')
            down(self.key('f13')); up(self.key('f13'))
            self.assertEqual(manager._emulator.emulating, set())
            manager._restoring_keys.add('f13')
            down(self.key('f13')); up(self.key('f13'))
            self.assertEqual(self.events(manager), [])
            down(self.key('f13'))
            self.assertEqual(self.events(manager), [('down', 'a+f13')])

    def test_vk_identity_survives_character_changes(self):
        with loaded_shortcut_manager() as module:
            manager = self.manager(module, 'a+vk_222')
            down, up = manager.create_portable_keyboard_callbacks()
            down(SimpleNamespace(char='A', vk=65))
            down(SimpleNamespace(char="'", vk=222))
            up(SimpleNamespace(char='"', vk=222))
            self.assertEqual(self.events(manager), [('down', 'a+vk_222'), ('up', 'a+vk_222')])

    def test_win32_filter_repeat_suppression_and_injection(self):
        with loaded_shortcut_manager() as module:
            manager = self.manager(module, 'caps_lock', 'f13+f14')
            manager.tasks['caps_lock'].shortcut.suppress = True
            with patch.object(module.KeyMapper, 'vk_to_name', create=True,
                              side_effect=lambda vk: {20: 'caps_lock', 124: 'f13', 125: 'f14'}[vk]):
                hook = manager.create_keyboard_filter()
                def send(vk, down=True, flags=0):
                    return hook(1 if down else 2, SimpleNamespace(vkCode=vk, flags=flags))
                for pressed in (True, True, False):
                    self.assertTrue(send(20, pressed))
                self.assertEqual(manager.keyboard_listener.suppress_event.call_count, 3)
                send(124, flags=0x10); send(125)
                self.assertNotIn(('down', 'f13+f14'), self.events(manager))
                manager._emulator.emulating.add('f13')
                send(124); send(124, False, 0x10)
                self.assertEqual(manager._emulator.emulating, set())
                send(124); send(124)
                send(125, False); send(125)
                self.assertEqual(self.events(manager), [
                    ('down', 'caps_lock'), ('up', 'caps_lock'),
                    ('down', 'f13+f14'), ('up', 'f13+f14')])
                self.assertEqual(manager.keyboard_listener.suppress_event.call_count, 3)

    def test_chord_suppress_disabled_without_mutating_config(self):
        with loaded_shortcut_manager() as module:
            shortcut = FakeShortcut('f1+f2', 'keyboard', suppress=True)
            manager = self.manager(module)
            manager.app = object()
            manager.shortcuts = [shortcut]
            manager._pool = object()
            manager._hotkey_support = PLATFORM_SUPPORT.detect_hotkey_backend('Windows', {})
            with patch.object(module.logger, 'warning', create=True) as warning:
                manager._init_tasks()
                warning.assert_called_once()
            self.assertTrue(shortcut.suppress)
            self.assertFalse(manager.tasks['f1+f2'].shortcut.suppress)

    def test_generic_modifier_replay_is_not_captured(self):
        with loaded_shortcut_manager() as module:
            for generic, physical in (('shift', 'shift'), ('ctrl', 'ctrl_l'), ('alt', 'alt_l')):
                manager = self.manager(module, generic)
                down, up = manager.create_portable_keyboard_callbacks()
                manager._emulator.emulating.add(generic)
                down(self.key(physical)); up(self.key(physical))
                self.assertEqual(self.events(manager), [])
                self.assertEqual(manager._emulator.emulating, set())

    def test_short_tap_chord_never_replays_while_singleton_still_does(self):
        with loaded_shortcut_manager() as module:
            handler_module = load_module(
                'core.client.shortcut._combination_event_handler_test',
                ROOT / 'core/client/shortcut/event_handler.py',
            )
            try:
                for binding in ('f1+f2', 'f1'):
                    manager = self.manager(module, binding)
                    task = manager.tasks[binding]
                    task.shortcut.suppress = binding == 'f1'
                    task.is_recording = False
                    task.threshold = 0.3
                    task.cancel = Mock()
                    def launch():
                        task.is_recording = True
                        task.recording_start_time = 10.0
                    task.launch = Mock(side_effect=launch)
                    pool = Mock()
                    manager._emulator.emulate_key = Mock()
                    manager._event_handler = handler_module.ShortcutEventHandler(
                        manager.tasks, pool, manager._emulator,
                    )
                    down, up = manager.create_portable_keyboard_callbacks()
                    down(self.key('f1'))
                    if '+' in binding:
                        down(self.key('f2'))
                    with patch.object(handler_module.time, 'time', return_value=10.1):
                        up(self.key('f1'))
                    task.launch.assert_called_once()
                    task.cancel.assert_called_once()
                    if '+' in binding:
                        pool.submit.assert_not_called()
                    else:
                        pool.submit.assert_called_once_with(manager._emulator.emulate_key, 'f1')
            finally:
                import sys
                sys.modules.pop(handler_module.__name__, None)

    def test_click_mode_gets_one_press_and_first_release_signal(self):
        with loaded_shortcut_manager() as module:
            handler_module = load_module(
                'core.client.shortcut._combination_click_handler_test',
                ROOT / 'core/client/shortcut/event_handler.py',
            )
            try:
                manager = self.manager(module, 'f1+f2')
                task = manager.tasks['f1+f2']
                task.shortcut.hold_mode = False
                task.released = True
                task.pressed = False
                task.event = Event()
                pool = Mock()
                manager._event_handler = handler_module.ShortcutEventHandler(
                    manager.tasks, pool, manager._emulator,
                )
                down, up = manager.create_portable_keyboard_callbacks()
                down(self.key('f1')); down(self.key('f2')); down(self.key('f2'))
                self.assertEqual(pool.submit.call_count, 2)
                self.assertTrue(task.pressed)
                up(self.key('f1'))
                self.assertTrue(task.released)
                self.assertTrue(task.event.is_set())
                down(self.key('f1'))
                self.assertEqual(pool.submit.call_count, 2)
                up(self.key('f1')); up(self.key('f2'))
                down(self.key('f2')); down(self.key('f1'))
                self.assertEqual(pool.submit.call_count, 4)
            finally:
                import sys
                sys.modules.pop(handler_module.__name__, None)

    def test_win32_arbitrary_chord_all_release_orders_and_raw_vk(self):
        with loaded_shortcut_manager() as module:
            for release in permutations((0xC0, 49, 124)):
                manager = self.manager(module, '`+1+vk_124')
                with patch.object(module.KeyMapper, 'vk_to_name', create=True,
                                  side_effect=lambda vk: {0xC0: '`', 49: '1', 124: 'f13'}[vk]):
                    hook = manager.create_keyboard_filter()
                    for vk in (124, 49, 0xC0, 0xC0):
                        hook(1, SimpleNamespace(vkCode=vk, flags=0))
                    for vk in release:
                        hook(2, SimpleNamespace(vkCode=vk, flags=0))
                self.assertEqual(self.events(manager), [
                    ('down', '1+`+vk_124'), ('up', '1+`+vk_124')])
                manager.keyboard_listener.suppress_event.assert_not_called()

    def test_plus_and_equal_events_share_physical_identity(self):
        with loaded_shortcut_manager() as module:
            manager = self.manager(module, 'plus+f14')
            down, up = manager.create_portable_keyboard_callbacks()
            down(SimpleNamespace(char='=', vk=187))
            down(self.key('f14'))
            up(SimpleNamespace(char='+', vk=187))
            self.assertEqual(self.events(manager), [('down', 'f14+plus'), ('up', 'f14+plus')])

    def test_x11_keysyms_change_when_shift_is_released_first(self):
        with loaded_shortcut_manager() as module, patch.object(
            PLATFORM_SUPPORT.platform, 'system', return_value='Linux'
        ):
            manager = self.manager(module, 'a+shift', 'shift+1')
            down, up = manager.create_portable_keyboard_callbacks()
            down(self.key('shift'))
            down(SimpleNamespace(char='A', vk=65))
            down(SimpleNamespace(char='!', vk=33))
            up(self.key('shift'))
            up(SimpleNamespace(char='a', vk=97))
            up(SimpleNamespace(char='1', vk=49))
            self.assertEqual(self.events(manager), [
                ('down', 'a+shift'), ('down', '1+shift'),
                ('up', 'a+shift'), ('up', '1+shift')])
            self.assertEqual(manager._physical_keys, {})
            down(self.key('shift'))
            down(SimpleNamespace(char='A', vk=65))
            self.assertEqual(self.events(manager)[-1], ('down', 'a+shift'))

    def test_replayed_raw_vk_is_cleared_on_injected_release(self):
        with loaded_shortcut_manager() as module:
            manager = self.manager(module, 'vk_65')
            down, up = manager.create_portable_keyboard_callbacks()
            manager._emulator.emulating.add('vk_65')
            down(SimpleNamespace(char='a', vk=65))
            up(SimpleNamespace(char='a', vk=65), injected=True)
            self.assertEqual(manager._emulator.emulating, set())
            self.assertEqual(self.events(manager), [])


class MapperTest(unittest.TestCase):
    def test_unshifted_win32_scan_name_uses_cached_layout(self):
        with loaded_shortcut_manager():
            name = 'core.client.shortcut._combination_scan_mapper_test'
            module = load_module(name, ROOT / 'core/client/shortcut/key_mapper.py')
            try:
                translator = SimpleNamespace(char_from_scan=Mock(return_value='é'))
                with patch.object(module, '_get_key_translator', return_value=translator):
                    self.assertEqual(module.KeyMapper.scan_to_name(3), 'é')
                    translator.char_from_scan.assert_called_once_with(3)
            finally:
                import sys
                sys.modules.pop(name, None)
    def test_extended_function_keys_keypad_and_raw_vk(self):
        with loaded_shortcut_manager():
            name = 'core.client.shortcut._combination_mapper_test'
            module = load_module(name, ROOT / 'core/client/shortcut/key_mapper.py')
            try:
                for number in range(1, 25):
                    self.assertEqual(module.KeyMapper.vk_to_name(0x6F + number), f'f{number}')
                with patch.object(module, '_get_special_keys', return_value={}), patch.object(
                    module, '_get_key_translator', return_value=None
                ):
                    self.assertEqual(module.KeyMapper.vk_to_name(0x60), 'numpad0')
                    self.assertEqual(module.KeyMapper.vk_to_name(0x6B), 'numpad_add')
                    self.assertEqual(module.KeyMapper.vk_to_name(65535), 'vk_65535')
                    self.assertEqual(module.KeyMapper.vk_to_name(0xA1), 'shift_r')
            finally:
                import sys
                sys.modules.pop(name, None)

    def test_x11_keypad_and_f13_f14_without_enum_names(self):
        with patch.object(PLATFORM_SUPPORT.platform, 'system', return_value='Linux'):
            for vk, char, expected in (
                (0xFFB0, '0', 'numpad0'), (0xFFAB, '+', 'numpad_add'),
                (0xFFCA, None, 'f13'), (0xFFCB, None, 'f14'),
            ):
                self.assertEqual(PLATFORM_SUPPORT.pynput_key_name(
                    SimpleNamespace(vk=vk, char=char)), expected)

    def test_win32_oem_names_follow_layout_translator(self):
        with loaded_shortcut_manager():
            name = 'core.client.shortcut._layout_mapper_test'
            module = load_module(name, ROOT / 'core/client/shortcut/key_mapper.py')
            try:
                with patch.object(module, '_get_special_keys', return_value={}), patch.object(
                    module, '_get_key_translator', return_value=lambda vk, is_press: {'char': 'ö'}
                ):
                    # VK_OEM_1 is ';' only on a US layout.
                    self.assertEqual(module.KeyMapper.vk_to_name(0xBA), 'ö')
            finally:
                import sys
                sys.modules.pop(name, None)

    def test_replay_name_mapping_uses_platform_vk_codes(self):
        with loaded_shortcut_manager():
            name = 'core.client.shortcut._combination_replay_mapper_test'
            module = load_module(name, ROOT / 'core/client/shortcut/key_mapper.py')
            keyboard = SimpleNamespace(KeyCode=SimpleNamespace(
                from_vk=Mock(side_effect=lambda vk: ('vk', vk)),
                from_char=Mock(side_effect=lambda char: ('char', char)),
            ))
            try:
                with patch.object(module, '_get_keyboard_module', return_value=keyboard), patch.object(
                    module.KeyMapper, '_get_special_key_objects', return_value={}
                ):
                    with patch.object(module, 'system', return_value='Windows'):
                        for key, vk in (('f13', 124), ('f24', 135), ('numpad0', 96), ('numpad_add', 107), ('vk_65535', 65535)):
                            self.assertEqual(module.KeyMapper.name_to_key(key), ('vk', vk))
                    with patch.object(module, 'system', return_value='Linux'):
                        for key, vk in (('numpad0', 0xFFB0), ('numpad_add', 0xFFAB), ('f13', 0xFFCA), ('f24', 0xFFD5)):
                            self.assertEqual(module.KeyMapper.name_to_key(key), ('vk', vk))
                    self.assertEqual(module.KeyMapper.name_to_key('plus'), ('char', '+'))
            finally:
                import sys
                sys.modules.pop(name, None)

    def test_shortcut_config_uses_shared_parser_and_restores_only_single_toggle(self):
        with loaded_shortcut_manager():
            module = load_module('capswriter_combination_config_test', ROOT / 'core/client/shortcut/shortcut_config.py')
            try:
                self.assertEqual(module.Shortcut('F2+F1').key, 'f1+f2')
                with self.assertRaises(ValueError):
                    module.Shortcut('ctrl+control')
                # No restore of a whole chord, even when it contains CapsLock.
                self.assertEqual(module.Shortcut('capslock+f1').key, 'caps_lock+f1')
                import sys
                with patch.object(sys.modules['core.client.shortcut.key_mapper'],
                                  'RESTORABLE_KEYS', {'caps_lock', 'num_lock', 'scroll_lock'}, create=True):
                    self.assertTrue(module.Shortcut('capslock').is_toggle_key())
                    self.assertFalse(module.Shortcut('capslock+f1').is_toggle_key())
            finally:
                import sys
                sys.modules.pop(module.__name__, None)


if __name__ == '__main__':
    unittest.main()
