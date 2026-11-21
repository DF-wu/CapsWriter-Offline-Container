"""Exercise desktop hotword code without importing GUI or model dependencies."""
import ast
import logging
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unicodedata
import unittest
from unittest.mock import Mock, patch

import user_config

ROOT = Path(__file__).resolve().parents[2]


def definitions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    tree.body = [node for node in tree.body if getattr(node, 'name', None) in names]
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace


class HotwordUserSaveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.shipped = self.root / 'hot.txt'
        self.shipped.write_text('default', encoding='utf-8')
        self.target = self.root / 'user' / 'hot.txt'
        for name, value in [('program_root', self.root), ('user_dir', self.target.parent)]:
            patcher = patch.object(user_config, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.ns = definitions('core/ui/hotword_menu_handler.py', {'_save_hotwords'},
                              {'logger': logging.getLogger(__name__)})

    def save(self):
        self.ns['_save_hotwords'](' new \n second ')

    def test_missing_user_file_preserves_defaults(self):
        self.save()
        self.assertEqual(self.target.read_text(encoding='utf-8'), 'default\nnew\nsecond\n')
        self.assertEqual(self.shipped.read_text(encoding='utf-8'), 'default')

    def test_existing_custom_content_is_not_reseeded(self):
        self.target.parent.mkdir()
        self.target.write_text('custom\n')
        self.save()
        self.assertEqual(self.target.read_text(encoding='utf-8'), 'custom\nnew\nsecond\n')
        self.assertEqual(self.shipped.read_text(encoding='utf-8'), 'default')

    def test_creation_error_propagates_without_modifying_defaults(self):
        self.target.parent.write_text('blocked')
        with self.assertRaises(OSError):
            self.save()
        self.assertEqual(self.shipped.read_text(encoding='utf-8'), 'default')

    def test_append_error_propagates_without_modifying_defaults(self):
        self.target.parent.mkdir()
        self.target.write_text('custom')
        with patch('builtins.open', side_effect=PermissionError('synthetic denial')):
            with self.assertRaises(PermissionError):
                self.save()
        self.assertEqual(self.target.read_text(encoding='utf-8'), 'custom')
        self.assertEqual(self.shipped.read_text(encoding='utf-8'), 'default')

    def open_editor(self, opener):
        tree = ast.parse((ROOT / 'core/client/manager/tray_manager.py').read_text(encoding='utf-8'))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and
                   any(getattr(child, 'name', '') == '_add_hotword' for child in node.body))
        method = next(node for node in cls.body if getattr(node, 'name', '') == '_add_hotword')
        namespace = {'logger': logging.getLogger(__name__), '_open_with_default_app': opener}
        exec(compile(ast.Module(body=[method], type_ignores=[]), 'tray_manager.py', 'exec'), namespace)
        namespace['_add_hotword'](None)

    def test_editor_opens_initialized_persistent_path(self):
        opener = Mock()
        self.open_editor(opener)
        opener.assert_called_once_with(str(self.target))
        self.assertEqual(self.target.read_text(encoding='utf-8'), 'default')
        self.assertEqual(self.shipped.read_text(encoding='utf-8'), 'default')

    def test_editor_preserves_existing_custom_content(self):
        self.target.parent.mkdir()
        self.target.write_text('custom')
        opener = Mock()
        self.open_editor(opener)
        opener.assert_called_once_with(str(self.target))
        self.assertEqual(self.target.read_text(encoding='utf-8'), 'custom')

    def test_editor_does_not_launch_when_directory_is_blocked(self):
        self.target.parent.write_text('blocked')
        opener = Mock()
        with self.assertLogs(level='WARNING'):
            self.open_editor(opener)
        opener.assert_not_called()
        self.assertEqual(self.shipped.read_text(encoding='utf-8'), 'default')

    def make_manager(self, files=None, use_user_files=True):
        ns = definitions('core/client/hotword/manager.py',
                         {'HotwordManager', '_HotwordFileHandler'}, {
            'Optional': __import__('typing').Optional, 'Dict': dict, 'Path': Path,
            'Any': object, 'PhonemeCorrector': object, 'RuleCorrector': object,
            'FileSystemEventHandler': object, 'threading': threading, 'time': time,
            'logger': logging.getLogger(__name__), 'unicodedata': unicodedata,
            'console': SimpleNamespace(print=lambda *a, **k: None)})
        manager = ns['HotwordManager'].__new__(ns['HotwordManager'])
        manager._use_user_files = use_user_files
        manager.files = files or {'hot': self.shipped, 'rule': self.root / 'hot-rule.txt'}
        loaded = []
        manager.phoneme_corrector = SimpleNamespace(update_hotwords=lambda text: loaded.append(text) or 1)
        manager._observer = None
        manager._is_watcher_started = False
        scheduled = []
        class Observer:
            def schedule(self, handler, path, recursive):
                scheduled.append((handler, Path(path), recursive))
            def start(self): pass
        ns['Observer'] = Observer
        manager.start_file_watcher()
        if use_user_files:
            self.assertTrue(any(path == self.target.parent and not recursive for _, path, recursive in scheduled))
        return manager, loaded, scheduled

    def test_effective_reload_after_first_save_and_removed_user_file(self):
        manager, loaded, scheduled = self.make_manager()
        for removed in (False, True):
            if removed:
                self.target.unlink()
            self.save()
            handler = scheduled[-1][0]
            handler._debounce_delay = 0
            event = SimpleNamespace(is_directory=False, src_path=str(self.target))
            handler.on_created(event)
            handler._timer.join(timeout=2)
            self.assertEqual(loaded[-1], 'default\nnew\nsecond\n')
            self.target.write_text('editor replacement\n')
            handler.on_moved(SimpleNamespace(is_directory=False,
                             src_path=str(self.target.with_suffix('.tmp')), dest_path=str(self.target)))
            handler._timer.join(timeout=2)
            self.assertEqual(loaded[-1], 'editor replacement\n')
            self.assertEqual(manager.files['hot'], self.target)

    def test_existing_overrides_delete_and_root_edit_reload(self):
        self.target.parent.mkdir()
        self.target.write_text('custom hot')
        rule = self.target.parent / 'hot-rule.txt'
        rule.write_text('custom rule')
        (self.root / 'hot-rule.txt').write_text('default rule')
        manager, loaded, scheduled = self.make_manager({'hot': self.target, 'rule': rule})
        self.assertEqual({path for _, path, _ in scheduled}, {self.root, self.target.parent})
        handler = scheduled[0][0]
        handler._debounce_delay = 0
        self.target.unlink()
        handler.on_deleted(SimpleNamespace(is_directory=False, src_path=str(self.target)))
        handler._timer.join(timeout=2)
        self.assertEqual(loaded[-1], 'default')
        self.assertEqual(manager.files['hot'], self.shipped)
        self.shipped.write_text('updated default')
        handler.on_modified(SimpleNamespace(is_directory=False, src_path=str(self.shipped)))
        handler._timer.join(timeout=2)
        self.assertEqual(loaded[-1], 'updated default')
        rules = []
        manager.rule_corrector = SimpleNamespace(update_rules=lambda text: rules.append(text) or 1)
        renamed = rule.with_suffix('.backup')
        rule.rename(renamed)
        handler.on_moved(SimpleNamespace(is_directory=False, src_path=str(rule), dest_path=str(renamed)))
        handler._timer.join(timeout=2)
        self.assertEqual(rules[-1], 'default rule')
        self.assertEqual(manager.files['rule'], self.root / 'hot-rule.txt')

    def test_explicit_mapping_does_not_gain_fallback_or_removal_handling(self):
        manager, loaded, scheduled = self.make_manager(use_user_files=False)
        handler = scheduled[0][0]
        handler._debounce_delay = 0
        handler.on_deleted(SimpleNamespace(is_directory=False, src_path=str(self.shipped)))
        handler.on_moved(SimpleNamespace(is_directory=False, src_path=str(self.shipped),
                                        dest_path=str(self.shipped.with_suffix('.backup'))))
        self.assertIsNone(handler._timer)
        self.assertEqual(loaded, [])
        self.assertEqual(manager.files['hot'], self.shipped)

    def test_move_between_hotword_files_reloads_both_in_one_batch(self):
        _, _, scheduled = self.make_manager()
        handler = scheduled[0][0]
        calls = []
        handler._file_mapping = {name: lambda name=name: calls.append(name)
                                 for name in ('hot.txt', 'hot-rule.txt')}
        # Hold the worker until both move endpoints have been queued.
        handler._worker_running = True
        handler.on_moved(SimpleNamespace(is_directory=False, src_path=str(self.target),
                         dest_path=str(self.target.parent / 'hot-rule.txt')))
        handler._debounce_delay = 0
        handler._debounced_worker()
        self.assertCountEqual(calls, ['hot.txt', 'hot-rule.txt'])

    def test_event_during_reload_is_not_lost(self):
        _, _, scheduled = self.make_manager()
        handler = scheduled[0][0]
        calls = []
        def reload_hot():
            calls.append('hot')
            handler.on_modified(SimpleNamespace(is_directory=False,
                                src_path=str(self.target.parent / 'hot-rule.txt')))
        handler._file_mapping = {'hot.txt': reload_hot,
                                 'hot-rule.txt': lambda: calls.append('rule')}
        handler._debounce_delay = 0
        handler.on_modified(SimpleNamespace(is_directory=False, src_path=str(self.target)))
        handler._timer.join(timeout=2)
        self.assertFalse(handler._timer.is_alive())
        self.assertEqual(calls, ['hot', 'rule'])
