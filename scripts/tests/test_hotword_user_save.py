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
from unittest.mock import patch

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
        self.assertEqual(self.target.read_text(), 'default\nnew\nsecond\n')
        self.assertEqual(self.shipped.read_text(), 'default')

    def test_existing_custom_content_is_not_reseeded(self):
        self.target.parent.mkdir()
        self.target.write_text('custom\n')
        self.save()
        self.assertEqual(self.target.read_text(), 'custom\nnew\nsecond\n')
        self.assertEqual(self.shipped.read_text(), 'default')

    def test_creation_error_propagates_without_modifying_defaults(self):
        self.target.parent.write_text('blocked')
        with self.assertRaises(OSError):
            self.save()
        self.assertEqual(self.shipped.read_text(), 'default')

    def test_append_error_propagates_without_modifying_defaults(self):
        self.target.parent.mkdir()
        self.target.write_text('custom')
        with patch('builtins.open', side_effect=PermissionError('synthetic denial')):
            with self.assertRaises(PermissionError):
                self.save()
        self.assertEqual(self.target.read_text(), 'custom')
        self.assertEqual(self.shipped.read_text(), 'default')

    def test_effective_reload_after_first_save_and_removed_user_file(self):
        ns = definitions('core/client/hotword/manager.py',
                         {'HotwordManager', '_HotwordFileHandler'}, {
            'Optional': __import__('typing').Optional, 'Dict': dict, 'Path': Path,
            'Any': object, 'PhonemeCorrector': object, 'RuleCorrector': object,
            'FileSystemEventHandler': object, 'threading': threading, 'time': time,
            'logger': logging.getLogger(__name__), 'unicodedata': unicodedata,
            'console': SimpleNamespace(print=lambda *a, **k: None)})
        manager = ns['HotwordManager'].__new__(ns['HotwordManager'])
        manager._use_user_files = True
        manager.files = {'hot': self.shipped, 'rule': self.root / 'hot-rule.txt'}
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
        self.assertTrue(any(path == self.target.parent and not recursive for _, path, recursive in scheduled))
        for removed in (False, True):
            if removed:
                self.target.unlink()
            self.save()
            handler = scheduled[-1][0]
            handler._debounce_delay = 0
            handler.on_modified(SimpleNamespace(is_directory=False, src_path=str(self.target)))
            handler._timer.join(timeout=2)
            self.assertEqual(loaded[-1], 'default\nnew\nsecond\n')
            self.assertEqual(manager.files['hot'], self.target)
