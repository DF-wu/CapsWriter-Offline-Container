"""Redirected legacy Windows output must not interrupt application work."""
import os
import subprocess
import sys
import unittest


class StdioEncodingTest(unittest.TestCase):
    def test_plain_and_rich_output_survive_cp1252_redirection(self):
        script = """
import sys
import core
from rich.console import Console
assert sys.stdout.encoding == 'cp1252'
print('\u5ba2\u6237\u7aef\u5df2\u9023\u63a5')
Console().print('\u97f3\u8a0a\u5b8c\u6210')
print('\u932f\u8aa4\u8a0a\u606f', file=sys.stderr)
print('handler continued')
"""
        result = subprocess.run(
            [sys.executable, '-c', script],
            env=dict(os.environ, PYTHONIOENCODING='cp1252:strict'),
            capture_output=True, check=True,
        )
        self.assertIn(b'\\u5ba2', result.stdout)
        self.assertIn(b'\\u97f3', result.stdout)
        self.assertIn(b'\\u932f', result.stderr)
        self.assertIn(b'handler continued', result.stdout)

    def test_gui_startup_allows_absent_standard_streams(self):
        subprocess.run(
            [sys.executable, '-c', 'import sys; sys.stdout = sys.stderr = None; import core'],
            capture_output=True, check=True,
        )
