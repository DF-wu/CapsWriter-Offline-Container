# coding: utf-8

import sys

# Redirected Windows streams may use cp1252 instead of the Unicode console.
# Preserve the selected encoding, but never abort recognition just because a
# status message contains a character it cannot represent. Configure the real
# streams before colorama wraps them; frozen GUI builds may have no streams.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(errors='backslashreplace')
        except (AttributeError, OSError, ValueError):
            pass  # Closed or embedding-owned streams cannot be reconfigured.

from .logger import get_logger, setup_logger

import colorama
colorama.init()
