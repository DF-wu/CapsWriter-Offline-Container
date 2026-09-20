"""Optional error prompts for desktop launches and unattended services."""

import os
import sys


def pause_on_error(prompt: str) -> None:
    """Wait only when error prompts are enabled and stdin is a terminal."""
    if os.environ.get("CAPSWRITER_INTERACTIVE_ERRORS", "").strip().lower() in {
        "0", "false", "no", "off",
    }:
        return
    try:
        if sys.stdin is not None and sys.stdin.isatty():
            input(prompt)
    except (EOFError, OSError, ValueError):
        # A terminal can close between the TTY check and the actual read.
        pass
