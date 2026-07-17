#!/usr/bin/env python3
# coding: utf-8
"""Zipapp/module entrypoint for the no-GUI CLI."""

from __future__ import annotations

try:
    from .capswriter_cli import main
except ImportError:  # zipapp built from this directory has no package parent.
    from capswriter_cli import main  # type: ignore[no-redef]


if __name__ == "__main__":
    raise SystemExit(main())
