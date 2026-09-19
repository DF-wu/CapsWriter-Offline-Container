# coding: utf-8
"""Per-task privacy context shared by the synchronous recognition pipeline."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator


_TRANSCRIPT_LOGGING_ENABLED: ContextVar[bool] = ContextVar(
    "capswriter_transcript_logging_enabled",
    default=True,
)


def transcript_logging_enabled() -> bool:
    """Return whether the current task may emit prompt/transcript text."""
    return bool(_TRANSCRIPT_LOGGING_ENABLED.get())


@contextmanager
def transcript_logging(enabled: bool) -> Iterator[None]:
    """Apply a task-local logging policy and restore it on every exit."""
    token = _TRANSCRIPT_LOGGING_ENABLED.set(bool(enabled))
    try:
        yield
    finally:
        _TRANSCRIPT_LOGGING_ENABLED.reset(token)
