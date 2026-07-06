# coding: utf-8
"""Privacy-preserving logging helpers for HTTP API text payloads."""

from __future__ import annotations

from typing import Any


def single_line_text(text: str) -> str:
    return text.replace("\n", " ").replace("\r", "")


def log_prompt_context(
    logger: Any,
    task_id: str,
    context: str,
    *,
    log_sensitive_text: bool,
) -> None:
    if not context:
        return
    if log_sensitive_text:
        logger.debug(f"[HTTP] task={task_id[:8]} context={context[:50]!r}")
        return
    logger.debug(
        f"[HTTP] task={task_id[:8]} context_chars={len(context)} "
        "context=<redacted>"
    )


def log_transcription_result(
    logger: Any,
    console: Any,
    task_id: str,
    delay: float,
    text: str,
    *,
    log_sensitive_text: bool,
) -> None:
    if log_sensitive_text:
        logger.info(
            f"[HTTP] task={task_id[:8]} done, delay={delay:.1f}s, "
            f"text={single_line_text(text)}"
        )
        console.print(f"    转录时延：{delay:.2f}s")
        console.print("    识别结果：", end="")
        console.print(text, style="green")
        console.line()
        return
    logger.info(
        f"[HTTP] task={task_id[:8]} done, delay={delay:.1f}s, "
        f"text_chars={len(text)} text=<redacted>"
    )
    console.print(f"    转录时延：{delay:.2f}s")
    console.print("    识别结果：[redacted]")
    console.line()
