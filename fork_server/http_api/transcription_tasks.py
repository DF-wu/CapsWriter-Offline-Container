# coding: utf-8
"""Dependency-light helpers for planning HTTP transcription work items."""

from __future__ import annotations

from typing import Iterator, Optional, TypedDict


PCM_SAMPLE_RATE = 16000
PCM_BYTES_PER_SAMPLE = 4
PCM_CHANNELS = 1
PCM_BYTES_PER_SECOND = PCM_SAMPLE_RATE * PCM_BYTES_PER_SAMPLE * PCM_CHANNELS

DEFAULT_SEG_DURATION = 60.0
DEFAULT_SEG_OVERLAP = 4.0
MAX_PROMPT_CONTEXT_CHARS = 2048

LANGUAGE_ALIASES = {
    "": "auto",
    "auto": "auto",
    "zh": "chinese",
    "zh-cn": "chinese",
    "zh-tw": "chinese",
    "cn": "chinese",
    "en": "english",
    "ja": "japanese",
    "jp": "japanese",
    "ko": "korean",
    "kr": "korean",
    "yue": "cantonese",
    "ar": "arabic",
    "de": "german",
    "fr": "french",
    "es": "spanish",
    "pt": "portuguese",
    "id": "indonesian",
    "it": "italian",
    "ru": "russian",
    "th": "thai",
    "vi": "vietnamese",
    "tr": "turkish",
    "hi": "hindi",
    "ms": "malay",
    "nl": "dutch",
    "sv": "swedish",
    "da": "danish",
    "fi": "finnish",
    "pl": "polish",
    "cs": "czech",
    "tl": "filipino",
    "fil": "filipino",
    "fa": "persian",
    "el": "greek",
    "ro": "romanian",
    "hu": "hungarian",
    "mk": "macedonian",
}


class TaskSpec(TypedDict):
    type: str
    data: bytes
    offset: float
    overlap: float
    task_id: str
    socket_id: str
    is_final: bool
    time_start: float
    context: str
    language: str


def seconds_to_bytes(seconds: float) -> int:
    return int(seconds * PCM_BYTES_PER_SECOND)


def normalize_prompt_context(prompt: Optional[str]) -> str:
    if not prompt:
        return ""
    context = prompt.replace("\r\n", "\n").replace("\r", "\n").strip()
    return context[:MAX_PROMPT_CONTEXT_CHARS]


def normalize_language_hint(language: Optional[str]) -> str:
    normalized = (language or "").strip().lower().replace("_", "-")
    return LANGUAGE_ALIASES.get(normalized, normalized or "auto")


def iter_transcription_task_specs(
    *,
    task_id: str,
    socket_id: str,
    pcm: bytes,
    time_start: float,
    context: str,
    language: str,
    seg_duration: float = DEFAULT_SEG_DURATION,
    seg_overlap: float = DEFAULT_SEG_OVERLAP,
) -> Iterator[TaskSpec]:
    segment_bytes = seconds_to_bytes(seg_duration + seg_overlap)
    stride_bytes = seconds_to_bytes(seg_duration)
    total_bytes = len(pcm)

    def spec(data: bytes, offset: float, is_final: bool) -> TaskSpec:
        return {
            "type": "file",
            "data": data,
            "offset": offset,
            "overlap": seg_overlap,
            "task_id": task_id,
            "socket_id": socket_id,
            "is_final": is_final,
            "time_start": time_start,
            "context": context,
            "language": language,
        }

    if total_bytes <= segment_bytes:
        yield spec(pcm, 0.0, True)
        return

    offset = 0.0
    pos = 0
    while pos + segment_bytes < total_bytes:
        yield spec(pcm[pos : pos + segment_bytes], offset, False)
        offset += seg_duration
        pos += stride_bytes

    yield spec(pcm[pos:], offset, True)
