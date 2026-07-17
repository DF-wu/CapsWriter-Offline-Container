# coding: utf-8
"""
OpenAI ASR 回應格式化器

把 CapsWriter Result 轉為 OpenAI Whisper API 規範的五種 response_format:
json / text / verbose_json / srt / vtt。
"""

from __future__ import annotations
import re
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.server.schema import Result


# 句末標點 (中/英)。匹配到時切出一個 segment。
_SENT_END = re.compile(r"[。！？!?]+|[.](?:\s|$)|[，、;；,]\s")
_TIMESTAMP_GRANULARITIES = frozenset({"segment", "word"})
UNDETECTED_LANGUAGE_SENTINEL = "auto"


def _safe_timestamp(value: float) -> float:
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(timestamp):
        return 0.0
    return max(0.0, timestamp)


def _monotonic_timestamps(values: Sequence[float]) -> List[float]:
    """Return finite, nonnegative timestamps without backward movement."""
    normalized: List[float] = []
    previous = 0.0
    for raw in values:
        previous = max(previous, _safe_timestamp(raw))
        normalized.append(previous)
    return normalized


def _segment_payload(
    *,
    segment_id: int,
    start: float,
    end: float,
    text: str,
    temperature: float,
) -> Dict[str, Any]:
    """Return the required ``whisper-1`` verbose segment fields.

    CapsWriter engines do not expose Whisper token ids or probability metrics.
    The shape remains SDK-compatible while unavailable metrics use neutral
    numeric values rather than inventing backend confidence data.
    """
    safe_start = _safe_timestamp(start)
    safe_end = max(safe_start, _safe_timestamp(end))
    return {
        "id": segment_id,
        "seek": 0,
        "start": safe_start,
        "end": safe_end,
        "text": text,
        "tokens": [],
        "temperature": float(temperature),
        "avg_logprob": 0.0,
        "compression_ratio": 0.0,
        "no_speech_prob": 0.0,
    }


def _segments_from_tokens(
    tokens: List[str],
    timestamps: List[float],
    total_duration: float,
    temperature: float = 0.0,
) -> List[Dict[str, Any]]:
    """依句末標點切分 segments; 若資訊不足則返回空 list。"""
    if not tokens or len(tokens) != len(timestamps):
        return []

    normalized_timestamps = _monotonic_timestamps(timestamps)
    segments: List[Dict[str, Any]] = []
    seg_start_idx = 0
    text_buf = ""

    for i, tok in enumerate(tokens):
        text_buf += tok
        is_last = i == len(tokens) - 1
        is_break = bool(_SENT_END.search(tok)) or is_last
        if not is_break:
            continue

        start = normalized_timestamps[seg_start_idx]
        if is_last:
            end = max(normalized_timestamps[i], _safe_timestamp(total_duration))
        else:
            end = (
                normalized_timestamps[i + 1]
                if i + 1 < len(normalized_timestamps)
                else normalized_timestamps[i]
            )

        text = text_buf.strip()
        if text:
            segments.append(
                _segment_payload(
                    segment_id=len(segments),
                    start=start,
                    end=end,
                    text=text,
                    temperature=temperature,
                )
            )
        seg_start_idx = i + 1
        text_buf = ""

    return segments


def _words_from_tokens(
    tokens: List[str],
    timestamps: List[float],
    total_duration: float,
) -> List[Dict[str, Any]]:
    if not tokens or len(tokens) != len(timestamps):
        return []
    normalized_timestamps = _monotonic_timestamps(timestamps)
    out: List[Dict[str, Any]] = []
    for i, (tok, start) in enumerate(zip(tokens, normalized_timestamps)):
        if i + 1 < len(normalized_timestamps):
            end = normalized_timestamps[i + 1]
        else:
            end = max(start, _safe_timestamp(total_duration))
        out.append({"word": tok, "start": start, "end": end})
    return out


def _split_timestamp(seconds: float) -> Tuple[int, int, int, int]:
    total_ms = int(round(max(0.0, seconds) * 1000))
    h, remainder = divmod(total_ms, 60 * 60 * 1000)
    m, remainder = divmod(remainder, 60 * 1000)
    s, ms = divmod(remainder, 1000)
    return h, m, s, ms


def _fmt_srt_ts(seconds: float) -> str:
    h, m, s, ms = _split_timestamp(seconds)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_vtt_ts(seconds: float) -> str:
    h, m, s, ms = _split_timestamp(seconds)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _final_text(result: Result) -> str:
    return result.text_accu or result.text or ""


def _normalize_timestamp_granularities(
    values: Optional[Sequence[str]],
) -> tuple[str, ...]:
    if values is None:
        return ("segment",)
    normalized: list[str] = []
    for raw in values:
        value = str(raw).strip().lower()
        if value not in _TIMESTAMP_GRANULARITIES:
            raise ValueError(f"unsupported timestamp granularity: {value!r}")
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def format_response(
    result: Result,
    response_format: str,
    language: Optional[str] = None,
    *,
    temperature: float = 0.0,
    timestamp_granularities: Optional[Sequence[str]] = None,
) -> Tuple[Any, str]:
    """
    Returns (body, media_type):
        body  — str (text/srt/vtt) 或 dict (json/verbose_json)
        media_type — 對應 Content-Type
    """
    text = _final_text(result)
    fmt = response_format.lower()

    if fmt == "text":
        return text, "text/plain; charset=utf-8"

    if fmt in ("srt", "vtt"):
        segments = _segments_from_tokens(
            result.tokens,
            result.timestamps,
            result.duration,
            temperature,
        )
        if not segments:
            segments = [
                _segment_payload(
                    segment_id=0,
                    start=0.0,
                    end=float(result.duration or 0.0),
                    text=text,
                    temperature=temperature,
                )
            ]

        lines: List[str] = []
        if fmt == "vtt":
            lines.extend(["WEBVTT", ""])
            for seg in segments:
                lines.append(f"{_fmt_vtt_ts(seg['start'])} --> {_fmt_vtt_ts(seg['end'])}")
                lines.append(seg["text"])
                lines.append("")
            return "\n".join(lines), "text/vtt; charset=utf-8"

        # srt
        for i, seg in enumerate(segments, 1):
            lines.append(str(i))
            lines.append(f"{_fmt_srt_ts(seg['start'])} --> {_fmt_srt_ts(seg['end'])}")
            lines.append(seg["text"])
            lines.append("")
        return "\n".join(lines), "application/x-subrip; charset=utf-8"

    if fmt == "verbose_json":
        granularities = _normalize_timestamp_granularities(
            timestamp_granularities
        )
        body = {
            "task": "transcribe",
            # Result currently has no backend-agnostic, reliable detected-
            # language field. Preserve an explicit hint; otherwise expose the
            # documented local sentinel instead of fabricating a detection.
            "language": language or UNDETECTED_LANGUAGE_SENTINEL,
            "duration": _safe_timestamp(result.duration or 0.0),
            "text": text,
        }
        if "segment" in granularities:
            segments = _segments_from_tokens(
                result.tokens,
                result.timestamps,
                result.duration,
                temperature,
            )
            if not segments and text:
                segments = [
                    _segment_payload(
                        segment_id=0,
                        start=0.0,
                        end=float(result.duration or 0.0),
                        text=text,
                        temperature=temperature,
                    )
                ]
            body["segments"] = segments
        if "word" in granularities:
            body["words"] = _words_from_tokens(
                result.tokens,
                result.timestamps,
                result.duration,
            )
        return body, "application/json"

    # default: json
    return {"text": text}, "application/json"
