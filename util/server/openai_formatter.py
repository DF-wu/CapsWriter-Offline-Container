# coding: utf-8
"""
OpenAI ASR 响应格式化器

将 CapsWriter 内部 Result 对象转为 OpenAI Whisper API 规范的
五种 response_format: json / text / verbose_json / srt / vtt。

- json:         {"text": "..."}            最常用,SDK 默认。
- text:         纯文本字符串。
- verbose_json: 含 duration / segments / words 的详尽结构。
- srt / vtt:    字幕格式。

切段策略: 利用 token + timestamp 序列, 在句末标点处切分;
退化时退回单一 segment (覆盖整个 duration)。
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from util.server.server_classes import Result


# 句末标点 (中/英)。匹配到时切出一个 segment。
_SENT_END = re.compile(r"[。！？!?]+|[.](?:\s|$)|[，、;；,]\s")


def _segments_from_tokens(
    tokens: List[str],
    timestamps: List[float],
    total_duration: float,
) -> List[Dict[str, Any]]:
    """依句末标点切分 segments; 若信息不足则返回空 list。"""
    if not tokens or len(tokens) != len(timestamps):
        return []

    segments: List[Dict[str, Any]] = []
    seg_start_idx = 0
    text_buf = ""

    for i, tok in enumerate(tokens):
        text_buf += tok
        is_last = i == len(tokens) - 1
        is_break = bool(_SENT_END.search(tok)) or is_last
        if not is_break:
            continue

        start = float(timestamps[seg_start_idx])
        if is_last:
            end = float(max(timestamps[i], total_duration))
        else:
            end = float(timestamps[i + 1]) if i + 1 < len(timestamps) else float(timestamps[i])

        text = text_buf.strip()
        if text:
            segments.append({
                "id": len(segments),
                "seek": 0,
                "start": start,
                "end": end,
                "text": text,
            })
        seg_start_idx = i + 1
        text_buf = ""

    return segments


def _words_from_tokens(
    tokens: List[str],
    timestamps: List[float],
    total_duration: float,
) -> List[Dict[str, Any]]:
    """逐字时间戳。CapsWriter 永远计算它们,verbose_json 中无条件包含。"""
    if not tokens or len(tokens) != len(timestamps):
        return []
    out: List[Dict[str, Any]] = []
    for i, (tok, ts) in enumerate(zip(tokens, timestamps)):
        start = float(ts)
        if i + 1 < len(timestamps):
            end = float(timestamps[i + 1])
        else:
            end = float(max(start, total_duration))
        out.append({"word": tok, "start": start, "end": end})
    return out


def _fmt_srt_ts(seconds: float) -> str:
    if not math.isfinite(seconds):
        seconds = 0.0
    total_ms = max(0, int(round(seconds * 1000)))
    h, remainder = divmod(total_ms, 3_600_000)
    m, remainder = divmod(remainder, 60_000)
    s, ms = divmod(remainder, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_vtt_ts(seconds: float) -> str:
    if not math.isfinite(seconds):
        seconds = 0.0
    total_ms = max(0, int(round(seconds * 1000)))
    h, remainder = divmod(total_ms, 3_600_000)
    m, remainder = divmod(remainder, 60_000)
    s, ms = divmod(remainder, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _final_text(result: Result) -> str:
    """优先取 text_accu (时间戳拼接,更精确),退化时取 text。"""
    return result.text_accu or result.text or ""


def format_response(
    result: Result,
    response_format: str,
    language: Optional[str] = None,
) -> Tuple[Any, str]:
    """
    返回 (body, media_type) 元组。

    Args:
        result:          识别完成的 Result (is_final=True)。
        response_format: 五种之一; 未知值视为 'json'。
        language:        请求声明的 language 字段; 用于 verbose_json 中回填。

    Returns:
        body 为 str (text/srt/vtt) 或 dict (json/verbose_json)。
        media_type 为对应 Content-Type 字符串。
    """
    text = _final_text(result)
    fmt = response_format.lower()

    if fmt == "text":
        return text, "text/plain; charset=utf-8"

    if fmt in ("srt", "vtt"):
        segments = _segments_from_tokens(result.tokens, result.timestamps, result.duration)
        if not segments:
            segments = [{
                "id": 0, "seek": 0,
                "start": 0.0, "end": float(result.duration or 0.0),
                "text": text,
            }]

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
        body = {
            "task": "transcribe",
            "language": language or "zh",
            "duration": float(result.duration or 0.0),
            "text": text,
            "segments": _segments_from_tokens(result.tokens, result.timestamps, result.duration),
            "words": _words_from_tokens(result.tokens, result.timestamps, result.duration),
        }
        return body, "application/json"

    # default: json
    return {"text": text}, "application/json"
