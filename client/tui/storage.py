"""Portable transcript filenames and durable atomic text writes."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path


WINDOWS_RESERVED_FILENAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
)
WINDOWS_UNSAFE_FILENAME_CHARS = frozenset('<>:"/\\|?*')
MAX_OUTPUT_STEM_CHARS = 120
FORMAT_EXTENSIONS = {
    "text": ".txt",
    "json": ".json",
    "verbose_json": ".json",
    "srt": ".srt",
    "vtt": ".vtt",
}


def safe_output_stem(stem: str) -> str:
    """Return a portable, bounded filename stem for Windows and Linux."""

    cleaned = "".join(
        "_" if ord(char) < 32 or char in WINDOWS_UNSAFE_FILENAME_CHARS else char
        for char in stem
    ).strip(" .")
    if not cleaned:
        cleaned = "audio"
    if cleaned.split(".", 1)[0].upper() in WINDOWS_RESERVED_FILENAMES:
        cleaned = f"{cleaned}_audio"
    if len(cleaned) > MAX_OUTPUT_STEM_CHARS:
        digest = hashlib.sha256(stem.encode("utf-8", errors="surrogatepass")).hexdigest()[:8]
        prefix = cleaned[: MAX_OUTPUT_STEM_CHARS - len(digest) - 1].rstrip(" .-_")
        cleaned = f"{prefix or 'audio'}-{digest}"
    return cleaned


def suggested_output_path(audio_path: Path, response_format: str) -> Path:
    try:
        extension = FORMAT_EXTENSIONS[response_format]
    except KeyError as exc:
        raise ValueError(f"unsupported response format: {response_format}") from exc
    target = audio_path.with_name(f"{safe_output_stem(audio_path.stem)}{extension}")
    if os.path.normcase(str(target.absolute())).casefold() == os.path.normcase(
        str(audio_path.absolute())
    ).casefold():
        target = audio_path.with_name(
            f"{safe_output_stem(audio_path.stem)}.transcript{extension}"
        )
    return target


def atomic_write_text(path: Path, text: str) -> Path:
    """Write UTF-8 text via same-directory fsync and atomic replacement."""

    target = path.expanduser()
    if not target.name:
        raise ValueError("output path must include a filename")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.is_dir():
        raise IsADirectoryError(target)

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
        return target
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
