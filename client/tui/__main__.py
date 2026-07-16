"""Package entry point: ``python -m client.tui``."""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Sequence

from .api import (
    DEFAULT_BASE_URL,
    DEFAULT_DIAGNOSTIC_TIMEOUT,
    DEFAULT_MAX_RESPONSE_BYTES,
    DEFAULT_TRANSCRIPTION_TIMEOUT,
    MAX_RESPONSE_BYTES,
    MAX_TIMEOUT_SECONDS,
    normalize_base_url,
)
from .app import CapsWriterTui
from .i18n import normalize_locale
from .recorder import (
    DEFAULT_MAX_BUFFER_BYTES,
    DEFAULT_MAX_RECORDING_SECONDS,
    MAX_BUFFER_BYTES,
    MAX_RECORDING_SECONDS,
)


def bounded_timeout(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    if parsed > MAX_TIMEOUT_SECONDS:
        raise argparse.ArgumentTypeError(f"must not exceed {MAX_TIMEOUT_SECONDS:g}")
    return parsed


def response_megabytes(value: str) -> int:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    maximum = MAX_RESPONSE_BYTES / (1024 * 1024)
    if not math.isfinite(parsed) or parsed <= 0 or parsed > maximum:
        raise argparse.ArgumentTypeError(f"must be > 0 and <= {maximum:g}")
    return max(1, math.ceil(parsed * 1024 * 1024))


def recording_seconds(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0 or parsed > MAX_RECORDING_SECONDS:
        raise argparse.ArgumentTypeError(
            f"must be > 0 and <= {MAX_RECORDING_SECONDS:g}"
        )
    return parsed


def recording_megabytes(value: str) -> int:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    maximum = MAX_BUFFER_BYTES / (1024 * 1024)
    if not math.isfinite(parsed) or parsed <= 0 or parsed > maximum:
        raise argparse.ArgumentTypeError(f"must be > 0 and <= {maximum:g}")
    return max(1, math.ceil(parsed * 1024 * 1024))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="capswriter-tui",
        description=(
            "CapsWriter TUI v2: diagnostics, file transcription, and optional "
            "microphone capture."
        ),
    )
    parser.add_argument(
        "--lang",
        default=os.environ.get("CAPSWRITER_TUI_LANG", "en"),
        type=normalize_locale,
        metavar="{en,zh-Hant}",
        help="interface language (default: CAPSWRITER_TUI_LANG or en)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("CAPSWRITER_API_BASE", DEFAULT_BASE_URL),
        type=normalize_base_url,
        help="HTTP API root, with or without trailing /v1",
    )
    parser.add_argument(
        "--diagnostic-timeout",
        type=bounded_timeout,
        default=DEFAULT_DIAGNOSTIC_TIMEOUT,
        help=f"diagnostic timeout in seconds (maximum {MAX_TIMEOUT_SECONDS:g})",
    )
    parser.add_argument(
        "--transcription-timeout",
        type=bounded_timeout,
        default=DEFAULT_TRANSCRIPTION_TIMEOUT,
        help=f"transcription timeout in seconds (maximum {MAX_TIMEOUT_SECONDS:g})",
    )
    parser.add_argument(
        "--max-response-mb",
        type=response_megabytes,
        default=DEFAULT_MAX_RESPONSE_BYTES,
        help="response-body limit in MiB (maximum 64)",
    )
    parser.add_argument(
        "--max-recording-seconds",
        type=recording_seconds,
        default=DEFAULT_MAX_RECORDING_SECONDS,
        help=f"microphone duration limit (maximum {MAX_RECORDING_SECONDS:g} seconds)",
    )
    parser.add_argument(
        "--recording-buffer-mb",
        type=recording_megabytes,
        default=DEFAULT_MAX_BUFFER_BYTES,
        help="microphone callback-memory limit in MiB (maximum 64)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app = CapsWriterTui(
        base_url=args.base_url,
        initial_api_key=os.environ.get("CAPSWRITER_HTTP_API_KEY", ""),
        locale=args.lang,
        diagnostic_timeout=args.diagnostic_timeout,
        transcription_timeout=args.transcription_timeout,
        max_response_bytes=args.max_response_mb,
        max_recording_seconds=args.max_recording_seconds,
        recording_buffer_bytes=args.recording_buffer_mb,
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
