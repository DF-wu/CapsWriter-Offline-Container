#!/usr/bin/env python3
# coding: utf-8
"""No-GUI CapsWriter client for OpenAI-compatible HTTP API workflows.

The module intentionally uses only the Python standard library so it can run in
isolated environments on Linux and Windows without adding project-wide
dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional
from urllib import error, request


DEFAULT_BASE_URL = "http://127.0.0.1:6017"
DEFAULT_MODEL = "whisper-1"
RESPONSE_FORMATS = ("json", "text", "verbose_json", "srt", "vtt")


@dataclass(frozen=True)
class ApiConfig:
    base_url: str
    api_key: str = ""
    timeout: float = 120.0


@dataclass(frozen=True)
class HttpResult:
    status: int
    content_type: str
    body: bytes

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    def json(self):
        return json.loads(self.text)


def normalize_base_url(value: str) -> str:
    base = (value or DEFAULT_BASE_URL).strip().rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3].rstrip("/")
    return base or DEFAULT_BASE_URL


def auth_headers(config: ApiConfig) -> dict[str, str]:
    return (
        {"Authorization": f"Bearer {config.api_key.strip()}"}
        if config.api_key.strip()
        else {}
    )


def _read_response(response) -> HttpResult:
    return HttpResult(
        status=getattr(response, "status", response.getcode()),
        content_type=response.headers.get("Content-Type", ""),
        body=response.read(),
    )


def http_get_json(config: ApiConfig, path: str):
    req = request.Request(
        f"{normalize_base_url(config.base_url)}{path}",
        headers=auth_headers(config),
    )
    with request.urlopen(req, timeout=config.timeout) as response:
        return _read_response(response).json()


def http_get_json_status(config: ApiConfig, path: str) -> tuple[int, object]:
    req = request.Request(
        f"{normalize_base_url(config.base_url)}{path}",
        headers=auth_headers(config),
    )
    try:
        with request.urlopen(req, timeout=config.timeout) as response:
            result = _read_response(response)
    except error.HTTPError as exc:
        try:
            body = exc.read()
        finally:
            exc.close()
        result = HttpResult(
            status=exc.code,
            content_type=exc.headers.get("Content-Type", ""),
            body=body,
        )
    return result.status, result.json()


def build_multipart(
    file_path: Path,
    fields: dict[str, str],
    boundary: Optional[str] = None,
) -> tuple[bytes, str]:
    boundary = boundary or f"----CapsWriterBoundary{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    audio_data = file_path.read_bytes()

    chunks.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode("utf-8")
    )
    chunks.append(audio_data)
    chunks.append(b"\r\n")

    for name, value in fields.items():
        if value == "":
            continue
        chunks.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode("utf-8")
        )

    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), boundary


def transcribe_file(
    config: ApiConfig,
    file_path: Path,
    *,
    response_format: str = "text",
    model: str = DEFAULT_MODEL,
    language: str = "",
    prompt: str = "",
) -> HttpResult:
    if response_format not in RESPONSE_FORMATS:
        raise ValueError(f"Unsupported response_format: {response_format}")
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    body, boundary = build_multipart(
        file_path,
        {
            "model": model,
            "response_format": response_format,
            "language": language,
            "prompt": prompt,
        },
    )
    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        **auth_headers(config),
    }
    req = request.Request(
        f"{normalize_base_url(config.base_url)}/v1/audio/transcriptions",
        data=body,
        headers=headers,
        method="POST",
    )
    with request.urlopen(req, timeout=config.timeout) as response:
        return _read_response(response)


def render_transcription(result: HttpResult, response_format: str) -> str:
    if response_format in {"text", "srt", "vtt"}:
        return result.text
    payload = result.json()
    if response_format == "json":
        return payload.get("text", "")
    return json.dumps(payload, ensure_ascii=False, indent=2)


def output_path_for(audio_path: Path, response_format: str, output_dir: Path) -> Path:
    ext = {
        "text": ".txt",
        "json": ".json",
        "verbose_json": ".json",
        "srt": ".srt",
        "vtt": ".vtt",
    }[response_format]
    return output_dir / f"{audio_path.stem}{ext}"


def select_tts_command(
    text: str,
    *,
    platform_name: Optional[str] = None,
    which: Callable[[str], Optional[str]] = shutil.which,
    voice: str = "",
    rate: Optional[int] = None,
) -> list[str]:
    system = (platform_name or platform.system()).lower()
    if system.startswith("win"):
        # System.Speech ships with Windows PowerShell/.NET on normal desktop installs.
        escaped = text.replace("'", "''")
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        )
        if voice:
            script += f"$s.SelectVoice('{voice.replace("'", "''")}'); "
        if rate is not None:
            script += f"$s.Rate = {max(-10, min(10, int(rate)))}; "
        script += f"$s.Speak('{escaped}')"
        return ["powershell", "-NoProfile", "-Command", script]

    if system == "darwin" and which("say"):
        cmd = ["say"]
        if voice:
            cmd.extend(["-v", voice])
        if rate is not None:
            cmd.extend(["-r", str(rate)])
        cmd.append(text)
        return cmd

    if which("spd-say"):
        cmd = ["spd-say"]
        if rate is not None:
            cmd.extend(["--rate", str(max(-100, min(100, int(rate))))])
        cmd.append(text)
        return cmd

    for engine in ("espeak-ng", "espeak"):
        if which(engine):
            cmd = [engine]
            if voice:
                cmd.extend(["-v", voice])
            if rate is not None:
                cmd.extend(["-s", str(max(80, min(450, int(rate))))])
            cmd.append(text)
            return cmd

    raise RuntimeError(
        "No supported TTS engine found. Install spd-say, espeak-ng, espeak, "
        "or use Windows PowerShell System.Speech."
    )


def speak_text(text: str, *, voice: str = "", rate: Optional[int] = None, dry_run: bool = False) -> int:
    cmd = select_tts_command(text, voice=voice, rate=rate)
    if dry_run:
        print(" ".join(cmd))
        return 0
    return subprocess.run(cmd, check=False).returncode


def read_text_argument(value: str, from_file: bool) -> str:
    if from_file:
        return Path(value).read_text(encoding="utf-8")
    return value


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--base-url",
        default=os.environ.get("CAPSWRITER_API_BASE", DEFAULT_BASE_URL),
        help=f"HTTP API root, with or without /v1 (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--key",
        default=os.environ.get("CAPSWRITER_HTTP_API_KEY", ""),
        help="Bearer token, or CAPSWRITER_HTTP_API_KEY",
    )
    parser.add_argument("--timeout", type=float, default=120.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="capswriter-cli",
        description="No-GUI CapsWriter client for HTTP STT and local TTS.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    health = sub.add_parser("health", help="Check server health")
    add_common_options(health)

    ready = sub.add_parser("ready", help="Check server readiness diagnostics")
    add_common_options(ready)

    models = sub.add_parser("models", help="List server models")
    add_common_options(models)

    transcribe = sub.add_parser("transcribe", help="Transcribe one or more audio files")
    add_common_options(transcribe)
    transcribe.add_argument("audio", nargs="+", type=Path)
    transcribe.add_argument(
        "--format",
        choices=RESPONSE_FORMATS,
        default="text",
        dest="response_format",
    )
    transcribe.add_argument("--model", default=DEFAULT_MODEL)
    transcribe.add_argument("--language", default="")
    transcribe.add_argument("--prompt", default="")
    transcribe.add_argument("--output", type=Path, help="Output path for a single audio file")
    transcribe.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for one or more audio files",
    )

    speak = sub.add_parser("speak", help="Speak text with the local OS TTS engine")
    speak.add_argument("text", help="Text to speak, or a UTF-8 file path with --file")
    speak.add_argument("--file", action="store_true", help="Read text from a file")
    speak.add_argument("--voice", default="")
    speak.add_argument("--rate", type=int)
    speak.add_argument("--dry-run", action="store_true", help="Print the command only")

    return parser


def _config(args) -> ApiConfig:
    return ApiConfig(
        base_url=normalize_base_url(args.base_url),
        api_key=args.key,
        timeout=args.timeout,
    )


def command_health(args) -> int:
    print(json.dumps(http_get_json(_config(args), "/health"), ensure_ascii=False, indent=2))
    return 0


def command_ready(args) -> int:
    status, payload = http_get_json_status(_config(args), "/ready")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if status < 400 else 1


def command_models(args) -> int:
    print(json.dumps(http_get_json(_config(args), "/v1/models"), ensure_ascii=False, indent=2))
    return 0


def _write_output(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def command_transcribe(args) -> int:
    if args.output and len(args.audio) != 1:
        raise SystemExit("--output can only be used with one audio file")

    config = _config(args)
    outputs: list[tuple[Path, str]] = []
    for audio_path in args.audio:
        result = transcribe_file(
            config,
            audio_path,
            response_format=args.response_format,
            model=args.model,
            language=args.language,
            prompt=args.prompt,
        )
        rendered = render_transcription(result, args.response_format)
        if args.output:
            outputs.append((args.output, rendered))
        elif args.output_dir:
            outputs.append(
                (output_path_for(audio_path, args.response_format, args.output_dir), rendered)
            )
        else:
            if len(args.audio) > 1:
                print(f"==> {audio_path}")
            print(rendered)
            if not rendered.endswith("\n"):
                print()

    for path, text in outputs:
        _write_output(path, text)
        print(f"Wrote {path}")
    return 0


def command_speak(args) -> int:
    text = read_text_argument(args.text, args.file)
    return speak_text(text, voice=args.voice, rate=args.rate, dry_run=args.dry_run)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    commands = {
        "health": command_health,
        "ready": command_ready,
        "models": command_models,
        "transcribe": command_transcribe,
        "speak": command_speak,
    }
    try:
        return commands[args.command](args)
    except (OSError, error.URLError, RuntimeError, ValueError) as exc:
        print(f"capswriter-cli: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
