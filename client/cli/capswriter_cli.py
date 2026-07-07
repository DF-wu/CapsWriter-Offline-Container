#!/usr/bin/env python3
# coding: utf-8
"""No-GUI CapsWriter client for OpenAI-compatible HTTP API workflows.

The module intentionally uses only the Python standard library so it can run in
isolated environments on Linux and Windows without adding project-wide
dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
from http import client as http_client
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, TextIO
from urllib import error, parse, request


DEFAULT_BASE_URL = "http://127.0.0.1:6017"
DEFAULT_MODEL = "whisper-1"
DEFAULT_TIMEOUT_SECONDS = 600.0
DEFAULT_TTS_TIMEOUT_SECONDS = 120.0
RESPONSE_FORMATS = ("json", "text", "verbose_json", "srt", "vtt")
MAX_ERROR_BODY_CHARS = 500
SUPPORTED_BASE_URL_SCHEMES = {"http", "https"}
OUTPUT_STEM_FALLBACK = "audio"
MAX_OUTPUT_STEM_CHARS = 120
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


@dataclass(frozen=True)
class ApiConfig:
    base_url: str
    api_key: str = ""
    timeout: float = DEFAULT_TIMEOUT_SECONDS


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


class ApiError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        self.status = status
        self.message = message
        super().__init__(f"HTTP {status}: {message}")


@dataclass(frozen=True)
class MultipartBody:
    file_path: Path
    fields: tuple[tuple[str, str], ...]
    boundary: str
    content_length: int
    chunk_size: int

    def __iter__(self) -> Iterator[bytes]:
        yield _multipart_file_header(self.file_path, self.boundary)
        with self.file_path.open("rb") as audio:
            while True:
                chunk = audio.read(self.chunk_size)
                if not chunk:
                    break
                yield chunk
        yield b"\r\n"
        for name, value in self.fields:
            yield _multipart_field(name, value, self.boundary)
        yield _multipart_closing_boundary(self.boundary)


def normalize_base_url(value: str) -> str:
    base = (value or DEFAULT_BASE_URL).strip().rstrip("/")
    if not base:
        return DEFAULT_BASE_URL
    parts = parse.urlsplit(base)
    if parts.scheme not in SUPPORTED_BASE_URL_SCHEMES or not parts.hostname:
        raise ValueError("API base URL must be an absolute http:// or https:// URL")
    if parts.username or parts.password:
        raise ValueError("API base URL must not include username or password")
    if parts.query or parts.fragment:
        raise ValueError("API base URL must not include query or fragment")
    try:
        parts.port
    except ValueError as exc:
        raise ValueError("API base URL has an invalid port") from exc
    path = parts.path.rstrip("/")
    if path == "/v1":
        path = ""
    elif path.endswith("/v1"):
        path = path[:-3].rstrip("/")
    return parse.urlunsplit(parts._replace(path=path, query="", fragment=""))


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def auth_headers(config: ApiConfig) -> dict[str, str]:
    return (
        {"Authorization": f"Bearer {config.api_key.strip()}"}
        if config.api_key.strip()
        else {}
    )


def read_api_key_file(path: str) -> str:
    if not path:
        return ""
    value = Path(path).read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"API key file must not be empty: {path}")
    return value


def resolve_api_key(api_key: str, api_key_file: str) -> str:
    explicit_key = api_key.strip()
    if explicit_key:
        return explicit_key
    return read_api_key_file(api_key_file)


def _read_response(response) -> HttpResult:
    return HttpResult(
        status=getattr(response, "status", response.getcode()),
        content_type=response.headers.get("Content-Type", ""),
        body=response.read(),
    )


def _result_from_http_error(exc: error.HTTPError) -> HttpResult:
    try:
        body = exc.read()
    finally:
        exc.close()
    return HttpResult(
        status=exc.code,
        content_type=exc.headers.get("Content-Type", ""),
        body=body,
    )


def error_message_from_body(body: bytes) -> str:
    text = body.decode("utf-8", errors="replace").strip()
    if not text:
        return ""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        preview = " ".join(text.split())
        if len(preview) > MAX_ERROR_BODY_CHARS:
            return f"{preview[:MAX_ERROR_BODY_CHARS].rstrip()}..."
        return preview
    if isinstance(payload, dict):
        error_payload = payload.get("error")
        if isinstance(error_payload, dict) and error_payload.get("message"):
            return str(error_payload["message"])
        if payload.get("detail"):
            return str(payload["detail"])
    return text


def _expected_json_message(result: HttpResult, path: str) -> str:
    message = error_message_from_body(result.body)
    if message:
        return f"Expected JSON response from {path}: {message}"
    return (
        f"Expected JSON response from {path}, got "
        f"{result.content_type or 'unknown content type'}"
    )


def _json_or_raise(
    result: HttpResult,
    path: str,
    *,
    json_error_statuses: tuple[int, ...] = (),
):
    try:
        return result.json()
    except json.JSONDecodeError:
        if result.status < 400 or result.status in json_error_statuses:
            raise ApiError(result.status, _expected_json_message(result, path)) from None
        message = error_message_from_body(result.body)
        raise ApiError(result.status, message or _expected_json_message(result, path)) from None


def _raise_api_error(exc: error.HTTPError) -> None:
    result = _result_from_http_error(exc)
    _raise_result_api_error(result, exc.reason)


def _raise_result_api_error(result: HttpResult, reason: str = "Request failed") -> None:
    message = error_message_from_body(result.body) or reason or "Request failed"
    raise ApiError(result.status, message) from None


def http_get_json(config: ApiConfig, path: str):
    req = request.Request(
        f"{normalize_base_url(config.base_url)}{path}",
        headers=auth_headers(config),
    )
    try:
        with request.urlopen(req, timeout=config.timeout) as response:
            return _json_or_raise(_read_response(response), path)
    except error.HTTPError as exc:
        _raise_api_error(exc)


def http_get_json_status(
    config: ApiConfig,
    path: str,
    *,
    json_error_statuses: tuple[int, ...] = (),
) -> tuple[int, object]:
    req = request.Request(
        f"{normalize_base_url(config.base_url)}{path}",
        headers=auth_headers(config),
    )
    try:
        with request.urlopen(req, timeout=config.timeout) as response:
            result = _read_response(response)
    except error.HTTPError as exc:
        result = _result_from_http_error(exc)
    return result.status, _json_or_raise(
        result,
        path,
        json_error_statuses=json_error_statuses,
    )


def http_post_stream(
    config: ApiConfig,
    path: str,
    body: Iterable[bytes],
    headers: dict[str, str],
) -> HttpResult:
    url = parse.urlsplit(f"{normalize_base_url(config.base_url)}{path}")
    if url.scheme not in {"http", "https"} or not url.hostname:
        raise ValueError(f"Unsupported URL: {url.geturl()}")
    target = parse.urlunsplit(("", "", url.path or "/", url.query, ""))
    connection_class = (
        http_client.HTTPSConnection if url.scheme == "https" else http_client.HTTPConnection
    )
    connection = connection_class(url.hostname, url.port, timeout=config.timeout)
    send_error: OSError | None = None
    try:
        connection.putrequest("POST", target)
        for name, value in headers.items():
            connection.putheader(name, value)
        connection.endheaders()
        for chunk in body:
            if not chunk:
                continue
            try:
                connection.send(chunk)
            except (BrokenPipeError, ConnectionResetError) as exc:
                send_error = exc
                break
        try:
            response = connection.getresponse()
        except OSError as exc:
            if send_error is not None:
                raise error.URLError(send_error) from exc
            raise
        result = HttpResult(
            status=response.status,
            content_type=response.getheader("Content-Type", ""),
            body=response.read(),
        )
        if result.status >= 400:
            _raise_result_api_error(result, response.reason)
        return result
    finally:
        connection.close()


def multipart_header_value(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace('"', r"\"")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def _multipart_file_header(file_path: Path, boundary: str) -> bytes:
    return (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; '
        f'filename="{multipart_header_value(file_path.name)}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8")


def _multipart_field(name: str, value: str, boundary: str) -> bytes:
    return (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
        f"{value}\r\n"
    ).encode("utf-8")


def _multipart_closing_boundary(boundary: str) -> bytes:
    return f"--{boundary}--\r\n".encode("utf-8")


def build_multipart_stream(
    file_path: Path,
    fields: dict[str, str],
    boundary: Optional[str] = None,
    *,
    chunk_size: int = 1024 * 1024,
) -> tuple[Iterable[bytes], str, int]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    boundary = boundary or f"----CapsWriterBoundary{uuid.uuid4().hex}"
    active_fields = tuple((name, value) for name, value in fields.items() if value != "")
    file_header = _multipart_file_header(file_path, boundary)
    field_chunks = [_multipart_field(name, value, boundary) for name, value in active_fields]
    closing = _multipart_closing_boundary(boundary)
    content_length = (
        len(file_header)
        + file_path.stat().st_size
        + len(b"\r\n")
        + sum(len(chunk) for chunk in field_chunks)
        + len(closing)
    )
    return (
        MultipartBody(file_path, active_fields, boundary, content_length, chunk_size),
        boundary,
        content_length,
    )


def build_multipart(
    file_path: Path,
    fields: dict[str, str],
    boundary: Optional[str] = None,
) -> tuple[bytes, str]:
    body, boundary, _content_length = build_multipart_stream(file_path, fields, boundary)
    return b"".join(body), boundary


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
    body, boundary, content_length = build_multipart_stream(
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
        "Content-Length": str(content_length),
        **auth_headers(config),
    }
    return http_post_stream(
        config,
        "/v1/audio/transcriptions",
        body,
        headers,
    )


def render_transcription(result: HttpResult, response_format: str) -> str:
    if response_format in {"text", "srt", "vtt"}:
        return result.text
    payload = _json_or_raise(result, "/v1/audio/transcriptions")
    return json.dumps(payload, ensure_ascii=False, indent=2)


def safe_output_stem(stem: str) -> str:
    cleaned = "".join(
        "_"
        if ord(char) < 32 or char in WINDOWS_UNSAFE_FILENAME_CHARS
        else char
        for char in stem
    ).strip(" .")
    if not cleaned:
        cleaned = OUTPUT_STEM_FALLBACK
    if cleaned.split(".", 1)[0].upper() in WINDOWS_RESERVED_FILENAMES:
        cleaned = f"{cleaned}_{OUTPUT_STEM_FALLBACK}"
    if len(cleaned) > MAX_OUTPUT_STEM_CHARS:
        digest = hashlib.sha1(
            stem.encode("utf-8", errors="surrogatepass")
        ).hexdigest()[:8]
        prefix = cleaned[: MAX_OUTPUT_STEM_CHARS - len(digest) - 1].rstrip(" .-_")
        cleaned = f"{prefix or OUTPUT_STEM_FALLBACK}-{digest}"
    return cleaned


def output_path_for(audio_path: Path, response_format: str, output_dir: Path) -> Path:
    ext = {
        "text": ".txt",
        "json": ".json",
        "verbose_json": ".json",
        "srt": ".srt",
        "vtt": ".vtt",
    }[response_format]
    return output_dir / f"{safe_output_stem(audio_path.stem)}{ext}"


def output_target_collision_key(target: Path) -> str:
    return os.path.normcase(str(target)).casefold()


def output_targets_for(
    audio_paths: Iterable[Path],
    response_format: str,
    output_dir: Path,
) -> dict[Path, Path]:
    targets: dict[Path, Path] = {}
    seen: dict[str, tuple[Path, Path]] = {}
    for audio_path in audio_paths:
        target = output_path_for(audio_path, response_format, output_dir)
        key = output_target_collision_key(target)
        if key in seen:
            first_audio, first_target = seen[key]
            raise ValueError(
                "--output-dir would write multiple inputs to "
                f"{target}: {first_audio} and {audio_path}"
            )
        seen[key] = (audio_path, target)
        targets[audio_path] = target
    return targets


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


def speak_text(
    text: str,
    *,
    voice: str = "",
    rate: Optional[int] = None,
    dry_run: bool = False,
    timeout: float = DEFAULT_TTS_TIMEOUT_SECONDS,
) -> int:
    cmd = select_tts_command(text, voice=voice, rate=rate)
    if dry_run:
        print(" ".join(cmd))
        return 0
    try:
        return subprocess.run(cmd, check=False, timeout=timeout).returncode
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"TTS engine timed out after {timeout:g}s") from exc


def read_text_argument(
    value: str | None,
    *,
    from_file: bool,
    from_stdin: bool = False,
    stdin: TextIO | None = None,
) -> str:
    if from_stdin:
        if value is not None:
            raise ValueError("--stdin cannot be combined with a text argument")
        return (stdin or sys.stdin).read()
    if value is None:
        source = "file path" if from_file else "text"
        raise ValueError(f"{source} is required unless --stdin is used")
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
    parser.add_argument(
        "--key-file",
        default=os.environ.get("CAPSWRITER_HTTP_API_KEY_FILE", ""),
        help="UTF-8 file containing the Bearer token, or CAPSWRITER_HTTP_API_KEY_FILE",
    )
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=(
            "HTTP request timeout in seconds "
            f"(default: {DEFAULT_TIMEOUT_SECONDS:g}; matches server task timeout)"
        ),
    )


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
    speak.add_argument("text", nargs="?", help="Text to speak, or a UTF-8 file path with --file")
    speak_source = speak.add_mutually_exclusive_group()
    speak_source.add_argument("--file", action="store_true", help="Read text from a file")
    speak_source.add_argument("--stdin", action="store_true", help="Read text from standard input")
    speak.add_argument("--voice", default="")
    speak.add_argument("--rate", type=int)
    speak.add_argument(
        "--tts-timeout",
        type=positive_float,
        default=DEFAULT_TTS_TIMEOUT_SECONDS,
        help=f"Local TTS engine timeout in seconds (default: {DEFAULT_TTS_TIMEOUT_SECONDS:g})",
    )
    speak.add_argument("--dry-run", action="store_true", help="Print the command only")

    return parser


def _config(args) -> ApiConfig:
    return ApiConfig(
        base_url=normalize_base_url(args.base_url),
        api_key=resolve_api_key(args.key, args.key_file),
        timeout=args.timeout,
    )


def command_health(args) -> int:
    print(json.dumps(http_get_json(_config(args), "/health"), ensure_ascii=False, indent=2))
    return 0


def command_ready(args) -> int:
    status, payload = http_get_json_status(
        _config(args),
        "/ready",
        json_error_statuses=(503,),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if status < 400 else 1


def command_models(args) -> int:
    print(json.dumps(http_get_json(_config(args), "/v1/models"), ensure_ascii=False, indent=2))
    return 0


def _write_output(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)
    except Exception:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def command_transcribe(args) -> int:
    if args.output and len(args.audio) != 1:
        raise SystemExit("--output can only be used with one audio file")

    config = _config(args)
    output_targets = (
        output_targets_for(args.audio, args.response_format, args.output_dir)
        if args.output_dir
        else {}
    )
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
            outputs.append((output_targets[audio_path], rendered))
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
    text = read_text_argument(args.text, from_file=args.file, from_stdin=args.stdin)
    return speak_text(
        text,
        voice=args.voice,
        rate=args.rate,
        dry_run=args.dry_run,
        timeout=args.tts_timeout,
    )


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
