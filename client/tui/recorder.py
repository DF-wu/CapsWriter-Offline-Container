"""Optional, bounded microphone recording for CapsWriter TUI v2.

``sounddevice`` is never imported at module import time. The PortAudio callback
only copies PCM into a byte-bounded queue; a dedicated writer thread owns WAV
I/O. This keeps the callback non-blocking and makes memory use explicit.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import os
import queue
import shutil
import tempfile
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final, Protocol


SAMPLE_RATE: Final = 16_000
CHANNELS: Final = 1
SAMPLE_WIDTH_BYTES: Final = 2
BYTES_PER_SECOND: Final = SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH_BYTES
DEFAULT_MAX_RECORDING_SECONDS: Final = 300.0
MAX_RECORDING_SECONDS: Final = 1_800.0
DEFAULT_MAX_BUFFER_BYTES: Final = 2 * 1024 * 1024
MAX_BUFFER_BYTES: Final = 64 * 1024 * 1024
DEFAULT_BLOCKSIZE: Final = 1_600
WRITER_READY_TIMEOUT: Final = 3.0
WRITER_STOP_TIMEOUT: Final = 5.0
_SENTINEL: Final = object()


class RecorderError(RuntimeError):
    """Base class for safe-to-display recorder failures."""


class RecorderUnavailable(RecorderError):
    """The optional dependency or an input device is unavailable."""


class RecorderDeviceError(RecorderError):
    """PortAudio or WAV I/O failed."""


class RecorderOverflow(RecorderError):
    """The callback queue exceeded its explicit memory budget."""


class _RecordingLimitReached(Exception):
    """Internal control flow used to stop PortAudio at the duration cap."""


class _RecordingEnded(Exception):
    """Internal control flow for a callback from a retired stream."""


@dataclass(frozen=True)
class RecordedAudio:
    path: Path
    duration_seconds: float
    pcm_bytes: int
    limit_reached: bool = False


@dataclass(frozen=True)
class RecorderSnapshot:
    is_recording: bool
    elapsed_seconds: float
    pcm_bytes: int
    buffered_bytes: int
    limit_reached: bool
    error: RecorderError | None


class RecorderSurface(Protocol):
    available: bool
    unavailable_reason: str
    max_recording_seconds: float

    @property
    def is_recording(self) -> bool: ...

    def start(self) -> Path: ...

    def stop(self) -> RecordedAudio: ...

    def cancel(self) -> None: ...

    def snapshot(self) -> RecorderSnapshot: ...

    def owns(self, path: Path) -> bool: ...

    def is_private_path(self, path: Path) -> bool: ...

    def discard(self, path: Path) -> None: ...

    def cleanup(self) -> None: ...


def _bounded_float(value: float, *, name: str, maximum: float) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    if parsed > maximum:
        raise ValueError(f"{name} must not exceed {maximum:g}")
    return parsed


def _bounded_buffer(value: int) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError("recording buffer limit must be a positive integer")
    if value > MAX_BUFFER_BYTES:
        raise ValueError(f"recording buffer limit must not exceed {MAX_BUFFER_BYTES} bytes")
    return value


class UnavailableRecorder:
    """No-op capability object that keeps file-only mode fully functional."""

    available = False

    def __init__(
        self,
        reason: str = "optional sounddevice package is not installed",
        *,
        max_recording_seconds: float = DEFAULT_MAX_RECORDING_SECONDS,
    ) -> None:
        self.unavailable_reason = reason
        self.max_recording_seconds = _bounded_float(
            max_recording_seconds,
            name="maximum recording duration",
            maximum=MAX_RECORDING_SECONDS,
        )

    @property
    def is_recording(self) -> bool:
        return False

    def start(self) -> Path:
        raise RecorderUnavailable(self.unavailable_reason)

    def stop(self) -> RecordedAudio:
        raise RecorderUnavailable(self.unavailable_reason)

    def cancel(self) -> None:
        return None

    def snapshot(self) -> RecorderSnapshot:
        return RecorderSnapshot(False, 0.0, 0, 0, False, None)

    def owns(self, path: Path) -> bool:
        return False

    def is_private_path(self, path: Path) -> bool:
        return False

    def discard(self, path: Path) -> None:
        return None

    def cleanup(self) -> None:
        return None


class SoundDeviceRecorder:
    """Lazy ``sounddevice.RawInputStream`` recorder with bounded buffering."""

    available = True
    unavailable_reason = ""

    def __init__(
        self,
        *,
        module_loader: Callable[[], Any] | None = None,
        max_recording_seconds: float = DEFAULT_MAX_RECORDING_SECONDS,
        max_buffer_bytes: int = DEFAULT_MAX_BUFFER_BYTES,
        blocksize: int = DEFAULT_BLOCKSIZE,
        temp_root: Path | None = None,
    ) -> None:
        self.max_recording_seconds = _bounded_float(
            max_recording_seconds,
            name="maximum recording duration",
            maximum=MAX_RECORDING_SECONDS,
        )
        self.max_buffer_bytes = _bounded_buffer(max_buffer_bytes)
        if not isinstance(blocksize, int) or blocksize <= 0:
            raise ValueError("recording blocksize must be a positive integer")
        self.blocksize = blocksize
        self._max_pcm_bytes = int(self.max_recording_seconds * BYTES_PER_SECOND)
        self._max_pcm_bytes -= self._max_pcm_bytes % SAMPLE_WIDTH_BYTES
        self._module_loader = module_loader or (
            lambda: importlib.import_module("sounddevice")
        )
        self._temp_root = temp_root
        self._temp_dir: Path | None = None
        self._private_dirs: set[Path] = set()
        self._cleanup_dirs: set[Path] = set()
        self._managed_paths: set[Path] = set()
        self._deferred_discards: dict[Path, threading.Event] = {}
        self._control_lock = threading.RLock()
        self._paths_lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._active_token: object | None = None
        self._accepting_audio = False
        self._is_recording = False
        self._current_path: Path | None = None
        self._stream: Any | None = None
        self._module: Any | None = None
        self._pcm_queue: queue.Queue[bytes | object] | None = None
        self._writer_thread: threading.Thread | None = None
        self._writer_ready: threading.Event | None = None
        self._writer_done: threading.Event | None = None
        self._captured_bytes = 0
        self._written_bytes = 0
        self._queued_bytes = 0
        self._limit_reached = False
        self._terminal_error: RecorderError | None = None

    @property
    def is_recording(self) -> bool:
        with self._state_lock:
            return self._is_recording

    def _set_error(self, token: object, error: RecorderError) -> RecorderError:
        with self._state_lock:
            if token is not self._active_token:
                return error
            if self._terminal_error is None:
                self._terminal_error = error
            return self._terminal_error

    def _ensure_temp_dir(self) -> Path:
        if self._temp_dir is None or not self._temp_dir.exists():
            parent = str(self._temp_root) if self._temp_root is not None else None
            self._temp_dir = Path(
                tempfile.mkdtemp(prefix="capswriter-tui-audio-", dir=parent)
            ).resolve()
            try:
                self._temp_dir.chmod(0o700)
            except OSError:
                pass
            with self._paths_lock:
                self._private_dirs.add(self._temp_dir)
        return self._temp_dir

    def _new_path(self) -> Path:
        descriptor, raw_path = tempfile.mkstemp(
            prefix="recording-", suffix=".wav", dir=self._ensure_temp_dir()
        )
        os.close(descriptor)
        path = Path(raw_path).resolve()
        try:
            path.chmod(0o600)
        except OSError:
            pass
        with self._paths_lock:
            self._managed_paths.add(path)
        return path

    def _load_module(self) -> Any:
        try:
            return self._module_loader()
        except Exception as exc:
            raise RecorderUnavailable(f"sounddevice is unavailable: {exc}") from exc

    def _reset_counters(self) -> None:
        with self._state_lock:
            self._captured_bytes = 0
            self._written_bytes = 0
            self._queued_bytes = 0
            self._limit_reached = False
            self._terminal_error = None

    def _writer(
        self,
        token: object,
        path: Path,
        pcm_queue: queue.Queue[bytes | object],
        ready: threading.Event,
        done: threading.Event,
    ) -> None:
        try:
            with wave.open(str(path), "wb") as output:
                output.setnchannels(CHANNELS)
                output.setsampwidth(SAMPLE_WIDTH_BYTES)
                output.setframerate(SAMPLE_RATE)
                ready.set()
                while True:
                    item = pcm_queue.get()
                    if item is _SENTINEL:
                        break
                    if not isinstance(item, bytes):
                        continue
                    wrote = False
                    try:
                        output.writeframesraw(item)
                        wrote = True
                    finally:
                        with self._state_lock:
                            if token is self._active_token:
                                self._queued_bytes = max(
                                    0, self._queued_bytes - len(item)
                                )
                                if wrote:
                                    self._written_bytes += len(item)
        except Exception as exc:
            self._set_error(
                token, RecorderDeviceError(f"WAV writer failed: {exc}")
            )
        finally:
            ready.set()
            done.set()
            self._writer_finished(path, done)

    def _writer_finished(self, path: Path, done: threading.Event) -> None:
        with self._paths_lock:
            discard = self._deferred_discards.get(path) is done
        if discard:
            self.discard(path)
        self._retry_cleanup_dir(path.parent)

    def _retry_cleanup_dir(self, directory: Path) -> None:
        with self._paths_lock:
            if directory not in self._cleanup_dirs:
                return
        shutil.rmtree(directory, ignore_errors=True)
        if directory.exists():
            return
        with self._paths_lock:
            self._cleanup_dirs.discard(directory)
            self._private_dirs.discard(directory)
            for path in tuple(self._managed_paths):
                try:
                    path.relative_to(directory)
                except ValueError:
                    continue
                self._managed_paths.discard(path)
                self._deferred_discards.pop(path, None)

    def _accept_chunk(
        self,
        token: object,
        pcm_queue: queue.Queue[bytes | object],
        data: bytes,
        status: Any,
    ) -> None:
        with self._state_lock:
            if token is not self._active_token or not self._accepting_audio:
                raise _RecordingEnded
        if status:
            if getattr(status, "input_overflow", False):
                raise self._set_error(
                    token, RecorderOverflow("microphone input overflow")
                )
            raise self._set_error(
                token,
                RecorderDeviceError(f"microphone callback error: {str(status)[:300]}")
            )
        if not data:
            return
        if len(data) % SAMPLE_WIDTH_BYTES:
            raise self._set_error(
                token,
                RecorderDeviceError("microphone returned misaligned PCM data"),
            )

        with self._state_lock:
            if token is not self._active_token or not self._accepting_audio:
                raise _RecordingEnded
            if self._terminal_error is not None:
                raise self._terminal_error
            remaining = self._max_pcm_bytes - self._captured_bytes
            at_limit = len(data) >= remaining
            if remaining <= 0:
                self._limit_reached = True
                raise _RecordingLimitReached
            chunk = data[:remaining]
            chunk = chunk[: len(chunk) - (len(chunk) % SAMPLE_WIDTH_BYTES)]
            if self._queued_bytes + len(chunk) > self.max_buffer_bytes:
                error = RecorderOverflow(
                    f"recording buffer exceeded {self.max_buffer_bytes} bytes"
                )
                self._terminal_error = error
                raise error
            try:
                pcm_queue.put_nowait(chunk)
            except queue.Full as exc:
                error = RecorderOverflow("recording callback queue overflow")
                self._terminal_error = error
                raise error from exc
            self._queued_bytes += len(chunk)
            self._captured_bytes += len(chunk)
            if at_limit:
                self._limit_reached = True
        if at_limit:
            raise _RecordingLimitReached

    def _callback(
        self,
        token: object,
        module: Any,
        pcm_queue: queue.Queue[bytes | object],
        indata: Any,
        _frames: int,
        _time_info: Any,
        status: Any,
    ) -> None:
        try:
            self._accept_chunk(token, pcm_queue, bytes(indata), status)
        except _RecordingLimitReached:
            raise module.CallbackStop
        except _RecordingEnded:
            raise module.CallbackAbort
        except RecorderError:
            raise module.CallbackAbort
        except Exception as exc:
            self._set_error(
                token,
                RecorderDeviceError(f"microphone callback failed: {str(exc)[:300]}"),
            )
            raise module.CallbackAbort

    def _drain_queue(
        self, token: object, pcm_queue: queue.Queue[bytes | object]
    ) -> None:
        while True:
            try:
                item = pcm_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, bytes):
                with self._state_lock:
                    if token is self._active_token:
                        self._queued_bytes = max(
                            0, self._queued_bytes - len(item)
                        )

    def _finish_writer(
        self, token: object, *, discard_pending: bool
    ) -> RecorderError | None:
        pcm_queue = self._pcm_queue
        writer = self._writer_thread
        if pcm_queue is None or writer is None:
            return None
        if discard_pending:
            self._drain_queue(token, pcm_queue)
        try:
            pcm_queue.put(_SENTINEL, timeout=WRITER_STOP_TIMEOUT)
        except queue.Full:
            self._drain_queue(token, pcm_queue)
            try:
                pcm_queue.put_nowait(_SENTINEL)
            except queue.Full:
                return RecorderDeviceError("WAV writer queue could not be stopped")
        writer.join(WRITER_STOP_TIMEOUT)
        if writer.is_alive():
            return RecorderDeviceError(
                f"WAV writer did not stop within {WRITER_STOP_TIMEOUT:g} seconds"
            )
        return None

    def _close_stream(
        self, token: object, *, abort: bool
    ) -> RecorderError | None:
        stream = self._stream
        self._stream = None
        if abort:
            with self._state_lock:
                if token is self._active_token:
                    self._accepting_audio = False
        if stream is None:
            with self._state_lock:
                if token is self._active_token:
                    self._accepting_audio = False
                    self._is_recording = False
            return None
        failure: RecorderError | None = None
        try:
            if abort and hasattr(stream, "abort"):
                stream.abort()
            else:
                stream.stop()
        except Exception as exc:
            failure = RecorderDeviceError(f"microphone stop failed: {exc}")
        try:
            stream.close()
        except Exception as exc:
            failure = failure or RecorderDeviceError(f"microphone close failed: {exc}")
        with self._state_lock:
            if token is self._active_token:
                self._accepting_audio = False
                self._is_recording = False
        return failure

    def _defer_discard(self, path: Path, done: threading.Event | None) -> None:
        if done is None or done.is_set():
            return
        with self._paths_lock:
            self._deferred_discards[path] = done

    def _clear_session(self, token: object) -> None:
        self._stream = None
        self._module = None
        self._pcm_queue = None
        self._writer_thread = None
        self._writer_ready = None
        self._writer_done = None
        self._current_path = None
        with self._state_lock:
            if token is self._active_token:
                self._active_token = None
                self._accepting_audio = False
                self._is_recording = False

    def start(self) -> Path:
        with self._control_lock:
            if self.is_recording or self._current_path is not None:
                raise RecorderError("a microphone recording is already active")
            module = self._load_module()
            try:
                path = self._new_path()
            except OSError as exc:
                raise RecorderDeviceError(
                    f"temporary recording file could not be created: {exc}"
                ) from exc
            token = object()
            self._module = module
            self._current_path = path
            self._reset_counters()
            queue_capacity = max(
                2,
                math.ceil(
                    self.max_buffer_bytes
                    / max(1, self.blocksize * CHANNELS * SAMPLE_WIDTH_BYTES)
                ),
            )
            pcm_queue: queue.Queue[bytes | object] = queue.Queue(
                maxsize=queue_capacity
            )
            writer_ready = threading.Event()
            writer_done = threading.Event()
            self._pcm_queue = pcm_queue
            self._writer_ready = writer_ready
            self._writer_done = writer_done
            with self._state_lock:
                self._active_token = token
                self._accepting_audio = False
                self._is_recording = False
            self._writer_thread = threading.Thread(
                target=self._writer,
                args=(token, path, pcm_queue, writer_ready, writer_done),
                name="capswriter-tui-wav-writer",
                daemon=True,
            )
            self._writer_thread.start()
            if not writer_ready.wait(WRITER_READY_TIMEOUT):
                self._set_error(
                    token, RecorderDeviceError("WAV writer failed to initialize")
                )
            with self._state_lock:
                writer_error = self._terminal_error
            if writer_error is not None:
                self._finish_writer(token, discard_pending=True)
                self._defer_discard(path, writer_done)
                self._clear_session(token)
                self.discard(path)
                raise writer_error

            try:
                stream = module.RawInputStream(
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype="int16",
                    blocksize=self.blocksize,
                    callback=lambda indata, frames, time_info, status: self._callback(
                        token,
                        module,
                        pcm_queue,
                        indata,
                        frames,
                        time_info,
                        status,
                    ),
                )
                self._stream = stream
                with self._state_lock:
                    self._accepting_audio = True
                    self._is_recording = True
                stream.start()
            except Exception as exc:
                self._close_stream(token, abort=True)
                self._finish_writer(token, discard_pending=True)
                with self._state_lock:
                    callback_error = self._terminal_error
                failure = callback_error or RecorderDeviceError(
                    f"microphone could not start: {exc}"
                )
                self._defer_discard(path, writer_done)
                self._clear_session(token)
                self.discard(path)
                raise failure from exc
            return path

    def stop(self) -> RecordedAudio:
        with self._control_lock:
            path = self._current_path
            token = self._active_token
            if path is None or token is None or not self.is_recording:
                raise RecorderError("no microphone recording is active")
            close_error = self._close_stream(token, abort=False)
            with self._state_lock:
                callback_error = self._terminal_error
            writer_error = self._finish_writer(
                token,
                discard_pending=callback_error is not None or close_error is not None
            )
            with self._state_lock:
                written = self._written_bytes
                limit_reached = self._limit_reached
                terminal_error = self._terminal_error
            failure = terminal_error or close_error or writer_error
            if failure is not None:
                self._defer_discard(path, self._writer_done)
            self._clear_session(token)
            if failure is not None:
                self.discard(path)
                raise failure
            if written <= 0:
                self.discard(path)
                raise RecorderDeviceError("recording contained no audio data")
            return RecordedAudio(
                path=path,
                duration_seconds=written / BYTES_PER_SECOND,
                pcm_bytes=written,
                limit_reached=limit_reached,
            )

    def cancel(self) -> None:
        with self._control_lock:
            path = self._current_path
            token = self._active_token
            if path is not None and token is not None:
                self._close_stream(token, abort=True)
                self._finish_writer(token, discard_pending=True)
                self._defer_discard(path, self._writer_done)
                self._clear_session(token)
                self.discard(path)

    def snapshot(self) -> RecorderSnapshot:
        with self._state_lock:
            return RecorderSnapshot(
                is_recording=self._is_recording,
                elapsed_seconds=self._captured_bytes / BYTES_PER_SECOND,
                pcm_bytes=self._captured_bytes,
                buffered_bytes=self._queued_bytes,
                limit_reached=self._limit_reached,
                error=self._terminal_error,
            )

    @staticmethod
    def _resolved(path: Path) -> Path:
        return path.expanduser().resolve(strict=False)

    def owns(self, path: Path) -> bool:
        resolved = self._resolved(path)
        with self._paths_lock:
            return resolved in self._managed_paths

    def is_private_path(self, path: Path) -> bool:
        resolved = self._resolved(path)
        with self._paths_lock:
            private_dirs = tuple(self._private_dirs)
        for directory in private_dirs:
            try:
                resolved.relative_to(directory)
            except ValueError:
                continue
            return True
        return False

    def discard(self, path: Path) -> None:
        resolved = self._resolved(path)
        with self._state_lock:
            if (
                self._active_token is not None and resolved == self._current_path
            ):
                return
        with self._paths_lock:
            if resolved not in self._managed_paths:
                return
            writer_done = self._deferred_discards.get(resolved)
            if writer_done is not None and not writer_done.is_set():
                return
        try:
            resolved.unlink(missing_ok=True)
        except OSError:
            return
        with self._paths_lock:
            self._managed_paths.discard(resolved)
            self._deferred_discards.pop(resolved, None)

    def cleanup(self) -> None:
        with self._control_lock:
            try:
                self.cancel()
            except Exception:
                pass
            self._temp_dir = None
            with self._paths_lock:
                paths = tuple(self._managed_paths)
                private_dirs = tuple(self._private_dirs)
                self._cleanup_dirs.update(private_dirs)
            for path in paths:
                self.discard(path)
            for directory in private_dirs:
                self._retry_cleanup_dir(directory)


def create_optional_recorder(
    *,
    max_recording_seconds: float = DEFAULT_MAX_RECORDING_SECONDS,
    max_buffer_bytes: int = DEFAULT_MAX_BUFFER_BYTES,
    temp_root: Path | None = None,
) -> RecorderSurface:
    """Return a lazy recorder or a harmless file-only capability object."""

    try:
        spec = importlib.util.find_spec("sounddevice")
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        return UnavailableRecorder(max_recording_seconds=max_recording_seconds)
    return SoundDeviceRecorder(
        max_recording_seconds=max_recording_seconds,
        max_buffer_bytes=max_buffer_bytes,
        temp_root=temp_root,
    )
