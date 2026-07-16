from __future__ import annotations

import importlib
import os
import stat
import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path
from unittest import mock

from client.tui import recorder as recorder_module
from client.tui.recorder import (
    BYTES_PER_SECOND,
    CHANNELS,
    SAMPLE_RATE,
    SAMPLE_WIDTH_BYTES,
    RecorderDeviceError,
    RecorderOverflow,
    RecorderUnavailable,
    SoundDeviceRecorder,
    UnavailableRecorder,
    create_optional_recorder,
)


class FakeCallbackStop(Exception):
    pass


class FakeCallbackAbort(Exception):
    pass


class FakeStatus:
    def __init__(self, message: str = "", *, input_overflow: bool = False) -> None:
        self.message = message
        self.input_overflow = input_overflow

    def __bool__(self) -> bool:
        return bool(self.message or self.input_overflow)

    def __str__(self) -> str:
        return self.message or "input overflow"


class FakeRawInputStream:
    def __init__(self, module: "FakeSoundDevice", **kwargs: object) -> None:
        self.module = module
        self.kwargs = kwargs
        self.started = False
        self.closed = False
        self.aborted = False
        self.stopped_by_callback = False
        self.aborted_by_callback = False

    @property
    def callback(self):
        return self.kwargs["callback"]

    def start(self) -> None:
        if self.module.start_error is not None:
            raise self.module.start_error
        self.started = True

    def stop(self) -> None:
        if self.module.stop_error is not None:
            raise self.module.stop_error
        self.started = False

    def abort(self) -> None:
        self.aborted = True
        self.started = False

    def close(self) -> None:
        self.closed = True

    def emit(self, data: bytes, status: FakeStatus | None = None) -> str:
        try:
            self.callback(
                data,
                len(data) // SAMPLE_WIDTH_BYTES,
                None,
                status or FakeStatus(),
            )
        except FakeCallbackStop:
            self.stopped_by_callback = True
            self.started = False
            return "stop"
        except FakeCallbackAbort:
            self.aborted_by_callback = True
            self.started = False
            return "abort"
        return "continue"


class FakeSoundDevice:
    CallbackStop = FakeCallbackStop
    CallbackAbort = FakeCallbackAbort

    def __init__(self) -> None:
        self.construct_error: Exception | None = None
        self.start_error: Exception | None = None
        self.stop_error: Exception | None = None
        self.streams: list[FakeRawInputStream] = []

    def RawInputStream(self, **kwargs: object) -> FakeRawInputStream:
        if self.construct_error is not None:
            raise self.construct_error
        stream = FakeRawInputStream(self, **kwargs)
        self.streams.append(stream)
        return stream

    @property
    def stream(self) -> FakeRawInputStream:
        return self.streams[-1]


def wait_until(predicate, *, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class SoundDeviceRecorderTest(unittest.TestCase):
    def test_threaded_callback_writes_private_16khz_mono_pcm_wav(self) -> None:
        module = FakeSoundDevice()
        pcm = b"\x01\x00\xff\x7f" * 160
        with tempfile.TemporaryDirectory() as root:
            recorder = SoundDeviceRecorder(
                module_loader=lambda: module,
                temp_root=Path(root),
            )
            try:
                path = recorder.start()
                callback_results: list[str] = []
                callback_thread = threading.Thread(
                    target=lambda: callback_results.append(module.stream.emit(pcm))
                )
                callback_thread.start()
                callback_thread.join(timeout=1)
                self.assertFalse(callback_thread.is_alive())
                self.assertEqual(callback_results, ["continue"])
                self.assertEqual(recorder.snapshot().pcm_bytes, len(pcm))

                recording = recorder.stop()
                self.assertEqual(recording.path, path)
                self.assertEqual(recording.pcm_bytes, len(pcm))
                self.assertAlmostEqual(
                    recording.duration_seconds, len(pcm) / BYTES_PER_SECOND
                )
                self.assertTrue(recorder.owns(path))
                self.assertTrue(recorder.is_private_path(path))

                with wave.open(str(path), "rb") as source:
                    self.assertEqual(source.getframerate(), SAMPLE_RATE)
                    self.assertEqual(source.getnchannels(), CHANNELS)
                    self.assertEqual(source.getsampwidth(), SAMPLE_WIDTH_BYTES)
                    self.assertEqual(source.readframes(source.getnframes()), pcm)

                stream_options = module.stream.kwargs
                self.assertEqual(stream_options["samplerate"], SAMPLE_RATE)
                self.assertEqual(stream_options["channels"], CHANNELS)
                self.assertEqual(stream_options["dtype"], "int16")
                if os.name != "nt":
                    self.assertEqual(stat.S_IMODE(path.parent.stat().st_mode), 0o700)
                    self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            finally:
                recorder.cleanup()

    def test_duration_cap_truncates_current_block_and_stops_callback(self) -> None:
        module = FakeSoundDevice()
        maximum_seconds = 0.01
        maximum_bytes = int(maximum_seconds * BYTES_PER_SECOND)
        pcm = b"\x11\x22" * (maximum_bytes // 2 + 100)
        with tempfile.TemporaryDirectory() as root:
            recorder = SoundDeviceRecorder(
                module_loader=lambda: module,
                max_recording_seconds=maximum_seconds,
                temp_root=Path(root),
            )
            try:
                recorder.start()
                self.assertEqual(module.stream.emit(pcm), "stop")
                snapshot = recorder.snapshot()
                self.assertTrue(snapshot.limit_reached)
                self.assertEqual(snapshot.pcm_bytes, maximum_bytes)

                recording = recorder.stop()
                self.assertTrue(recording.limit_reached)
                self.assertEqual(recording.pcm_bytes, maximum_bytes)
                with wave.open(str(recording.path), "rb") as source:
                    self.assertEqual(
                        source.readframes(source.getnframes()), pcm[:maximum_bytes]
                    )
            finally:
                recorder.cleanup()

    def test_oversize_chunk_overflows_without_queueing_and_can_restart(self) -> None:
        module = FakeSoundDevice()
        with tempfile.TemporaryDirectory() as root:
            recorder = SoundDeviceRecorder(
                module_loader=lambda: module,
                max_buffer_bytes=32,
                blocksize=16,
                temp_root=Path(root),
            )
            try:
                failed_path = recorder.start()
                self.assertEqual(module.stream.emit(b"\x00\x00" * 17), "abort")
                snapshot = recorder.snapshot()
                self.assertIsInstance(snapshot.error, RecorderOverflow)
                self.assertEqual(snapshot.pcm_bytes, 0)
                self.assertEqual(snapshot.buffered_bytes, 0)
                with self.assertRaises(RecorderOverflow):
                    recorder.stop()
                self.assertFalse(failed_path.exists())
                self.assertFalse(recorder.owns(failed_path))

                recorder.start()
                self.assertEqual(module.stream.emit(b"\x01\x00" * 16), "continue")
                recording = recorder.stop()
                self.assertEqual(recording.pcm_bytes, 32)
            finally:
                recorder.cleanup()

    def test_portaudio_overflow_status_is_recoverable(self) -> None:
        module = FakeSoundDevice()
        with tempfile.TemporaryDirectory() as root:
            recorder = SoundDeviceRecorder(
                module_loader=lambda: module,
                temp_root=Path(root),
            )
            try:
                failed_path = recorder.start()
                status = FakeStatus("overflowed input", input_overflow=True)
                self.assertEqual(module.stream.emit(b"\x00\x00", status), "abort")
                with self.assertRaisesRegex(RecorderOverflow, "input overflow"):
                    recorder.stop()
                self.assertFalse(failed_path.exists())

                recorder.start()
                module.stream.emit(b"\x00\x00")
                self.assertEqual(recorder.stop().pcm_bytes, 2)
            finally:
                recorder.cleanup()

    def test_device_open_and_start_errors_leave_no_wav_and_allow_retry(self) -> None:
        module = FakeSoundDevice()
        with tempfile.TemporaryDirectory() as root:
            recorder = SoundDeviceRecorder(
                module_loader=lambda: module,
                temp_root=Path(root),
            )
            try:
                module.construct_error = RuntimeError("no input device")
                with self.assertRaisesRegex(RecorderDeviceError, "no input device"):
                    recorder.start()
                self.assertFalse(recorder.is_recording)
                self.assertEqual(list(Path(root).rglob("*.wav")), [])

                module.construct_error = None
                module.start_error = RuntimeError("device busy")
                with self.assertRaisesRegex(RecorderDeviceError, "device busy"):
                    recorder.start()
                self.assertFalse(recorder.is_recording)
                self.assertEqual(list(Path(root).rglob("*.wav")), [])

                module.start_error = None
                recorder.start()
                module.stream.emit(b"\x01\x00")
                self.assertEqual(recorder.stop().pcm_bytes, 2)
            finally:
                recorder.cleanup()

    def test_stop_error_discards_partial_wav_and_allows_retry(self) -> None:
        module = FakeSoundDevice()
        with tempfile.TemporaryDirectory() as root:
            recorder = SoundDeviceRecorder(
                module_loader=lambda: module,
                temp_root=Path(root),
            )
            try:
                failed_path = recorder.start()
                module.stream.emit(b"\x01\x00" * 8)
                module.stop_error = RuntimeError("device vanished")
                with self.assertRaisesRegex(RecorderDeviceError, "device vanished"):
                    recorder.stop()
                self.assertFalse(failed_path.exists())
                self.assertFalse(recorder.owns(failed_path))

                module.stop_error = None
                recorder.start()
                module.stream.emit(b"\x02\x00")
                self.assertEqual(recorder.stop().pcm_bytes, 2)
            finally:
                recorder.cleanup()

    def test_temporary_directory_error_is_reported_as_recorder_error(self) -> None:
        module = FakeSoundDevice()
        with tempfile.TemporaryDirectory() as root:
            missing_parent = Path(root, "missing", "parent")
            recorder = SoundDeviceRecorder(
                module_loader=lambda: module,
                temp_root=missing_parent,
            )
            with self.assertRaisesRegex(
                RecorderDeviceError, "temporary recording file"
            ):
                recorder.start()
            self.assertFalse(recorder.is_recording)
            self.assertEqual(module.streams, [])

    def test_writer_open_error_is_clean_and_recoverable(self) -> None:
        module = FakeSoundDevice()
        with tempfile.TemporaryDirectory() as root:
            recorder = SoundDeviceRecorder(
                module_loader=lambda: module,
                temp_root=Path(root),
            )
            try:
                with mock.patch.object(
                    recorder_module.wave,
                    "open",
                    side_effect=OSError("read-only filesystem"),
                ):
                    with self.assertRaisesRegex(RecorderDeviceError, "WAV writer"):
                        recorder.start()
                self.assertEqual(list(Path(root).rglob("*.wav")), [])

                recorder.start()
                module.stream.emit(b"\x00\x00")
                self.assertEqual(recorder.stop().pcm_bytes, 2)
            finally:
                recorder.cleanup()

    def test_cancel_deletes_active_recording(self) -> None:
        module = FakeSoundDevice()
        with tempfile.TemporaryDirectory() as root:
            recorder = SoundDeviceRecorder(
                module_loader=lambda: module,
                temp_root=Path(root),
            )
            try:
                path = recorder.start()
                module.stream.emit(b"\x00\x00" * 20)
                recorder.cancel()
                self.assertFalse(path.exists())
                self.assertFalse(recorder.owns(path))
                self.assertFalse(recorder.is_recording)
                self.assertTrue(module.stream.aborted)
                self.assertTrue(module.stream.closed)
            finally:
                recorder.cleanup()

    def test_cleanup_removes_stopped_and_active_files_and_private_directory(self) -> None:
        module = FakeSoundDevice()
        with tempfile.TemporaryDirectory() as root:
            recorder = SoundDeviceRecorder(
                module_loader=lambda: module,
                temp_root=Path(root),
            )
            stopped_path = recorder.start()
            module.stream.emit(b"\x01\x00")
            recorder.stop()
            active_path = recorder.start()
            module.stream.emit(b"\x02\x00")
            private_directory = stopped_path.parent

            recorder.cleanup()

            self.assertFalse(stopped_path.exists())
            self.assertFalse(active_path.exists())
            self.assertFalse(private_directory.exists())
            self.assertFalse(recorder.is_recording)
            self.assertFalse(recorder.owns(stopped_path))

    def test_discard_ignores_external_file_and_retries_failed_unlink(self) -> None:
        module = FakeSoundDevice()
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            external = root_path / "external.wav"
            external.write_bytes(b"keep")
            recorder = SoundDeviceRecorder(
                module_loader=lambda: module,
                temp_root=root_path,
            )
            try:
                recorder.discard(external)
                self.assertEqual(external.read_bytes(), b"keep")

                recorder.start()
                module.stream.emit(b"\x00\x00")
                managed = recorder.stop().path
                with mock.patch.object(
                    Path, "unlink", side_effect=PermissionError("in use")
                ):
                    recorder.discard(managed)
                self.assertTrue(managed.exists())
                self.assertTrue(recorder.owns(managed))

                recorder.discard(managed)
                self.assertFalse(managed.exists())
                self.assertFalse(recorder.owns(managed))
            finally:
                recorder.cleanup()

    def test_delayed_writer_timeout_isolated_from_next_recording(self) -> None:
        module = FakeSoundDevice()
        entered = threading.Event()
        release = threading.Event()
        real_wave_open = wave.open

        def delayed_first_open(*args, **kwargs):
            if not entered.is_set():
                entered.set()
                release.wait(timeout=1)
            return real_wave_open(*args, **kwargs)

        with tempfile.TemporaryDirectory() as root:
            recorder = SoundDeviceRecorder(
                module_loader=lambda: module,
                temp_root=Path(root),
            )
            try:
                with mock.patch.object(
                    recorder_module.wave, "open", side_effect=delayed_first_open
                ), mock.patch.object(
                    recorder_module, "WRITER_READY_TIMEOUT", 0.01
                ), mock.patch.object(
                    recorder_module, "WRITER_STOP_TIMEOUT", 0.01
                ):
                    with self.assertRaisesRegex(
                        RecorderDeviceError, "failed to initialize"
                    ):
                        recorder.start()
                    self.assertTrue(entered.is_set())
                    failed_paths = set(Path(root).rglob("*.wav"))
                    self.assertEqual(len(failed_paths), 1)

                    recorder.start()
                    module.stream.emit(b"\x33\x00" * 8)
                    successful = recorder.stop().path
                    self.assertTrue(successful.exists())
                    release.set()
                    self.assertTrue(
                        wait_until(lambda: all(not path.exists() for path in failed_paths))
                    )
                    self.assertTrue(successful.exists())
            finally:
                release.set()
                recorder.cleanup()


class OptionalRecorderTest(unittest.TestCase):
    def test_factory_does_not_import_missing_or_present_optional_package(self) -> None:
        with mock.patch.object(
            recorder_module.importlib.util, "find_spec", return_value=None
        ), mock.patch.object(
            importlib, "import_module", side_effect=AssertionError("must stay lazy")
        ) as importer:
            unavailable = create_optional_recorder()
            self.assertIsInstance(unavailable, UnavailableRecorder)
            importer.assert_not_called()
            with self.assertRaises(RecorderUnavailable):
                unavailable.start()

        with mock.patch.object(
            recorder_module.importlib.util, "find_spec", return_value=object()
        ), mock.patch.object(
            importlib, "import_module", side_effect=AssertionError("must stay lazy")
        ) as importer:
            available = create_optional_recorder()
            self.assertIsInstance(available, SoundDeviceRecorder)
            importer.assert_not_called()
            available.cleanup()
            importer.assert_not_called()

    def test_direct_recorder_loads_module_only_when_start_is_requested(self) -> None:
        module = FakeSoundDevice()
        loads: list[bool] = []
        with tempfile.TemporaryDirectory() as root:
            recorder = SoundDeviceRecorder(
                module_loader=lambda: loads.append(True) or module,
                temp_root=Path(root),
            )
            self.assertEqual(loads, [])
            recorder.cleanup()
            self.assertEqual(loads, [])

            recorder.start()
            self.assertEqual(loads, [True])
            recorder.cancel()


if __name__ == "__main__":
    unittest.main()
