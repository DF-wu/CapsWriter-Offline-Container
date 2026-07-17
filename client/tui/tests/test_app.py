from __future__ import annotations

import asyncio
import shutil
import tempfile
import unittest
from pathlib import Path

from textual.widgets import Button, Input, Select, Static, TextArea

from client.tui.api import EndpointResult, TranscriptionResult
from client.tui.app import CapsWriterTui
from client.tui.recorder import (
    RecordedAudio,
    RecorderOverflow,
    RecorderSnapshot,
    UnavailableRecorder,
)


class FakeApi:
    def __init__(
        self,
        *,
        transcript: str = "Hello, 世界",
        readiness_checks: dict[str, bool] | None = None,
    ) -> None:
        self.transcript = transcript
        self.readiness_checks = (
            {
                "task_router_bound": True,
                "recognizer_process_alive": True,
                "ffmpeg_available": True,
            }
            if readiness_checks is None
            else dict(readiness_checks)
        )
        self.transcribe_started = asyncio.Event()
        self.transcribe_release = asyncio.Event()
        self.slow = False
        self.cancelled = False
        self.transcribed_paths: list[Path] = []

    async def health(self) -> EndpointResult:
        return EndpointResult(200, {"status": "ok", "model": "mock-asr"})

    async def ready(self) -> EndpointResult:
        checks = dict(self.readiness_checks)
        ready = all(checks.values())
        return EndpointResult(
            200 if ready else 503,
            {
                "status": "ok" if ready else "degraded",
                "model": "mock-asr",
                "version": "test-v2",
                "checks": checks,
                "config": {
                    "auth_enabled": False,
                    "max_upload_mb": 256,
                    "max_audio_seconds": 600.0,
                    "task_timeout": 120.0,
                    "max_concurrent_requests": 1,
                    "max_pending_requests": 4,
                    "max_websocket_connections": 32,
                    "max_websocket_task_seconds": 120.0,
                    "cors_enabled": False,
                    "cors_origins_count": 0,
                    "log_transcripts": False,
                },
            },
        )

    async def models(self) -> EndpointResult:
        return EndpointResult(200, {"object": "list", "data": [{"id": "mock-asr"}]})

    async def transcribe(self, audio_path: Path, **kwargs) -> TranscriptionResult:
        self.transcribed_paths.append(audio_path)
        self.transcribe_started.set()
        try:
            if self.slow:
                await self.transcribe_release.wait()
            return TranscriptionResult(self.transcript, kwargs["response_format"], 200)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class MockRecorder:
    available = True
    unavailable_reason = ""
    max_recording_seconds = 30.0

    def __init__(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="capswriter-app-recorder-test-"))
        self.recording = False
        self.current: Path | None = None
        self.managed: set[Path] = set()
        self.cleanup_calls = 0
        self.snapshot_error = None
        self.limit_reached = False
        self.start_count = 0

    @property
    def is_recording(self) -> bool:
        return self.recording

    def start(self) -> Path:
        if self.recording:
            raise RuntimeError("already recording")
        self.root.mkdir(parents=True, exist_ok=True)
        self.start_count += 1
        path = self.root / f"recording-{self.start_count}.wav"
        path.write_bytes(b"mock pcm")
        self.current = path
        self.managed.add(path.resolve())
        self.recording = True
        self.limit_reached = False
        return path

    def stop(self) -> RecordedAudio:
        if not self.recording or self.current is None:
            raise RuntimeError("not recording")
        path = self.current
        self.recording = False
        self.current = None
        return RecordedAudio(path, 0.5, 16_000, self.limit_reached)

    def cancel(self) -> None:
        path = self.current
        self.recording = False
        self.current = None
        if path is not None:
            self.discard(path)

    def snapshot(self) -> RecorderSnapshot:
        return RecorderSnapshot(
            self.recording,
            0.5 if self.recording else 0.0,
            16_000 if self.recording else 0,
            1_024 if self.recording else 0,
            self.limit_reached,
            self.snapshot_error,
        )

    @staticmethod
    def _resolved(path: Path) -> Path:
        return path.expanduser().resolve(strict=False)

    def owns(self, path: Path) -> bool:
        return self._resolved(path) in self.managed

    def is_private_path(self, path: Path) -> bool:
        try:
            self._resolved(path).relative_to(self.root.resolve(strict=False))
        except ValueError:
            return False
        return True

    def discard(self, path: Path) -> None:
        resolved = self._resolved(path)
        if resolved not in self.managed:
            return
        resolved.unlink(missing_ok=True)
        self.managed.discard(resolved)

    def cleanup(self) -> None:
        self.cleanup_calls += 1
        self.cancel()
        for path in tuple(self.managed):
            self.discard(path)
        shutil.rmtree(self.root, ignore_errors=True)


class AppPilotTest(unittest.IsolatedAsyncioTestCase):
    async def test_mount_is_responsive_file_only_and_key_is_masked(self) -> None:
        app = CapsWriterTui(
            initial_api_key="never-visible",
            recorder=UnavailableRecorder("not installed"),
        )
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            self.assertTrue(app.screen.has_class("narrow"))
            self.assertTrue(app.query_one("#api-key", Input).password)
            self.assertIn("record locally", str(app.query_one("#phase-note", Static).content))
            self.assertIn("[FILE ONLY]", str(app.query_one("#recorder-capability", Static).content))
            self.assertTrue(app.query_one("#record-start", Button).disabled)
            self.assertEqual(app.query_one("#api-key", Input).value, "never-visible")

        wide = CapsWriterTui()
        async with wide.run_test(size=(120, 36)) as pilot:
            await pilot.pause()
            self.assertFalse(wide.screen.has_class("narrow"))

    async def test_keyboard_locale_switch_updates_catalog(self) -> None:
        app = CapsWriterTui(locale="en")
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press("ctrl+l")
            await pilot.pause()
            self.assertEqual(app.locale, "zh-Hant")
            self.assertIn("離線語音工作台", str(app.query_one("#hero-title", Static).content))
            self.assertIn("轉錄檔案", str(app.query_one("#transcribe", Button).label))

    async def test_refresh_uses_in_memory_key_and_renders_textual_states(self) -> None:
        fake = FakeApi()
        calls: list[tuple[str, str]] = []

        def factory(base_url: str, key: str) -> FakeApi:
            calls.append((base_url, key))
            return fake

        app = CapsWriterTui(api_factory=factory)
        async with app.run_test(size=(100, 30)) as pilot:
            app.query_one("#api-key", Input).value = "super-secret"
            await pilot.press("f5")
            await app.workers.wait_for_complete()
            diagnostics = str(app.query_one("#diagnostics", Static).content)
            status = str(app.query_one("#transcript-status", Static).content)
            self.assertIn("mock-asr", diagnostics)
            self.assertIn("[OK]", status)
            self.assertNotIn("super-secret", diagnostics + status)
        self.assertEqual(calls, [("http://127.0.0.1:6017", "super-secret")])

    async def test_degraded_readiness_renders_nested_checks_bilingually(self) -> None:
        cases = (
            (
                "en",
                (
                    "Readiness — DEGRADED",
                    "Task router bound: OK",
                    "Recognizer process alive: FAILED",
                    "FFmpeg available: FAILED",
                ),
            ),
            (
                "zh-Hant",
                (
                    "就緒狀態 — 功能受限",
                    "任務路由已綁定: 正常",
                    "辨識程序仍在執行: 失敗",
                    "FFmpeg 可用: 失敗",
                ),
            ),
        )
        for locale, expected_details in cases:
            with self.subTest(locale=locale):
                fake = FakeApi(
                    readiness_checks={
                        "task_router_bound": True,
                        "recognizer_process_alive": False,
                        "ffmpeg_available": False,
                    }
                )
                app = CapsWriterTui(
                    locale=locale,
                    recorder=UnavailableRecorder("not used"),
                    api_factory=lambda _url, _key, fake=fake: fake,
                )
                async with app.run_test(size=(100, 30)) as pilot:
                    await pilot.press("f5")
                    await app.workers.wait_for_complete()
                    diagnostics = str(app.query_one("#diagnostics", Static).content)
                    for expected in expected_details:
                        self.assertIn(expected, diagnostics)
                    self.assertNotIn("task_router_bound", diagnostics)
                    self.assertNotIn("recognizer_process_alive", diagnostics)

    async def test_readiness_details_retranslate_after_locale_switch(self) -> None:
        fake = FakeApi(
            readiness_checks={
                "task_router_bound": True,
                "recognizer_process_alive": False,
                "ffmpeg_available": False,
            }
        )
        app = CapsWriterTui(
            locale="en",
            recorder=UnavailableRecorder("not used"),
            api_factory=lambda _url, _key: fake,
        )
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press("f5")
            await app.workers.wait_for_complete()
            english = str(app.query_one("#diagnostics", Static).content)
            self.assertIn("Task router bound: OK", english)

            await pilot.press("ctrl+l")
            await pilot.pause()
            chinese = str(app.query_one("#diagnostics", Static).content)
            self.assertIn("任務路由已綁定: 正常", chinese)
            self.assertIn("辨識程序仍在執行: 失敗", chinese)
            self.assertNotIn("Task router bound", chinese)
            self.assertNotIn("FAILED", chinese)

    async def test_transcribe_display_save_and_clear_flow(self) -> None:
        fake = FakeApi(transcript="A durable transcript.")
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory, "meeting.wav")
            audio.write_bytes(b"audio")
            output = Path(directory, "saved", "meeting.txt")
            app = CapsWriterTui(api_factory=lambda _url, _key: fake)
            async with app.run_test(size=(100, 36)) as pilot:
                app.query_one("#audio-file", Input).value = str(audio)
                await pilot.press("ctrl+t")
                await app.workers.wait_for_complete()
                self.assertEqual(app.query_one("#transcript", TextArea).text, fake.transcript)
                self.assertFalse(app.query_one("#save", Button).disabled)
                self.assertTrue(app.query_one("#output-file", Input).value.endswith("meeting.txt"))

                app.query_one("#output-file", Input).value = str(output)
                await pilot.press("ctrl+s")
                await app.workers.wait_for_complete()
                self.assertEqual(output.read_text(encoding="utf-8"), fake.transcript)
                self.assertIn("[OK]", str(app.query_one("#transcript-status", Static).content))

                app.action_clear()
                self.assertEqual(app.query_one("#transcript", TextArea).text, "")
                self.assertTrue(app.query_one("#save", Button).disabled)

    async def test_escape_cancels_active_transcription(self) -> None:
        fake = FakeApi()
        fake.slow = True
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory, "meeting.wav")
            audio.write_bytes(b"audio")
            app = CapsWriterTui(api_factory=lambda _url, _key: fake)
            async with app.run_test(size=(80, 24)) as pilot:
                app.query_one("#audio-file", Input).value = str(audio)
                await pilot.press("ctrl+t")
                await asyncio.wait_for(fake.transcribe_started.wait(), timeout=1)
                self.assertFalse(app.query_one("#cancel", Button).disabled)
                await pilot.press("escape")
                await pilot.pause()
                self.assertTrue(fake.cancelled)
                self.assertIn(
                    "[CANCELLED]", str(app.query_one("#transcript-status", Static).content)
                )
                self.assertTrue(app.query_one("#cancel", Button).disabled)

    async def test_rejected_shortcut_preserves_active_request_status(self) -> None:
        fake = FakeApi(transcript="completed after the rejected refresh")
        fake.slow = True
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory, "meeting.wav")
            audio.write_bytes(b"audio")
            app = CapsWriterTui(api_factory=lambda _url, _key: fake)
            async with app.run_test(size=(80, 24)) as pilot:
                app.query_one("#audio-file", Input).value = str(audio)
                await pilot.press("ctrl+t")
                await asyncio.wait_for(fake.transcribe_started.wait(), timeout=1)
                working_status = str(
                    app.query_one("#transcript-status", Static).content
                )
                self.assertIn("[WORKING]", working_status)
                self.assertIn("[WORKING]", app.sub_title)

                await pilot.press("f5")
                await pilot.pause()
                self.assertEqual(
                    str(app.query_one("#transcript-status", Static).content),
                    working_status,
                )
                self.assertIn("[WORKING]", app.sub_title)
                self.assertFalse(fake.cancelled)

                fake.transcribe_release.set()
                await app.workers.wait_for_complete()
                self.assertEqual(
                    app.query_one("#transcript", TextArea).text,
                    fake.transcript,
                )
                self.assertEqual(app.sub_title, "")

    async def test_error_reflecting_key_is_redacted_from_status(self) -> None:
        class FailingApi(FakeApi):
            async def transcribe(self, audio_path: Path, **kwargs) -> TranscriptionResult:
                raise RuntimeError("peer reflected sk-sensitive")

        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory, "meeting.wav")
            audio.write_bytes(b"audio")
            app = CapsWriterTui(api_factory=lambda _url, _key: FailingApi())
            async with app.run_test(size=(80, 24)) as pilot:
                app.query_one("#api-key", Input).value = "sk-sensitive"
                app.query_one("#audio-file", Input).value = str(audio)
                await pilot.press("ctrl+t")
                await app.workers.wait_for_complete()
                status = str(app.query_one("#transcript-status", Static).content)
                self.assertIn("[ERROR]", status)
                self.assertIn("[REDACTED]", status)
                self.assertNotIn("sk-sensitive", status)

    async def test_missing_audio_has_actionable_localized_error(self) -> None:
        app = CapsWriterTui(locale="zh-Hant", api_factory=lambda _url, _key: FakeApi())
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press("ctrl+t")
            await pilot.pause()
            status = str(app.query_one("#transcript-status", Static).content)
            self.assertIn("[錯誤]", status)
            self.assertIn("音訊檔", status)

    async def test_save_refuses_to_overwrite_source_audio(self) -> None:
        fake = FakeApi(transcript="safe output")
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory, "meeting.wav")
            audio.write_bytes(b"original audio")
            app = CapsWriterTui(api_factory=lambda _url, _key: fake)
            async with app.run_test(size=(80, 24)) as pilot:
                app.query_one("#audio-file", Input).value = str(audio)
                await pilot.press("ctrl+t")
                await app.workers.wait_for_complete()
                app.query_one("#output-file", Input).value = str(audio)
                await pilot.press("ctrl+s")
                await pilot.pause()
                self.assertEqual(audio.read_bytes(), b"original audio")
                self.assertIn(
                    "must not overwrite",
                    str(app.query_one("#transcript-status", Static).content),
                )

    async def test_save_refuses_parent_alias_of_source_audio(self) -> None:
        fake = FakeApi(transcript="safe output")
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory, "meeting.wav")
            audio.write_bytes(b"original audio")
            existing_subdirectory = Path(directory, "existing")
            existing_subdirectory.mkdir()
            aliased_output = existing_subdirectory / ".." / audio.name
            self.assertNotEqual(str(aliased_output), str(audio))
            self.assertEqual(aliased_output.resolve(), audio.resolve())

            app = CapsWriterTui(api_factory=lambda _url, _key: fake)
            async with app.run_test(size=(80, 24)) as pilot:
                app.query_one("#audio-file", Input).value = str(audio)
                await pilot.press("ctrl+t")
                await app.workers.wait_for_complete()
                app.query_one("#output-file", Input).value = str(aliased_output)
                await pilot.press("ctrl+s")
                await app.workers.wait_for_complete()
                self.assertEqual(audio.read_bytes(), b"original audio")
                self.assertIn(
                    "must not overwrite",
                    str(app.query_one("#transcript-status", Static).content),
                )

    async def test_unavailable_recorder_keeps_file_transcription_working(self) -> None:
        fake = FakeApi()
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory, "file-only.wav")
            audio.write_bytes(b"audio")
            app = CapsWriterTui(
                recorder=UnavailableRecorder("optional dependency missing"),
                api_factory=lambda _url, _key: fake,
            )
            async with app.run_test(size=(80, 24)) as pilot:
                app.query_one("#audio-file", Input).value = str(audio)
                await pilot.press("ctrl+t")
                await app.workers.wait_for_complete()
                self.assertEqual(fake.transcribed_paths, [audio])
                self.assertIn(
                    "[OK]", str(app.query_one("#transcript-status", Static).content)
                )

    async def test_f8_starts_and_stops_recording_into_audio_input(self) -> None:
        recorder = MockRecorder()
        app = CapsWriterTui(recorder=recorder)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press("f8")
            await app.workers.wait_for_complete()
            await pilot.pause()
            self.assertTrue(recorder.is_recording)
            self.assertTrue(app.query_one("#audio-file", Input).disabled)
            self.assertFalse(app.query_one("#record-stop", Button).disabled)
            recorder_status = app.query_one("#recorder-status", Static)
            self.assertIn("[RECORDING]", str(recorder_status.content))
            self.assertIn("[RECORDING]", app.sub_title)
            recorder_geometry = app.screen.find_widget(recorder_status)
            self.assertEqual(
                recorder_geometry.visible_region,
                recorder_geometry.region,
            )

            await pilot.press("f8")
            await app.workers.wait_for_complete()
            selected = Path(app.query_one("#audio-file", Input).value)
            self.assertFalse(recorder.is_recording)
            self.assertTrue(selected.is_file())
            self.assertTrue(recorder.owns(selected))
            self.assertIn(
                "[RECORDED]", str(app.query_one("#recorder-status", Static).content)
            )
            self.assertIn(
                "Press Transcribe once",
                str(app.query_one("#recorder-status", Static).content),
            )
            self.assertNotIn("[RECORDING]", app.sub_title)

    async def test_recording_transcribes_once_then_deletes_private_wav(self) -> None:
        recorder = MockRecorder()
        fake = FakeApi(transcript="recorded words")
        app = CapsWriterTui(
            recorder=recorder,
            api_factory=lambda _url, _key: fake,
        )
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press("f8")
            await app.workers.wait_for_complete()
            await pilot.press("f8")
            await app.workers.wait_for_complete()
            recorded = Path(app.query_one("#audio-file", Input).value)

            await pilot.press("ctrl+t")
            await app.workers.wait_for_complete()
            self.assertEqual(fake.transcribed_paths, [recorded])
            self.assertFalse(recorded.exists())
            self.assertFalse(recorder.owns(recorded))
            self.assertEqual(app.query_one("#audio-file", Input).value, "")
            self.assertEqual(app.query_one("#output-file", Input).value, "")
            self.assertEqual(app.query_one("#transcript", TextArea).text, "recorded words")

    async def test_escape_and_f9_cancel_active_recording_and_remove_wav(self) -> None:
        recorder = MockRecorder()
        app = CapsWriterTui(recorder=recorder)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press("f8")
            await app.workers.wait_for_complete()
            current = recorder.current
            self.assertIsNotNone(current)
            await pilot.press("escape")
            await app.workers.wait_for_complete()
            assert current is not None
            self.assertFalse(current.exists())
            self.assertFalse(recorder.is_recording)
            self.assertIn(
                "[MIC CANCELLED]",
                str(app.query_one("#recorder-status", Static).content),
            )

            await pilot.press("f8")
            await app.workers.wait_for_complete()
            second = recorder.current
            self.assertIsNotNone(second)
            await pilot.press("f9")
            await app.workers.wait_for_complete()
            assert second is not None
            self.assertFalse(second.exists())
            self.assertFalse(recorder.is_recording)

    async def test_duration_limit_auto_stops_and_selects_recording(self) -> None:
        recorder = MockRecorder()
        app = CapsWriterTui(recorder=recorder)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press("f8")
            await app.workers.wait_for_complete()
            recorder.limit_reached = True
            await pilot.pause(0.2)
            await app.workers.wait_for_complete()
            selected = Path(app.query_one("#audio-file", Input).value)
            self.assertFalse(recorder.is_recording)
            self.assertTrue(selected.is_file())
            self.assertIn(
                "[LIMIT REACHED]",
                str(app.query_one("#recorder-status", Static).content),
            )

    async def test_callback_overflow_is_cleaned_and_recording_can_restart(self) -> None:
        recorder = MockRecorder()
        app = CapsWriterTui(recorder=recorder)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press("f8")
            await app.workers.wait_for_complete()
            failed = recorder.current
            recorder.snapshot_error = RecorderOverflow("mock input overflow")
            await pilot.pause(0.2)
            await app.workers.wait_for_complete()
            assert failed is not None
            self.assertFalse(failed.exists())
            self.assertFalse(recorder.is_recording)
            self.assertIn(
                "mock input overflow",
                str(app.query_one("#recorder-status", Static).content),
            )

            recorder.snapshot_error = None
            await pilot.press("f8")
            await app.workers.wait_for_complete()
            self.assertTrue(recorder.is_recording)

    async def test_escape_during_recorded_upload_deletes_private_wav(self) -> None:
        recorder = MockRecorder()
        fake = FakeApi()
        fake.slow = True
        app = CapsWriterTui(
            recorder=recorder,
            api_factory=lambda _url, _key: fake,
        )
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press("f8")
            await app.workers.wait_for_complete()
            await pilot.press("f8")
            await app.workers.wait_for_complete()
            recorded = Path(app.query_one("#audio-file", Input).value)

            await pilot.press("ctrl+t")
            await asyncio.wait_for(fake.transcribe_started.wait(), timeout=1)
            await pilot.press("escape")
            await app.workers.wait_for_complete()
            self.assertTrue(fake.cancelled)
            self.assertFalse(recorded.exists())
            self.assertEqual(app.query_one("#audio-file", Input).value, "")

    async def test_failed_upload_keeps_recording_for_retry(self) -> None:
        class FailingApi(FakeApi):
            async def transcribe(self, audio_path: Path, **kwargs) -> TranscriptionResult:
                self.transcribed_paths.append(audio_path)
                raise RuntimeError("temporary server failure")

        recorder = MockRecorder()
        fake = FailingApi()
        app = CapsWriterTui(
            recorder=recorder,
            api_factory=lambda _url, _key: fake,
        )
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press("f8")
            await app.workers.wait_for_complete()
            await pilot.press("f8")
            await app.workers.wait_for_complete()
            recorded = Path(app.query_one("#audio-file", Input).value)
            await pilot.press("ctrl+t")
            await app.workers.wait_for_complete()
            self.assertTrue(recorded.is_file())
            self.assertTrue(recorder.owns(recorded))
            self.assertEqual(app.query_one("#audio-file", Input).value, str(recorded))

    async def test_new_recording_discards_previous_stopped_recording(self) -> None:
        recorder = MockRecorder()
        app = CapsWriterTui(recorder=recorder)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press("f8")
            await app.workers.wait_for_complete()
            await pilot.press("f8")
            await app.workers.wait_for_complete()
            old = Path(app.query_one("#audio-file", Input).value)
            await pilot.press("f8")
            await app.workers.wait_for_complete()
            self.assertFalse(old.exists())
            self.assertFalse(recorder.owns(old))
            self.assertTrue(recorder.is_recording)

    async def test_exit_cleans_stopped_and_active_recordings(self) -> None:
        stopped = MockRecorder()
        app = CapsWriterTui(recorder=stopped)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press("f8")
            await app.workers.wait_for_complete()
            await pilot.press("f8")
            await app.workers.wait_for_complete()
            stopped_path = Path(app.query_one("#audio-file", Input).value)
            self.assertTrue(stopped_path.exists())
        self.assertFalse(stopped_path.exists())
        self.assertFalse(stopped.root.exists())
        self.assertEqual(stopped.cleanup_calls, 1)

        active = MockRecorder()
        app = CapsWriterTui(recorder=active)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press("f8")
            await app.workers.wait_for_complete()
            active_path = active.current
            self.assertIsNotNone(active_path)
        assert active_path is not None
        self.assertFalse(active_path.exists())
        self.assertFalse(active.root.exists())
        self.assertEqual(active.cleanup_calls, 1)

    async def test_locale_switch_translates_live_recorder_state(self) -> None:
        recorder = MockRecorder()
        app = CapsWriterTui(locale="en", recorder=recorder)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press("f8")
            await app.workers.wait_for_complete()
            await pilot.press("ctrl+l")
            await pilot.pause()
            self.assertEqual(app.locale, "zh-Hant")
            self.assertIn(
                "錄音中", str(app.query_one("#recorder-status", Static).content)
            )
            self.assertIn("停止並使用", str(app.query_one("#record-stop", Button).label))

    async def test_save_rejects_private_recording_directory(self) -> None:
        recorder = MockRecorder()
        fake = FakeApi(transcript="safe output")
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory, "meeting.wav")
            audio.write_bytes(b"audio")
            target = recorder.root / "must-not-save.txt"
            app = CapsWriterTui(
                recorder=recorder,
                api_factory=lambda _url, _key: fake,
            )
            async with app.run_test(size=(80, 24)) as pilot:
                app.query_one("#audio-file", Input).value = str(audio)
                await pilot.press("ctrl+t")
                await app.workers.wait_for_complete()
                app.query_one("#output-file", Input).value = str(target)
                await pilot.press("ctrl+s")
                await pilot.pause()
                self.assertFalse(target.exists())
                self.assertIn(
                    "outside the private recording directory",
                    str(app.query_one("#transcript-status", Static).content),
                )


if __name__ == "__main__":
    unittest.main()
