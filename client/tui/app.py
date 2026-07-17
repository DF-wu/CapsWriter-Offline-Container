"""Textual application for CapsWriter TUI v2."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.timer import Timer
from textual.widgets import Button, Footer, Header, Input, Label, Select, Static, TextArea

from .api import (
    DEFAULT_BASE_URL,
    DEFAULT_DIAGNOSTIC_TIMEOUT,
    DEFAULT_MAX_RESPONSE_BYTES,
    DEFAULT_MODEL,
    DEFAULT_TRANSCRIPTION_TIMEOUT,
    CapsWriterApi,
    EndpointResult,
    TranscriptionResult,
    redact_secret,
)
from .i18n import Locale, Translator, normalize_locale
from .recorder import (
    DEFAULT_MAX_BUFFER_BYTES,
    DEFAULT_MAX_RECORDING_SECONDS,
    RecorderError,
    RecorderSurface,
    create_optional_recorder,
)
from .storage import atomic_write_text, suggested_output_path


class ApiSurface(Protocol):
    async def health(self) -> EndpointResult: ...

    async def ready(self) -> EndpointResult: ...

    async def models(self) -> EndpointResult: ...

    async def transcribe(
        self,
        audio_path: Path,
        *,
        response_format: str,
        model: str,
        language: str,
        prompt: str,
    ) -> TranscriptionResult: ...


ApiFactory = Callable[[str, str], ApiSurface]


@dataclass(frozen=True)
class DiagnosticLine:
    state: str
    detail: str = ""
    payload: dict[str, Any] | None = None


class CapsWriterTui(App[None]):
    """Keyboard-first, bilingual CapsWriter diagnostic/transcription client."""

    ENABLE_COMMAND_PALETTE = False
    TITLE = "CapsWriter TUI v2"
    BINDINGS = [
        Binding("f5", "refresh", "Refresh / 診斷", show=True),
        Binding("f8", "record_toggle", "Record / 錄音", show=True),
        Binding("f9", "cancel_recording", "Cancel mic / 取消錄音", show=True),
        Binding("ctrl+t", "transcribe", "Transcribe / 轉錄", show=True),
        Binding("escape", "cancel", "Cancel / 取消", show=True, priority=True),
        Binding("ctrl+s", "save", "Save / 儲存", show=True),
        Binding("ctrl+l", "toggle_locale", "EN / 中文", show=True),
        Binding("ctrl+o", "focus_audio", "Audio / 音訊", show=False),
        Binding("ctrl+q", "quit", "Quit / 離開", show=True, priority=True),
    ]

    CSS = """
    Screen {
        background: #020617;
        color: #f8fafc;
    }

    Header {
        background: #0f172a;
        color: #f8fafc;
    }

    Footer {
        background: #0f172a;
        color: #e2e8f0;
    }

    #main-scroll {
        height: 1fr;
        padding: 1 2;
        scrollbar-color: #38bdf8;
        scrollbar-background: #0f172a;
    }

    #hero {
        height: auto;
        min-height: 4;
        margin: 0 1 1 1;
        padding: 0 1;
        border-left: thick #38bdf8;
        background: #0f172a;
    }

    #hero-title {
        color: #7dd3fc;
        text-style: bold;
    }

    #hero-subtitle {
        color: #cbd5e1;
    }

    .panel {
        height: auto;
        margin: 0 1 1 1;
        padding: 1 2;
        border: round #334155;
        background: #0f172a;
    }

    .panel-title {
        height: 1;
        margin-bottom: 1;
        color: #7dd3fc;
        text-style: bold;
    }

    .field-label {
        height: 1;
        margin-top: 1;
        color: #e2e8f0;
        text-style: bold;
    }

    Input, Select, TextArea {
        border: tall #475569;
        background: #111827;
        color: #f8fafc;
    }

    Input:focus, Select:focus, TextArea:focus {
        border: tall #38bdf8;
    }

    Input:disabled, Select:disabled, TextArea:disabled, Button:disabled {
        opacity: 55%;
    }

    #connection-grid {
        layout: grid;
        grid-size: 3;
        grid-columns: 2fr 2fr 1fr;
        grid-gutter: 1 2;
        height: auto;
    }

    .connection-cell {
        height: auto;
    }

    #refresh {
        width: 100%;
        margin-top: 2;
        background: #0369a1;
        color: #ffffff;
    }

    #diagnostics {
        height: auto;
        min-height: 4;
        margin-top: 1;
        padding: 1;
        background: #111827;
        color: #e2e8f0;
    }

    #workspace {
        layout: horizontal;
        height: auto;
        min-height: 29;
    }

    #input-panel {
        width: 2fr;
        min-width: 40;
    }

    #result-panel {
        width: 3fr;
        min-width: 40;
    }

    #prompt {
        height: 5;
    }

    #phase-note {
        height: auto;
        margin: 1 0;
        padding: 1;
        border-left: thick #fbbf24;
        color: #fde68a;
        background: #1c1917;
    }

    #recorder-capability, #recorder-status {
        height: auto;
        min-height: 2;
        margin-top: 1;
        padding: 0 1;
        background: #111827;
        color: #cbd5e1;
    }

    #recorder-status {
        margin-top: 0;
        border-left: thick #64748b;
    }

    #recorder-status.state-working {
        border-left: thick #f87171;
        color: #fecaca;
    }

    #recorder-status.state-ok {
        border-left: thick #4ade80;
        color: #bbf7d0;
    }

    #recorder-status.state-error {
        border-left: thick #f87171;
        color: #fecaca;
    }

    #recorder-status.state-cancelled {
        border-left: thick #c4b5fd;
        color: #ddd6fe;
    }

    .actions {
        height: auto;
        margin-top: 1;
    }

    .actions Button {
        min-width: 18;
        margin-right: 1;
    }

    #transcribe, #save {
        background: #15803d;
        color: #ffffff;
    }

    #record-start {
        background: #0369a1;
        color: #ffffff;
    }

    #record-stop {
        background: #15803d;
        color: #ffffff;
    }

    #cancel, #record-cancel {
        background: #b91c1c;
        color: #ffffff;
    }

    #transcript-status {
        height: auto;
        min-height: 3;
        margin-bottom: 1;
        padding: 1;
        border-left: thick #64748b;
        background: #111827;
        color: #e2e8f0;
    }

    #transcript {
        height: 16;
        min-height: 8;
    }

    #transcript-status.state-working {
        border-left: thick #38bdf8;
        color: #bae6fd;
    }

    #transcript-status.state-ok {
        border-left: thick #4ade80;
        color: #bbf7d0;
    }

    #transcript-status.state-degraded {
        border-left: thick #fbbf24;
        color: #fde68a;
    }

    #transcript-status.state-error {
        border-left: thick #f87171;
        color: #fecaca;
    }

    #transcript-status.state-cancelled {
        border-left: thick #c4b5fd;
        color: #ddd6fe;
    }

    Screen.narrow #main-scroll {
        padding: 0 1;
    }

    Screen.narrow #connection-grid {
        grid-size: 1;
        grid-columns: 1fr;
    }

    Screen.narrow #refresh {
        margin-top: 1;
    }

    Screen.narrow #workspace {
        layout: vertical;
        min-height: 55;
    }

    Screen.narrow #input-panel, Screen.narrow #result-panel {
        width: 1fr;
        min-width: 20;
    }
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        initial_api_key: str = "",
        locale: str = "en",
        diagnostic_timeout: float = DEFAULT_DIAGNOSTIC_TIMEOUT,
        transcription_timeout: float = DEFAULT_TRANSCRIPTION_TIMEOUT,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        max_recording_seconds: float = DEFAULT_MAX_RECORDING_SECONDS,
        recording_buffer_bytes: int = DEFAULT_MAX_BUFFER_BYTES,
        show_clock: bool = True,
        api_factory: ApiFactory | None = None,
        recorder: RecorderSurface | None = None,
    ) -> None:
        super().__init__()
        self.locale: Locale = normalize_locale(locale)
        self._translator = Translator(self.locale)
        self._initial_base_url = base_url
        self._initial_api_key = initial_api_key
        self._diagnostic_timeout = diagnostic_timeout
        self._transcription_timeout = transcription_timeout
        self._max_response_bytes = max_response_bytes
        self._show_clock = show_clock
        self._api_factory = api_factory or self._default_api_factory
        self._recorder = (
            recorder
            if recorder is not None
            else create_optional_recorder(
                max_recording_seconds=max_recording_seconds,
                max_buffer_bytes=recording_buffer_bytes,
            )
        )
        self._busy = False
        self._request_cancellable = False
        self._recording_control_busy = False
        self._recording_cancel_requested = False
        self._recorded_path: Path | None = None
        self._recorder_timer: Timer | None = None
        self._recorder_status_kind = "idle"
        if self._recorder.available:
            self._recorder_status_key = "recorder_idle"
            self._recorder_status_values: dict[str, object] = {}
        else:
            self._recorder_status_key = "recorder_unavailable"
            self._recorder_status_values = {
                "reason": self._recorder.unavailable_reason,
            }
        self._transcript_text = ""
        self._transcript_format = "text"
        self._status_kind = "idle"
        self._status_key = "status_ready"
        self._status_values: dict[str, object] = {}
        self._network_activity_key: str | None = None
        self._network_activity_values: dict[str, object] = {}
        self._diagnostics: dict[str, DiagnosticLine] = {}

    def _default_api_factory(self, base_url: str, api_key: str) -> ApiSurface:
        return CapsWriterApi(
            base_url,
            api_key,
            diagnostic_timeout=self._diagnostic_timeout,
            transcription_timeout=self._transcription_timeout,
            max_response_bytes=self._max_response_bytes,
        )

    @property
    def t(self) -> Translator:
        return self._translator

    def compose(self) -> ComposeResult:
        t = self.t
        yield Header(show_clock=self._show_clock)
        with VerticalScroll(id="main-scroll"):
            with Container(id="hero"):
                yield Static(t("hero_title"), id="hero-title", markup=False)
                yield Static(t("hero_subtitle"), id="hero-subtitle", markup=False)

            with Container(classes="panel", id="connection-panel"):
                yield Static(t("connection_title"), classes="panel-title", id="connection-title", markup=False)
                with Container(id="connection-grid"):
                    with Container(classes="connection-cell"):
                        yield Label(t("ui_language"), classes="field-label", id="label-ui-language")
                        yield Select(
                            [(t("locale_en"), "en"), (t("locale_zh"), "zh-Hant")],
                            value=self.locale,
                            allow_blank=False,
                            id="ui-locale",
                        )
                    with Container(classes="connection-cell"):
                        yield Label(t("base_url"), classes="field-label", id="label-base-url")
                        yield Input(
                            value=self._initial_base_url,
                            placeholder=t("base_url_placeholder"),
                            id="base-url",
                        )
                    with Container(classes="connection-cell"):
                        yield Button(t("refresh"), id="refresh")
                    with Container(classes="connection-cell"):
                        yield Label(t("api_key"), classes="field-label", id="label-api-key")
                        yield Input(
                            value=self._initial_api_key,
                            placeholder=t("api_key_placeholder"),
                            password=True,
                            id="api-key",
                        )
                yield Static(t("diagnostics_empty"), id="diagnostics", markup=False)

            with Horizontal(id="workspace"):
                with Container(classes="panel", id="input-panel"):
                    yield Static(t("workspace_title"), classes="panel-title", id="workspace-title", markup=False)
                    yield Static(t("phase_note"), id="phase-note", markup=False)
                    yield Label(t("audio_file"), classes="field-label", id="label-audio-file")
                    yield Input(placeholder=t("audio_placeholder"), id="audio-file")
                    yield Static("", id="recorder-capability", markup=False)
                    yield Static("", id="recorder-status", classes="state-idle", markup=False)
                    with Horizontal(classes="actions", id="record-actions"):
                        yield Button(
                            t("record_start"),
                            id="record-start",
                            disabled=not self._recorder.available,
                        )
                        yield Button(t("record_stop"), id="record-stop", disabled=True)
                        yield Button(t("record_cancel"), id="record-cancel", disabled=True)
                    yield Label(t("model"), classes="field-label", id="label-model")
                    yield Input(value=DEFAULT_MODEL, placeholder=t("model_placeholder"), id="model")
                    yield Label(t("asr_language"), classes="field-label", id="label-asr-language")
                    yield Input(placeholder=t("asr_language_placeholder"), id="asr-language")
                    yield Label(t("prompt"), classes="field-label", id="label-prompt")
                    yield TextArea("", placeholder=t("prompt_placeholder"), id="prompt")
                    yield Label(t("response_format"), classes="field-label", id="label-response-format")
                    yield Select(
                        [(value, value) for value in ("text", "json", "verbose_json", "srt", "vtt")],
                        value="text",
                        allow_blank=False,
                        id="response-format",
                    )
                    yield Label(t("output_file"), classes="field-label", id="label-output-file")
                    yield Input(placeholder=t("output_placeholder"), id="output-file")
                    with Horizontal(classes="actions"):
                        yield Button(t("transcribe"), id="transcribe")
                        yield Button(t("cancel"), id="cancel", disabled=True)

                with Container(classes="panel", id="result-panel"):
                    yield Static(t("result_title"), classes="panel-title", id="result-title", markup=False)
                    yield Static(t("status_ready"), id="transcript-status", classes="state-idle", markup=False)
                    yield TextArea(
                        "",
                        placeholder=t("transcript_empty"),
                        read_only=True,
                        show_line_numbers=False,
                        id="transcript",
                    )
                    with Horizontal(classes="actions"):
                        yield Button(t("save"), id="save", disabled=True)
                        yield Button(t("clear"), id="clear", disabled=True)
        yield Footer(show_command_palette=False)

    def on_mount(self) -> None:
        self.title = self.t("app_title")
        self.screen.set_class(self.size.width < 100, "narrow")
        self._apply_language()
        self._sync_controls()
        if self._recorder.available:
            self._recorder_timer = self.set_interval(0.1, self._poll_recorder)
        self.query_one("#ui-locale", Select).focus()

    def on_unmount(self) -> None:
        if self._recorder_timer is not None:
            self._recorder_timer.pause()
            self._recorder_timer = None
        self._recorder.cleanup()
        self._recorded_path = None

    def on_resize(self, event: events.Resize) -> None:
        self.screen.set_class(event.size.width < 100, "narrow")

    def _api(self) -> ApiSurface:
        return self._api_factory(
            self.query_one("#base-url", Input).value,
            self.query_one("#api-key", Input).value,
        )

    def _ui_ready(self, selector: str = "#main-scroll") -> bool:
        try:
            return len(self.screen.query(selector)) > 0
        except Exception:
            return False

    def _public_error(self, error: BaseException) -> str:
        key = (
            self.query_one("#api-key", Input).value
            if self._ui_ready("#api-key")
            else ""
        )
        message = redact_secret(str(error) or error.__class__.__name__, key)
        return message[:1000]

    def _set_status(self, kind: str, key: str, **values: object) -> None:
        self._status_kind = kind
        self._status_key = key
        self._status_values = values
        if kind == "working":
            self._network_activity_key = key
            self._network_activity_values = values
        elif not self._busy:
            self._network_activity_key = None
            self._network_activity_values = {}
        self._update_header_activity()
        if not self._ui_ready("#transcript-status"):
            return
        status = self.query_one("#transcript-status", Static)
        for state in ("idle", "working", "ok", "degraded", "error", "cancelled"):
            status.remove_class(f"state-{state}")
        status.add_class(f"state-{kind}")
        status.update(self.t(key, **values))
        if kind in {"error", "degraded", "cancelled"} or (
            kind == "working" and key in {"status_transcribing", "status_saving"}
        ):
            status.scroll_visible(animate=False)

    def _set_recorder_status(self, kind: str, key: str, **values: object) -> None:
        meaningful_transition = (
            kind != self._recorder_status_kind or key != self._recorder_status_key
        )
        self._recorder_status_kind = kind
        self._recorder_status_key = key
        self._recorder_status_values = values
        self._update_header_activity()
        if not self._ui_ready("#recorder-status"):
            return
        status = self.query_one("#recorder-status", Static)
        for state in ("idle", "working", "ok", "error", "cancelled"):
            status.remove_class(f"state-{state}")
        status.add_class(f"state-{kind}")
        status.update(self.t(key, **values))
        if meaningful_transition:
            status.scroll_visible(animate=False)

    def _update_header_activity(self) -> None:
        """Keep active network and microphone work visible above every viewport."""

        if self._recorder_status_kind == "working":
            self.sub_title = self.t(
                self._recorder_status_key,
                **self._recorder_status_values,
            )
        elif self._network_activity_key is not None:
            self.sub_title = self.t(
                self._network_activity_key,
                **self._network_activity_values,
            )
        else:
            self.sub_title = ""

    def _render_recorder_capability(self) -> None:
        capability = self.query_one("#recorder-capability", Static)
        if self._recorder.available:
            capability.update(
                self.t(
                    "recorder_available",
                    seconds=self._recorder.max_recording_seconds,
                )
            )
        else:
            capability.update(
                self.t(
                    "recorder_unavailable",
                    reason=self._recorder.unavailable_reason,
                )
            )

    def _set_busy(self, busy: bool, *, cancellable: bool = True) -> None:
        self._busy = busy
        if not busy:
            self._network_activity_key = None
            self._network_activity_values = {}
        self._request_cancellable = busy and cancellable
        self._sync_controls()
        self._update_header_activity()

    def _sync_controls(self) -> None:
        if not self._ui_ready("#record-start"):
            return
        recording = self._recorder.is_recording
        for selector in (
            "#base-url",
            "#api-key",
            "#ui-locale",
            "#model",
            "#asr-language",
            "#prompt",
            "#response-format",
            "#output-file",
        ):
            self.query_one(selector).disabled = self._busy
        recording_locks_input = recording or self._recording_control_busy
        self.query_one("#audio-file", Input).disabled = self._busy or recording_locks_input
        self.query_one("#refresh", Button).disabled = self._busy or recording_locks_input
        self.query_one("#transcribe", Button).disabled = self._busy or recording_locks_input
        self.query_one("#cancel", Button).disabled = not self._request_cancellable
        self.query_one("#record-start", Button).disabled = (
            not self._recorder.available
            or self._busy
            or recording_locks_input
        )
        self.query_one("#record-stop", Button).disabled = (
            self._busy or self._recording_control_busy or not recording
        )
        self.query_one("#record-cancel", Button).disabled = (
            self._busy or self._recording_control_busy or not recording
        )
        self.query_one("#save", Button).disabled = (
            self._busy or not bool(self._transcript_text)
        )
        self.query_one("#clear", Button).disabled = (
            self._busy or not bool(self._transcript_text)
        )

    def _detail_text(self, payload: dict[str, Any], *, endpoint: str) -> str:
        if endpoint == "models":
            data = payload.get("data")
            if isinstance(data, list):
                names = [str(item.get("id")) for item in data if isinstance(item, dict) and item.get("id")]
                return ", ".join(names[:12]) or "—"
        parts: list[str] = []
        for key in ("status", "model", "version", "ready", "task_router_bound"):
            if key in payload and isinstance(payload[key], (str, int, float, bool)):
                parts.append(f"{key}={payload[key]}")

        checks = payload.get("checks")
        if isinstance(checks, dict):
            check_label_keys = {
                "task_router_bound": "diagnostic_check_task_router_bound",
                "recognizer_process_alive": "diagnostic_check_recognizer_process_alive",
                "ffmpeg_available": "diagnostic_check_ffmpeg_available",
            }
            ordered_keys = [
                *check_label_keys,
                *sorted(key for key in checks if key not in check_label_keys),
            ]
            for key in ordered_keys:
                value = checks.get(key)
                if not isinstance(value, (str, int, float, bool)):
                    continue
                label_key = check_label_keys.get(key)
                label = self.t(label_key) if label_key else key.replace("_", " ")
                if isinstance(value, bool):
                    rendered_value = self.t(
                        "diagnostic_check_ok" if value else "diagnostic_check_failed"
                    )
                else:
                    rendered_value = str(value)
                parts.append(f"{label}: {rendered_value}")
        return "\n  ".join(parts)[:500] or "—"

    def _render_diagnostics(self) -> None:
        if not self._diagnostics:
            rendered = self.t("diagnostics_empty")
        else:
            lines: list[str] = []
            for endpoint, label_key in (
                ("health", "diagnostic_health"),
                ("ready", "diagnostic_ready"),
                ("models", "diagnostic_models"),
            ):
                item = self._diagnostics.get(endpoint)
                if item is None:
                    state = self.t("diagnostic_none")
                    detail = "—"
                else:
                    state = self.t(f"diagnostic_{item.state}")
                    detail = (
                        self._detail_text(item.payload, endpoint=endpoint)
                        if item.payload is not None
                        else item.detail
                    )
                lines.append(f"{self.t(label_key)} — {state}\n  {detail}")
            rendered = "\n".join(lines)
        self.query_one("#diagnostics", Static).update(rendered)

    def _apply_language(self) -> None:
        t = self.t
        self.title = t("app_title")
        static_keys = {
            "#hero-title": "hero_title",
            "#hero-subtitle": "hero_subtitle",
            "#connection-title": "connection_title",
            "#workspace-title": "workspace_title",
            "#result-title": "result_title",
            "#phase-note": "phase_note",
            "#label-ui-language": "ui_language",
            "#label-base-url": "base_url",
            "#label-api-key": "api_key",
            "#label-audio-file": "audio_file",
            "#label-model": "model",
            "#label-asr-language": "asr_language",
            "#label-prompt": "prompt",
            "#label-response-format": "response_format",
            "#label-output-file": "output_file",
        }
        for selector, key in static_keys.items():
            self.query_one(selector, Static).update(t(key))

        input_placeholders = {
            "#base-url": "base_url_placeholder",
            "#api-key": "api_key_placeholder",
            "#audio-file": "audio_placeholder",
            "#model": "model_placeholder",
            "#asr-language": "asr_language_placeholder",
            "#output-file": "output_placeholder",
        }
        for selector, key in input_placeholders.items():
            self.query_one(selector, Input).placeholder = t(key)
        self.query_one("#prompt", TextArea).placeholder = t("prompt_placeholder")
        self.query_one("#transcript", TextArea).placeholder = t("transcript_empty")

        for selector, key in {
            "#refresh": "refresh",
            "#record-start": "record_start",
            "#record-stop": "record_stop",
            "#record-cancel": "record_cancel",
            "#transcribe": "transcribe",
            "#cancel": "cancel",
            "#save": "save",
            "#clear": "clear",
        }.items():
            self.query_one(selector, Button).label = t(key)

        locale_select = self.query_one("#ui-locale", Select)
        if locale_select.value != self.locale:
            locale_select.value = self.locale
        self._set_status(self._status_kind, self._status_key, **self._status_values)
        self._render_recorder_capability()
        self._set_recorder_status(
            self._recorder_status_kind,
            self._recorder_status_key,
            **self._recorder_status_values,
        )
        self._render_diagnostics()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "ui-locale" or not isinstance(event.value, str):
            return
        if len(self.screen.query("#hero-title")) == 0:
            return
        locale = normalize_locale(event.value)
        if locale != self.locale:
            self.locale = locale
            self._translator = Translator(locale)
            self._apply_language()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        actions = {
            "refresh": self.action_refresh,
            "record-start": self.action_start_recording,
            "record-stop": self.action_stop_recording,
            "record-cancel": self.action_cancel_recording,
            "transcribe": self.action_transcribe,
            "cancel": self.action_cancel,
            "save": self.action_save,
            "clear": self.action_clear,
        }
        action = actions.get(event.button.id or "")
        if action is not None:
            action()

    def _start_worker(self, coroutine: Awaitable[None], *, group: str = "network") -> None:
        if self._busy or self._recorder.is_recording or self._recording_control_busy:
            if hasattr(coroutine, "close"):
                coroutine.close()  # type: ignore[attr-defined]
            self._report_request_active()
            return
        self.run_worker(coroutine, group=group, exclusive=True, exit_on_error=False)

    def _report_request_active(self) -> None:
        if self._ui_ready():
            self.notify(self.t("error_request_active"), severity="warning")

    def action_refresh(self) -> None:
        self._start_worker(self._refresh_diagnostics())

    def _run_recording_control(self, coroutine: Awaitable[None]) -> None:
        if self._busy or self._recording_control_busy:
            if hasattr(coroutine, "close"):
                coroutine.close()  # type: ignore[attr-defined]
            if self._busy:
                self._report_request_active()
            return
        self._recording_control_busy = True
        self._sync_controls()
        self.run_worker(
            coroutine,
            group="recorder-control",
            exclusive=True,
            exit_on_error=False,
        )

    @staticmethod
    def _same_path(left: Path, right: Path) -> bool:
        return str(left.expanduser().resolve(strict=False)).casefold() == str(
            right.expanduser().resolve(strict=False)
        ).casefold()

    def _clear_recorded_input(self, path: Path) -> None:
        if not self._ui_ready("#audio-file"):
            if self._recorded_path is not None and self._same_path(
                self._recorded_path, path
            ):
                self._recorded_path = None
            return
        audio = self.query_one("#audio-file", Input)
        source_text = audio.value.strip().strip('"')
        if source_text and self._same_path(Path(source_text), path):
            audio.value = ""
        if self._recorded_path is not None and self._same_path(
            self._recorded_path, path
        ):
            self._recorded_path = None

    async def _discard_recording(self, path: Path) -> None:
        await asyncio.to_thread(self._recorder.discard, path)
        self._clear_recorded_input(path)

    def action_record_toggle(self) -> None:
        if self._recording_control_busy:
            return
        if self._recorder.is_recording:
            self.action_stop_recording()
        else:
            self.action_start_recording()

    def action_start_recording(self) -> None:
        if self._recording_control_busy:
            return
        if self._busy:
            self._report_request_active()
            return
        if not self._recorder.available:
            reason = self._recorder.unavailable_reason
            self._set_recorder_status(
                "error", "recorder_unavailable", reason=reason
            )
            self._set_status(
                "error",
                "status_failed",
                error=self.t("error_recorder_unavailable", reason=reason),
            )
            return
        if self._recorder.is_recording:
            return
        self._recording_cancel_requested = False
        self._run_recording_control(self._start_recording())

    async def _start_recording(self) -> None:
        self._set_recorder_status("working", "recorder_starting")
        try:
            if not self._ui_ready("#output-file"):
                return
            previous = self._recorded_path
            if previous is not None:
                await self._discard_recording(previous)
            self.query_one("#output-file", Input).value = ""
            await asyncio.to_thread(self._recorder.start)
            if not self._ui_ready("#record-start"):
                await asyncio.to_thread(self._recorder.cancel)
                return
            if self._recording_cancel_requested:
                await asyncio.to_thread(self._recorder.cancel)
                self._set_recorder_status("cancelled", "recorder_cancelled")
                self._set_status("cancelled", "status_recording_cancelled")
            else:
                self._set_recorder_status(
                    "working",
                    "recorder_recording",
                    elapsed=0.0,
                    limit=self._recorder.max_recording_seconds,
                    buffer_kib=0.0,
                )
        except Exception as exc:
            try:
                await asyncio.to_thread(self._recorder.cancel)
            except Exception:
                pass
            self._set_recorder_status(
                "error", "recorder_error", error=self._public_error(exc)
            )
        finally:
            self._recording_cancel_requested = False
            self._recording_control_busy = False
            self._sync_controls()

    def action_stop_recording(self) -> None:
        if not self._recorder.is_recording:
            return
        self._run_recording_control(self._stop_recording())

    async def _stop_recording(self) -> None:
        self._set_recorder_status("working", "recorder_stopping")
        try:
            recorded = await asyncio.to_thread(self._recorder.stop)
            if not self._ui_ready("#audio-file") or not self._ui_ready(
                "#output-file"
            ):
                await asyncio.to_thread(self._recorder.discard, recorded.path)
                return
            if self._recording_cancel_requested:
                await asyncio.to_thread(self._recorder.discard, recorded.path)
                self._clear_recorded_input(recorded.path)
                self._set_recorder_status("cancelled", "recorder_cancelled")
                self._set_status("cancelled", "status_recording_cancelled")
            else:
                self._recorded_path = recorded.path
                self.query_one("#audio-file", Input).value = str(recorded.path)
                self.query_one("#output-file", Input).value = ""
                recorder_message_id = (
                    "recorder_limit" if recorded.limit_reached else "recorder_recorded"
                )
                self._set_recorder_status(
                    "ok", recorder_message_id, elapsed=recorded.duration_seconds
                )
                self._set_status("ok", "status_recorded")
        except Exception as exc:
            try:
                await asyncio.to_thread(self._recorder.cancel)
            except Exception:
                pass
            self._set_recorder_status(
                "error", "recorder_error", error=self._public_error(exc)
            )
        finally:
            self._recording_cancel_requested = False
            self._recording_control_busy = False
            self._sync_controls()

    def action_cancel_recording(self) -> None:
        if self._recording_control_busy:
            self._recording_cancel_requested = True
            return
        if not self._recorder.is_recording and self._recorded_path is None:
            return
        self._recording_cancel_requested = False
        self._run_recording_control(self._cancel_recording())

    async def _cancel_recording(self) -> None:
        recorded = self._recorded_path
        try:
            await asyncio.to_thread(self._recorder.cancel)
            if recorded is not None:
                await self._discard_recording(recorded)
            self._set_recorder_status("cancelled", "recorder_cancelled")
            self._set_status("cancelled", "status_recording_cancelled")
        except Exception as exc:
            self._set_recorder_status(
                "error", "recorder_error", error=self._public_error(exc)
            )
        finally:
            self._recording_control_busy = False
            self._sync_controls()

    async def _abort_failed_recording(self, error: RecorderError) -> None:
        try:
            await asyncio.to_thread(self._recorder.cancel)
        finally:
            self._recording_control_busy = False
            self._set_recorder_status(
                "error", "recorder_error", error=self._public_error(error)
            )
            self._sync_controls()

    def _poll_recorder(self) -> None:
        if self._recording_control_busy or not self._ui_ready("#recorder-status"):
            return
        try:
            snapshot = self._recorder.snapshot()
        except Exception as exc:
            self._set_recorder_status(
                "error", "recorder_error", error=self._public_error(exc)
            )
            return
        if not snapshot.is_recording:
            return
        if snapshot.error is not None:
            self._run_recording_control(self._abort_failed_recording(snapshot.error))
            return
        if snapshot.limit_reached:
            self.action_stop_recording()
            return
        self._set_recorder_status(
            "working",
            "recorder_recording",
            elapsed=snapshot.elapsed_seconds,
            limit=self._recorder.max_recording_seconds,
            buffer_kib=snapshot.buffered_bytes / 1024,
        )

    async def _refresh_diagnostics(self) -> None:
        self._set_busy(True)
        self._set_status("working", "status_diagnostics")
        try:
            api = self._api()
            results = await asyncio.gather(
                api.health(), api.ready(), api.models(), return_exceptions=True
            )
            self._diagnostics = {}
            failed = False
            for endpoint, result in zip(("health", "ready", "models"), results, strict=True):
                if isinstance(result, BaseException):
                    failed = True
                    self._diagnostics[endpoint] = DiagnosticLine(
                        "error", self._public_error(result)
                    )
                    continue
                state = "ok" if result.ok else "degraded"
                failed = failed or state != "ok"
                self._diagnostics[endpoint] = DiagnosticLine(
                    state,
                    payload=result.payload,
                )
            self._render_diagnostics()
            if failed:
                self._set_status("degraded", "status_diagnostics_degraded")
            else:
                self._set_status("ok", "status_diagnostics_ok")
        except asyncio.CancelledError:
            self._set_status("cancelled", "status_cancelled")
            raise
        except Exception as exc:
            self._set_status("error", "status_failed", error=self._public_error(exc))
        finally:
            self._set_busy(False)

    def action_transcribe(self) -> None:
        self._start_worker(self._transcribe_file())

    async def _transcribe_file(self) -> None:
        source_text = self.query_one("#audio-file", Input).value.strip().strip('"')
        source = Path(source_text).expanduser()
        if not source_text or not source.is_file():
            self._set_status("error", "status_failed", error=self.t("error_audio_required"))
            return
        recorded_source = self._recorder.owns(source)

        format_value = self.query_one("#response-format", Select).value
        response_format = format_value if isinstance(format_value, str) else "text"
        self._set_busy(True)
        self._set_status("working", "status_transcribing", name=source.name)
        try:
            result = await self._api().transcribe(
                source,
                response_format=response_format,
                model=self.query_one("#model", Input).value,
                language=self.query_one("#asr-language", Input).value,
                prompt=self.query_one("#prompt", TextArea).text,
            )
            self._transcript_text = result.text
            self._transcript_format = result.response_format
            self.query_one("#transcript", TextArea).load_text(result.text)
            output = self.query_one("#output-file", Input)
            if not output.value.strip():
                suggested = suggested_output_path(source, result.response_format)
                if not self._recorder.is_private_path(suggested):
                    output.value = str(suggested)
            if recorded_source:
                await self._discard_recording(source)
            self._set_status("ok", "status_transcribed")
        except asyncio.CancelledError:
            if recorded_source:
                await self._discard_recording(source)
            self._set_status("cancelled", "status_cancelled")
            raise
        except Exception as exc:
            self._set_status("error", "status_failed", error=self._public_error(exc))
        finally:
            self._set_busy(False)

    def action_cancel(self) -> None:
        if self._recorder.is_recording or self._recording_control_busy:
            self.action_cancel_recording()
            return
        cancelled = self.workers.cancel_group(self, "network")
        if cancelled:
            self._set_status("cancelled", "status_cancelled")

    def action_save(self) -> None:
        if self._busy or self._recorder.is_recording or self._recording_control_busy:
            self._report_request_active()
            return
        if not self._transcript_text:
            self._set_status("error", "status_failed", error=self.t("error_no_transcript"))
            return
        output = self.query_one("#output-file", Input).value.strip().strip('"')
        if not output:
            self._set_status("error", "status_failed", error=self.t("error_output_required"))
            return
        target = Path(output).expanduser()
        if self._recorder.is_private_path(target):
            self._set_status(
                "error", "status_failed", error=self.t("error_output_private")
            )
            return
        audio_value = self.query_one("#audio-file", Input).value.strip().strip('"')
        if audio_value:
            source = Path(audio_value).expanduser()
            if self._same_path(target, source):
                self._set_status(
                    "error", "status_failed", error=self.t("error_output_is_audio")
                )
                return
        self._start_worker(self._save_transcript(target), group="save")

    async def _save_transcript(self, path: Path) -> None:
        self._set_busy(True, cancellable=False)
        self._set_status("working", "status_saving")
        try:
            target = await asyncio.to_thread(atomic_write_text, path, self._transcript_text)
            self._set_status("ok", "status_saved", path=target)
        except asyncio.CancelledError:
            self._set_status("cancelled", "status_cancelled")
            raise
        except Exception as exc:
            self._set_status("error", "status_failed", error=self._public_error(exc))
        finally:
            self._set_busy(False)

    def action_clear(self) -> None:
        if self._busy or self._recorder.is_recording or self._recording_control_busy:
            self._report_request_active()
            return
        self._transcript_text = ""
        self.query_one("#transcript", TextArea).load_text("")
        self.query_one("#save", Button).disabled = True
        self.query_one("#clear", Button).disabled = True
        self._set_status("idle", "status_cleared")

    def action_toggle_locale(self) -> None:
        target: Locale = "zh-Hant" if self.locale == "en" else "en"
        self.query_one("#ui-locale", Select).value = target

    def action_focus_audio(self) -> None:
        self.query_one("#audio-file", Input).focus()
