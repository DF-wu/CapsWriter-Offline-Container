"""Small, explicit message catalogs for the CapsWriter TUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal


Locale = Literal["en", "zh-Hant"]
DEFAULT_LOCALE: Final[Locale] = "en"


CATALOGS: Final[dict[Locale, dict[str, str]]] = {
    "en": {
        "app_title": "CapsWriter TUI v2",
        "hero_title": "CAPSWRITER  /  OFFLINE SPEECH WORKSPACE",
        "hero_subtitle": "OpenAI-compatible diagnostics, files, and optional microphone capture",
        "connection_title": "SERVER CONNECTION",
        "workspace_title": "SPEECH INPUT · FILE + OPTIONAL MICROPHONE",
        "result_title": "TRANSCRIPT",
        "ui_language": "Interface language",
        "base_url": "API root",
        "base_url_placeholder": "http://127.0.0.1:6017",
        "api_key": "API key (memory only)",
        "api_key_placeholder": "Optional Bearer token · hidden · never saved",
        "refresh": "Refresh diagnostics",
        "audio_file": "Audio file path",
        "audio_placeholder": "Paste a local .wav, .mp3, .m4a, .flac, or .webm path",
        "model": "Model",
        "model_placeholder": "whisper-1",
        "asr_language": "Recognition language hint",
        "asr_language_placeholder": "Optional, for example zh or en",
        "prompt": "Recognition prompt",
        "prompt_placeholder": "Optional vocabulary or context (maximum 2048 characters)",
        "response_format": "Response format",
        "output_file": "Save as",
        "output_placeholder": "Optional output path; choose after transcription if blank",
        "transcribe": "Transcribe file",
        "cancel": "Cancel request",
        "save": "Save transcript",
        "clear": "Clear transcript",
        "phase_note": "[INPUT] Paste an audio path or record locally. File mode remains available without sounddevice or an input device.",
        "recorder_available": "[MIC READY] 16 kHz mono PCM · {seconds:g}s maximum · private temporary storage",
        "recorder_unavailable": "[FILE ONLY] Microphone is optional and unavailable: {reason}",
        "recorder_idle": "[MIC IDLE] Start recording when ready.",
        "recorder_starting": "[MIC STARTING] Opening the default input device…",
        "recorder_recording": "[RECORDING] {elapsed:.1f}s / {limit:g}s · buffer {buffer_kib:.0f} KiB",
        "recorder_stopping": "[MIC STOPPING] Finalizing the private WAV file…",
        "recorder_recorded": "[RECORDED] {elapsed:.1f}s captured. Press Transcribe once to send it.",
        "recorder_limit": "[LIMIT REACHED] {elapsed:.1f}s captured and ready to transcribe.",
        "recorder_cancelled": "[MIC CANCELLED] Temporary recording removed.",
        "recorder_error": "[MIC ERROR] {error} Fix the device issue and try again; file input still works.",
        "record_start": "Start recording",
        "record_stop": "Stop & use",
        "record_cancel": "Cancel recording",
        "diagnostics_empty": "Health, readiness, and model results will appear here.",
        "transcript_empty": "No transcript yet. Select an audio file and start transcription.",
        "status_ready": "[IDLE] Ready for diagnostics, file input, or microphone recording.",
        "status_diagnostics": "[WORKING] Checking health, readiness, and models…",
        "status_transcribing": "[WORKING] Uploading and transcribing {name}…",
        "status_saving": "[WORKING] Saving transcript…",
        "status_diagnostics_ok": "[OK] Server diagnostics refreshed.",
        "status_diagnostics_degraded": "[DEGRADED] Some diagnostics failed; details remain visible.",
        "status_transcribed": "[OK] Transcription completed. Review or save the result.",
        "status_saved": "[OK] Transcript saved atomically to {path}.",
        "status_cancelled": "[CANCELLED] The active request was cancelled safely.",
        "status_cleared": "[IDLE] Transcript cleared.",
        "status_recorded": "[READY] Recorded audio is selected. Transcribe it with Ctrl+T.",
        "status_recording_cancelled": "[CANCELLED] Recording stopped and its temporary audio was removed.",
        "status_failed": "[ERROR] {error}",
        "error_audio_required": "Choose an existing audio file before transcribing.",
        "error_output_required": "Enter an output path before saving.",
        "error_output_is_audio": "The transcript output must not overwrite the source audio file.",
        "error_no_transcript": "There is no transcript to save.",
        "error_request_active": "Wait for or cancel the active request first.",
        "error_recorder_unavailable": "Microphone recording is unavailable: {reason}",
        "error_output_private": "Choose a permanent output path outside the private recording directory.",
        "diagnostic_health": "Health",
        "diagnostic_ready": "Readiness",
        "diagnostic_models": "Models",
        "diagnostic_ok": "OK",
        "diagnostic_degraded": "DEGRADED",
        "diagnostic_error": "ERROR",
        "diagnostic_none": "none reported",
        "diagnostic_check_task_router_bound": "Task router bound",
        "diagnostic_check_recognizer_process_alive": "Recognizer process alive",
        "diagnostic_check_ffmpeg_available": "FFmpeg available",
        "diagnostic_check_ok": "OK",
        "diagnostic_check_failed": "FAILED",
        "locale_en": "English",
        "locale_zh": "繁體中文",
    },
    "zh-Hant": {
        "app_title": "CapsWriter TUI v2",
        "hero_title": "CAPSWRITER  /  離線語音工作台",
        "hero_subtitle": "OpenAI 相容診斷、檔案與選用麥克風錄音",
        "connection_title": "伺服器連線",
        "workspace_title": "語音輸入 · 檔案 + 選用麥克風",
        "result_title": "轉錄文字",
        "ui_language": "介面語言",
        "base_url": "API 根網址",
        "base_url_placeholder": "http://127.0.0.1:6017",
        "api_key": "API 金鑰（僅存於記憶體）",
        "api_key_placeholder": "選填 Bearer token · 已遮罩 · 絕不儲存",
        "refresh": "重新整理診斷",
        "audio_file": "音訊檔案路徑",
        "audio_placeholder": "貼上本機 .wav、.mp3、.m4a、.flac 或 .webm 路徑",
        "model": "模型",
        "model_placeholder": "whisper-1",
        "asr_language": "辨識語言提示",
        "asr_language_placeholder": "選填，例如 zh 或 en",
        "prompt": "辨識提示詞",
        "prompt_placeholder": "選填詞彙或上下文（最多 2048 字元）",
        "response_format": "回應格式",
        "output_file": "另存路徑",
        "output_placeholder": "選填；若留白，可在轉錄完成後輸入",
        "transcribe": "轉錄檔案",
        "cancel": "取消請求",
        "save": "儲存轉錄",
        "clear": "清除轉錄",
        "phase_note": "[輸入方式] 可貼上音訊路徑或在本機錄音；即使未安裝 sounddevice 或沒有輸入裝置，檔案模式仍可使用。",
        "recorder_available": "[麥克風就緒] 16 kHz 單聲道 PCM · 最長 {seconds:g} 秒 · 私有暫存",
        "recorder_unavailable": "[僅檔案模式] 麥克風是選用功能，目前不可用：{reason}",
        "recorder_idle": "[麥克風待命] 準備好後即可開始錄音。",
        "recorder_starting": "[麥克風啟動中] 正在開啟預設輸入裝置…",
        "recorder_recording": "[錄音中] {elapsed:.1f} 秒 / {limit:g} 秒 · 緩衝 {buffer_kib:.0f} KiB",
        "recorder_stopping": "[麥克風停止中] 正在完成私有 WAV 暫存檔…",
        "recorder_recorded": "[錄音完成] 已錄製 {elapsed:.1f} 秒；按一次「轉錄檔案」即可送出。",
        "recorder_limit": "[已達上限] 已錄製 {elapsed:.1f} 秒，可立即轉錄。",
        "recorder_cancelled": "[錄音已取消] 暫存錄音已移除。",
        "recorder_error": "[麥克風錯誤] {error} 請排除裝置問題後重試；檔案輸入仍可使用。",
        "record_start": "開始錄音",
        "record_stop": "停止並使用",
        "record_cancel": "取消錄音",
        "diagnostics_empty": "健康、就緒狀態與模型結果會顯示於此。",
        "transcript_empty": "尚無轉錄內容。請選擇音訊檔並開始轉錄。",
        "status_ready": "[待命] 可執行伺服器診斷、檔案輸入或麥克風錄音。",
        "status_diagnostics": "[處理中] 正在檢查健康、就緒狀態與模型…",
        "status_transcribing": "[處理中] 正在上傳並轉錄 {name}…",
        "status_saving": "[處理中] 正在儲存轉錄…",
        "status_diagnostics_ok": "[完成] 已更新伺服器診斷。",
        "status_diagnostics_degraded": "[功能受限] 部分診斷失敗；已保留可用資訊。",
        "status_transcribed": "[完成] 轉錄完成，請檢查或儲存結果。",
        "status_saved": "[完成] 已以原子操作將轉錄儲存至 {path}。",
        "status_cancelled": "[已取消] 已安全取消目前請求。",
        "status_cleared": "[待命] 已清除轉錄內容。",
        "status_recorded": "[可轉錄] 已選取剛錄製的音訊，按 Ctrl+T 即可轉錄。",
        "status_recording_cancelled": "[已取消] 已停止錄音並移除暫存音訊。",
        "status_failed": "[錯誤] {error}",
        "error_audio_required": "開始轉錄前，請選擇存在的音訊檔。",
        "error_output_required": "儲存前請輸入輸出路徑。",
        "error_output_is_audio": "轉錄輸出不得覆寫來源音訊檔。",
        "error_no_transcript": "目前沒有可儲存的轉錄內容。",
        "error_request_active": "請先等待或取消目前請求。",
        "error_recorder_unavailable": "麥克風錄音目前不可用：{reason}",
        "error_output_private": "請選擇私有錄音暫存目錄以外的永久輸出路徑。",
        "diagnostic_health": "健康狀態",
        "diagnostic_ready": "就緒狀態",
        "diagnostic_models": "模型",
        "diagnostic_ok": "正常",
        "diagnostic_degraded": "功能受限",
        "diagnostic_error": "錯誤",
        "diagnostic_none": "未回報",
        "diagnostic_check_task_router_bound": "任務路由已綁定",
        "diagnostic_check_recognizer_process_alive": "辨識程序仍在執行",
        "diagnostic_check_ffmpeg_available": "FFmpeg 可用",
        "diagnostic_check_ok": "正常",
        "diagnostic_check_failed": "失敗",
        "locale_en": "English",
        "locale_zh": "繁體中文",
    },
}


def normalize_locale(value: str | None) -> Locale:
    """Normalize supported English and Traditional-Chinese locale aliases."""

    normalized = (value or DEFAULT_LOCALE).strip().replace("_", "-").lower()
    if normalized in {"en", "en-us", "en-gb"}:
        return "en"
    if normalized in {"zh", "zh-tw", "zh-hk", "zh-hant", "traditional"}:
        return "zh-Hant"
    raise ValueError(f"unsupported UI language: {value!r}; choose en or zh-Hant")


@dataclass(frozen=True)
class Translator:
    """Render messages from one validated catalog."""

    locale: Locale = DEFAULT_LOCALE

    def __post_init__(self) -> None:
        object.__setattr__(self, "locale", normalize_locale(self.locale))

    def text(self, key: str, **values: object) -> str:
        try:
            template = CATALOGS[self.locale][key]
        except KeyError as exc:
            raise KeyError(f"missing {self.locale} message: {key}") from exc
        return template.format(**values)

    __call__ = text
