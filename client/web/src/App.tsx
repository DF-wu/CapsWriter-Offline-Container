import { type ChangeEvent, type DragEvent, useEffect, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Copy,
  Download,
  FileAudio,
  Mic,
  Pause,
  Play,
  RefreshCw,
  Settings2,
  Square,
  Trash2,
  Upload,
  Volume2,
  XCircle,
} from "lucide-react";
import { fetchHealth, fetchModels, fetchReadiness, transcribeAudio } from "./api/capswriter";
import {
  chooseRecorderMimeType,
  extensionForMimeType,
  fileToBrowserAudio,
  formatDuration,
  revokeAudio,
  type BrowserAudio,
} from "./lib/audio";
import { downloadText, extensionForFormat, serialiseResult, timestampSlug } from "./lib/export";
import {
  DEFAULT_SETTINGS,
  WEB_SETTING_LIMITS,
  addHistory,
  clearHistory,
  loadHistory,
  loadSettings,
  saveHistory,
  saveSettings,
} from "./lib/storage";
import { loadVoices, speakText } from "./lib/speech";
import type { ApiSettings, HealthResponse, ReadinessResponse, ResponseFormat, TranscriptRecord, TranscriptionResult } from "./types";

type StatusKind = "idle" | "working" | "ok" | "degraded" | "error";
type SpeechState = "idle" | "speaking" | "paused";

function iconStatus(kind: StatusKind) {
  if (kind === "ok") return <CheckCircle2 size={18} aria-hidden="true" />;
  if (kind === "degraded") return <AlertTriangle size={18} aria-hidden="true" />;
  if (kind === "error") return <XCircle size={18} aria-hidden="true" />;
  if (kind === "working") return <RefreshCw size={18} aria-hidden="true" className="spin" />;
  return <Activity size={18} aria-hidden="true" />;
}

function readinessValue(value: boolean | undefined): string {
  if (value === true) return "ok";
  if (value === false) return "fail";
  return "-";
}

function readinessClass(value: boolean | undefined): string {
  if (value === true) return "meta-state ok";
  if (value === false) return "meta-state degraded";
  return "meta-state";
}

function diagnosticError(label: string, reason: unknown): string {
  const message = reason instanceof Error ? reason.message : "failed";
  return `${label}: ${message}`;
}

function makeRecord(
  result: TranscriptionResult,
  audio: BrowserAudio,
): TranscriptRecord {
  return {
    id: crypto.randomUUID?.() ?? `${Date.now()}`,
    createdAt: new Date().toISOString(),
    sourceName: audio.name,
    durationSeconds: audio.durationSeconds,
    format: result.format,
    text: result.text,
    raw: result.raw,
  };
}

export default function App() {
  const [settings, setSettings] = useState<ApiSettings>(() => loadSettings());
  const [statusKind, setStatusKind] = useState<StatusKind>("idle");
  const [statusText, setStatusText] = useState("未連線");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [readiness, setReadiness] = useState<ReadinessResponse | null>(null);
  const [models, setModels] = useState<string[]>([]);
  const [currentAudio, setCurrentAudio] = useState<BrowserAudio | null>(null);
  const [transcript, setTranscript] = useState<TranscriptionResult | null>(null);
  const [history, setHistory] = useState<TranscriptRecord[]>(() => loadHistory());
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [ttsText, setTtsText] = useState("");
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [voiceURI, setVoiceURI] = useState("");
  const [rate, setRate] = useState(1);
  const [pitch, setPitch] = useState(1);
  const [speechState, setSpeechState] = useState<SpeechState>("idle");
  const ttsAvailable = voices.length > 0;

  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const startedAtRef = useRef<number>(0);
  const abortRef = useRef<AbortController | null>(null);
  const currentAudioRef = useRef<BrowserAudio | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const dragDepthRef = useRef(0);
  const transcriptionRunRef = useRef(0);

  useEffect(() => saveSettings(settings), [settings]);

  useEffect(() => {
    currentAudioRef.current = currentAudio;
  }, [currentAudio]);

  useEffect(() => {
    let active = true;
    loadVoices().then((items) => {
      if (!active) return;
      setVoices(items);
      setVoiceURI((current) => current || items[0]?.voiceURI || "");
    });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    return () => {
      if (timerRef.current !== null) {
        window.clearInterval(timerRef.current);
        timerRef.current = null;
      }
      const recorder = recorderRef.current;
      if (recorder && recorder.state !== "inactive") {
        recorder.ondataavailable = null;
        recorder.onstop = null;
        recorder.stop();
      }
      recorderRef.current = null;
      revokeAudio(currentAudioRef.current);
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      abortRef.current?.abort();
      if ("speechSynthesis" in window) {
        window.speechSynthesis.cancel();
      }
    };
  }, []);

  const updateSettings = <K extends keyof ApiSettings>(key: K, value: ApiSettings[K]) => {
    setSettings((current) => ({ ...current, [key]: value }));
  };

  const setAudio = (audio: BrowserAudio | null) => {
    transcriptionRunRef.current += 1;
    setCurrentAudio((previous) => {
      revokeAudio(previous);
      return audio;
    });
    setTranscript(null);
    setStatusKind(audio ? "ok" : "idle");
    setStatusText(audio ? `已載入 ${audio.name}` : "未載入音訊");
  };

  const checkServer = async () => {
    setStatusKind("working");
    setStatusText("檢查服務");
    const [healthResult, readinessResult, modelsResult] = await Promise.allSettled([
      fetchHealth(settings),
      fetchReadiness(settings),
      fetchModels(settings),
    ]);

    const failures: string[] = [];
    const nextHealth = healthResult.status === "fulfilled" ? healthResult.value : null;
    const nextReadiness = readinessResult.status === "fulfilled" ? readinessResult.value : null;

    if (healthResult.status === "fulfilled") {
      setHealth(healthResult.value);
    } else {
      setHealth(null);
      failures.push(diagnosticError("Health", healthResult.reason));
    }

    if (readinessResult.status === "fulfilled") {
      setReadiness(readinessResult.value);
    } else {
      setReadiness(null);
      failures.push(diagnosticError("Ready", readinessResult.reason));
    }

    if (modelsResult.status === "fulfilled") {
      setModels(modelsResult.value.data.map((item) => item.id));
    } else {
      setModels([]);
      failures.push(diagnosticError("Models", modelsResult.reason));
    }

    if (failures.length > 0) {
      const hasPartialDiagnostics = Boolean(nextHealth || nextReadiness);
      setStatusKind(hasPartialDiagnostics ? "degraded" : "error");
      setStatusText(`服務檢查部分失敗：${failures[0]}`);
    } else if (nextReadiness?.status === "ok") {
      setStatusKind("ok");
      setStatusText(`服務正常：${nextHealth?.model ?? "unknown"} v${nextHealth?.version ?? "unknown"}`);
    } else if (nextReadiness) {
      setStatusKind("degraded");
      setStatusText(`服務降級：${nextReadiness.status}`);
    } else {
      setStatusKind("error");
      setStatusText("服務檢查失敗");
    }
  };

  const startRecording = async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setStatusKind("error");
      setStatusText("此瀏覽器不支援麥克風錄音");
      return;
    }
    let stream: MediaStream | null = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      const mimeType = chooseRecorderMimeType();
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      const recordingStream = stream;
      streamRef.current = stream;
      recorderRef.current = recorder;
      chunksRef.current = [];
      startedAtRef.current = Date.now();

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };
      recorder.onstop = () => {
        const finalMimeType = recorder.mimeType || mimeType || "audio/webm";
        const blob = new Blob(chunksRef.current, { type: finalMimeType });
        const durationSeconds = Math.max(0, (Date.now() - startedAtRef.current) / 1000);
        const ext = extensionForMimeType(finalMimeType);
        setAudio({
          blob,
          name: `recording-${timestampSlug()}.${ext}`,
          objectUrl: URL.createObjectURL(blob),
          durationSeconds,
        });
        recordingStream.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
        setRecordingSeconds(Math.round(durationSeconds));
      };

      recorder.start(250);
      setIsRecording(true);
      setRecordingSeconds(0);
      setStatusKind("working");
      setStatusText("錄音中");
      timerRef.current = window.setInterval(() => {
        setRecordingSeconds(Math.floor((Date.now() - startedAtRef.current) / 1000));
      }, 250);
    } catch (error) {
      if (timerRef.current !== null) {
        window.clearInterval(timerRef.current);
        timerRef.current = null;
      }
      recorderRef.current = null;
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
      if (streamRef.current === stream) {
        streamRef.current = null;
      }
      setIsRecording(false);
      setStatusKind("error");
      setStatusText(error instanceof Error ? error.message : "無法啟動錄音");
    }
  };

  const stopRecording = () => {
    if (timerRef.current) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setIsRecording(false);
    const recorder = recorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
    }
  };

  const handleFile = (file: File | null) => {
    if (!file) return;
    if (isTranscribing) {
      setStatusKind("working");
      setStatusText("轉錄中，請先取消再更換音訊");
      return;
    }
    if (!file.type.startsWith("audio/") && !file.name.match(/\.(wav|mp3|m4a|flac|ogg|webm)$/i)) {
      setStatusKind("error");
      setStatusText("請選擇音訊檔");
      return;
    }
    setAudio(fileToBrowserAudio(file));
  };

  const handleFileInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    const files = event.currentTarget.files;
    handleFile(files?.item?.(0) ?? files?.[0] ?? null);
    event.currentTarget.value = "";
  };

  const handleAudioDragEnter = (event: DragEvent<HTMLButtonElement>) => {
    event.preventDefault();
    if (isTranscribing) return;
    dragDepthRef.current += 1;
    setIsDragging(true);
  };

  const handleAudioDragLeave = (event: DragEvent<HTMLButtonElement>) => {
    event.preventDefault();
    if (isTranscribing) return;
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);
    if (dragDepthRef.current === 0) {
      setIsDragging(false);
    }
  };

  const handleAudioDrop = (event: DragEvent<HTMLButtonElement>) => {
    event.preventDefault();
    dragDepthRef.current = 0;
    setIsDragging(false);
    if (isTranscribing) return;
    handleFile(event.dataTransfer.files.item(0));
  };

  const runTranscription = async () => {
    if (!currentAudio) {
      setStatusKind("error");
      setStatusText("未載入音訊");
      return;
    }
    abortRef.current?.abort();
    const aborter = new AbortController();
    abortRef.current = aborter;
    const runId = transcriptionRunRef.current + 1;
    transcriptionRunRef.current = runId;
    const audio = currentAudio;
    setIsTranscribing(true);
    setStatusKind("working");
    setStatusText("轉錄中");
    try {
      const result = await transcribeAudio(
        audio.blob,
        audio.name,
        settings,
        aborter.signal,
      );
      if (runId !== transcriptionRunRef.current) return;
      setTranscript(result);
      setTtsText(result.text);
      setHistory(addHistory(makeRecord(result, audio)));
      setStatusKind("ok");
      setStatusText(`完成：${result.text.length} 字`);
    } catch (error) {
      if (runId !== transcriptionRunRef.current) return;
      if (error instanceof DOMException && error.name === "AbortError") {
        setStatusKind("idle");
        setStatusText("已取消");
      } else {
        setStatusKind("error");
        setStatusText(error instanceof Error ? error.message : "轉錄失敗");
      }
    } finally {
      if (runId === transcriptionRunRef.current) {
        abortRef.current = null;
        setIsTranscribing(false);
      }
    }
  };

  const cancelTranscription = () => {
    transcriptionRunRef.current += 1;
    abortRef.current?.abort();
    abortRef.current = null;
    setIsTranscribing(false);
    setStatusKind("idle");
    setStatusText("已取消");
  };

  const copyTranscript = async () => {
    if (!transcript) return;
    if (!navigator.clipboard?.writeText) {
      setStatusKind("error");
      setStatusText("此瀏覽器不支援剪貼簿");
      return;
    }
    try {
      await navigator.clipboard.writeText(transcript.text);
      setStatusKind("ok");
      setStatusText("已複製");
    } catch (error) {
      setStatusKind("error");
      setStatusText(error instanceof Error ? error.message : "複製失敗");
    }
  };

  const downloadTranscript = () => {
    if (!transcript) return;
    const extension = extensionForFormat(transcript.format);
    downloadText(`capswriter-${timestampSlug()}.${extension}`, serialiseResult(transcript));
  };

  const useHistoryRecord = (record: TranscriptRecord) => {
    setTranscript({
      text: record.text,
      format: record.format,
      raw: record.raw,
      contentType: "",
    });
    setTtsText(record.text);
    setStatusKind("ok");
    setStatusText(`已載入歷史：${record.sourceName}`);
  };

  const clearAllHistory = () => {
    clearHistory();
    setHistory([]);
    setStatusKind("ok");
    setStatusText("歷史已清除");
  };

  const startSpeech = () => {
    speakText({
      text: ttsText,
      voiceURI,
      rate,
      pitch,
      onEnd: () => setSpeechState("idle"),
      onError: (message) => {
        setSpeechState("idle");
        setStatusKind("error");
        setStatusText(message);
      },
    });
    setSpeechState("speaking");
  };

  const pauseSpeech = () => {
    window.speechSynthesis.pause();
    setSpeechState("paused");
  };

  const resumeSpeech = () => {
    window.speechSynthesis.resume();
    setSpeechState("speaking");
  };

  const stopSpeech = () => {
    window.speechSynthesis.cancel();
    setSpeechState("idle");
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">CapsWriter Offline</p>
          <h1>Web Console</h1>
        </div>
        <div className={`status-pill ${statusKind}`} aria-live="polite">
          {iconStatus(statusKind)}
          <span>{statusText}</span>
        </div>
      </header>

      <main className="workspace">
        <section className="panel controls-panel" aria-labelledby="connection-title">
          <div className="panel-heading">
            <Settings2 size={20} aria-hidden="true" />
            <h2 id="connection-title">連線</h2>
          </div>

          <label className="field">
            <span>API root</span>
            <input
              value={settings.baseUrl}
              onChange={(event) => updateSettings("baseUrl", event.target.value)}
              placeholder={DEFAULT_SETTINGS.baseUrl}
              inputMode="url"
              maxLength={WEB_SETTING_LIMITS.baseUrl}
            />
          </label>
          <label className="field">
            <span>API key</span>
            <input
              value={settings.apiKey}
              onChange={(event) => updateSettings("apiKey", event.target.value)}
              type="password"
              autoComplete="off"
              maxLength={WEB_SETTING_LIMITS.apiKey}
            />
          </label>
          <div className="field-row">
            <label className="field">
              <span>格式</span>
              <select
                value={settings.responseFormat}
                onChange={(event) => updateSettings("responseFormat", event.target.value as ResponseFormat)}
              >
                <option value="verbose_json">verbose_json</option>
                <option value="json">json</option>
                <option value="text">text</option>
                <option value="srt">srt</option>
                <option value="vtt">vtt</option>
              </select>
            </label>
            <label className="field">
              <span>語言</span>
              <input
                value={settings.language}
                onChange={(event) => updateSettings("language", event.target.value)}
                placeholder="auto"
                maxLength={WEB_SETTING_LIMITS.language}
              />
            </label>
          </div>
          <label className="field">
            <span>模型</span>
            <input
              value={settings.model}
              onChange={(event) => updateSettings("model", event.target.value)}
              placeholder="whisper-1"
              maxLength={WEB_SETTING_LIMITS.model}
            />
          </label>
          <label className="field">
            <span>Prompt</span>
            <textarea
              value={settings.prompt}
              onChange={(event) => updateSettings("prompt", event.target.value)}
              rows={3}
              maxLength={WEB_SETTING_LIMITS.prompt}
            />
          </label>
          <button className="secondary-action" type="button" onClick={checkServer}>
            <RefreshCw size={18} aria-hidden="true" />
            檢查服務
          </button>

          <dl className="server-meta">
            <div>
              <dt>Health</dt>
              <dd>{health ? health.status : "-"}</dd>
            </div>
            <div>
              <dt>Ready</dt>
              <dd className={`meta-state ${readiness?.status === "ok" ? "ok" : readiness ? "degraded" : ""}`}>
                {readiness ? readiness.status : "-"}
              </dd>
            </div>
            <div>
              <dt>Router</dt>
              <dd className={readinessClass(readiness?.checks.task_router_bound)}>
                {readinessValue(readiness?.checks.task_router_bound)}
              </dd>
            </div>
            <div>
              <dt>FFmpeg</dt>
              <dd className={readinessClass(readiness?.checks.ffmpeg_available)}>
                {readinessValue(readiness?.checks.ffmpeg_available)}
              </dd>
            </div>
            <div>
              <dt>Server model</dt>
              <dd>{health ? health.model : "-"}</dd>
            </div>
            <div>
              <dt>Models</dt>
              <dd>{models.length ? models.join(", ") : "-"}</dd>
            </div>
            <div>
              <dt>Auth</dt>
              <dd>{readiness ? (readiness.config.auth_enabled ? "enabled" : "off") : "-"}</dd>
            </div>
            <div>
              <dt>Limits</dt>
              <dd>
                {readiness
                  ? `${readiness.config.max_upload_mb} MB / ${readiness.config.max_concurrent_requests} slots`
                  : "-"}
              </dd>
            </div>
          </dl>
        </section>

        <section className="panel input-panel" aria-labelledby="audio-title">
          <div className="panel-heading">
            <FileAudio size={20} aria-hidden="true" />
            <h2 id="audio-title">音訊</h2>
          </div>

          <div className="recording-controls">
            <button
              className="primary-action"
              type="button"
              onClick={isRecording ? stopRecording : startRecording}
              disabled={isTranscribing}
            >
              {isRecording ? <Square size={18} aria-hidden="true" /> : <Mic size={18} aria-hidden="true" />}
              {isRecording ? "停止" : "錄音"}
            </button>
            <span className="timer" aria-label="錄音時間">
              {formatDuration(recordingSeconds)}
            </span>
          </div>

          <button
            type="button"
            className={`drop-zone ${isDragging ? "dragging" : ""}`}
            onClick={() => fileInputRef.current?.click()}
            disabled={isTranscribing}
            onDragEnter={handleAudioDragEnter}
            onDragOver={(event) => {
              event.preventDefault();
              event.dataTransfer.dropEffect = "copy";
            }}
            onDragLeave={handleAudioDragLeave}
            onDrop={handleAudioDrop}
          >
            <Upload size={22} aria-hidden="true" />
            <span>{currentAudio ? currentAudio.name : "選擇音訊檔"}</span>
          </button>
          <input
            ref={fileInputRef}
            className="file-input"
            type="file"
            accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg,.webm"
            tabIndex={-1}
            disabled={isTranscribing}
            onChange={handleFileInputChange}
          />

          {currentAudio ? (
            <div className="audio-preview">
              <audio
                controls
                src={currentAudio.objectUrl}
                onLoadedMetadata={(event) => {
                  const duration = event.currentTarget.duration;
                  if (Number.isFinite(duration)) {
                    setCurrentAudio((audio) =>
                      audio ? { ...audio, durationSeconds: duration } : audio,
                    );
                  }
                }}
              />
              <span>{formatDuration(currentAudio.durationSeconds)}</span>
            </div>
          ) : null}

          <div className="action-row">
            <button
              className="primary-action"
              type="button"
              onClick={runTranscription}
              disabled={!currentAudio || isTranscribing}
            >
              <Play size={18} aria-hidden="true" />
              轉錄
            </button>
            <button
              className="secondary-action"
              type="button"
              onClick={cancelTranscription}
              disabled={!isTranscribing}
            >
              <Square size={18} aria-hidden="true" />
              取消
            </button>
          </div>
        </section>

        <section className="panel transcript-panel" aria-labelledby="transcript-title">
          <div className="panel-heading split">
            <div className="heading-inline">
              <Activity size={20} aria-hidden="true" />
              <h2 id="transcript-title">轉錄</h2>
            </div>
            <div className="toolbar">
              <button type="button" onClick={copyTranscript} disabled={!transcript} title="複製" aria-label="複製">
                <Copy size={18} aria-hidden="true" />
              </button>
              <button type="button" onClick={downloadTranscript} disabled={!transcript} title="下載" aria-label="下載">
                <Download size={18} aria-hidden="true" />
              </button>
            </div>
          </div>
          <textarea
            className="transcript-output"
            value={transcript?.text ?? ""}
            onChange={(event) => {
              const text = event.target.value;
              setTranscript((current) =>
                current ? { ...current, text } : { text, format: "text", raw: text, contentType: "text/plain" },
              );
              setTtsText(text);
            }}
            rows={12}
          />
          <div className="result-meta">
            <span>{transcript ? transcript.format : "no result"}</span>
            <span>{transcript ? `${transcript.text.length} chars` : "0 chars"}</span>
          </div>
        </section>

        <section className="panel tts-panel" aria-labelledby="tts-title">
          <div className="panel-heading">
            <Volume2 size={20} aria-hidden="true" />
            <h2 id="tts-title">TTS</h2>
          </div>
          <label className="field">
            <span>文字</span>
            <textarea
              value={ttsText}
              onChange={(event) => setTtsText(event.target.value)}
              rows={6}
            />
          </label>
          <label className="field">
            <span>聲音</span>
            <select value={voiceURI} onChange={(event) => setVoiceURI(event.target.value)}>
              {voices.length === 0 ? <option value="">no voices</option> : null}
              {voices.map((voice) => (
                <option key={voice.voiceURI} value={voice.voiceURI}>
                  {voice.name} ({voice.lang})
                </option>
              ))}
            </select>
          </label>
          <div className="field-row">
            <label className="field">
              <span>速度 {rate.toFixed(1)}</span>
              <input
                type="range"
                min="0.5"
                max="2"
                step="0.1"
                value={rate}
                onChange={(event) => setRate(Number(event.target.value))}
              />
            </label>
            <label className="field">
              <span>音高 {pitch.toFixed(1)}</span>
              <input
                type="range"
                min="0.5"
                max="2"
                step="0.1"
                value={pitch}
                onChange={(event) => setPitch(Number(event.target.value))}
              />
            </label>
          </div>
          <div className="action-row">
            <button
              className="primary-action"
              type="button"
              onClick={speechState === "paused" ? resumeSpeech : startSpeech}
              disabled={!ttsText.trim() || !ttsAvailable}
            >
              <Play size={18} aria-hidden="true" />
              {speechState === "paused" ? "繼續" : "播放"}
            </button>
            <button
              className="secondary-action"
              type="button"
              onClick={pauseSpeech}
              disabled={speechState !== "speaking"}
            >
              <Pause size={18} aria-hidden="true" />
              暫停
            </button>
            <button
              className="secondary-action"
              type="button"
              onClick={stopSpeech}
              disabled={speechState === "idle"}
            >
              <Square size={18} aria-hidden="true" />
              停止
            </button>
          </div>
        </section>

        <section className="panel history-panel" aria-labelledby="history-title">
          <div className="panel-heading split">
            <div className="heading-inline">
              <Download size={20} aria-hidden="true" />
              <h2 id="history-title">歷史</h2>
            </div>
            <button type="button" onClick={clearAllHistory} disabled={history.length === 0} title="清除" aria-label="清除">
              <Trash2 size={18} aria-hidden="true" />
            </button>
          </div>
          <div className="history-list">
            {history.length === 0 ? (
              <p className="empty">No records</p>
            ) : (
              history.map((record) => (
                <article className="history-item" key={record.id}>
                  <button type="button" onClick={() => useHistoryRecord(record)}>
                    <span>{record.sourceName}</span>
                    <small>{new Date(record.createdAt).toLocaleString()}</small>
                  </button>
                  <button
                    type="button"
                    title="下載"
                    aria-label={`下載 ${record.sourceName}`}
                    onClick={() => {
                      saveHistory(history);
                      downloadText(
                        `capswriter-${record.id}.${extensionForFormat(record.format)}`,
                        typeof record.raw === "string" ? record.raw : JSON.stringify(record.raw, null, 2),
                      );
                    }}
                  >
                    <Download size={16} aria-hidden="true" />
                  </button>
                </article>
              ))
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
