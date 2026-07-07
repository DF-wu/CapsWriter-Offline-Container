import type { ApiSettings, TranscriptRecord } from "../types";

const SETTINGS_KEY = "capswriter.web.settings.v1";
const HISTORY_KEY = "capswriter.web.history.v1";
const HISTORY_LIMIT = 20;
const RESPONSE_FORMATS = new Set(["json", "text", "srt", "verbose_json", "vtt"]);

const BUILTIN_SETTINGS: ApiSettings = {
  baseUrl: "http://localhost:6017",
  apiKey: "",
  model: "whisper-1",
  language: "",
  prompt: "",
  responseFormat: "verbose_json",
};

export function settingsWithRuntimeDefaults(
  runtime: Partial<ApiSettings> | undefined = window.__CAPSWRITER_WEB_CONFIG__,
): ApiSettings {
  const responseFormat = runtime?.responseFormat ?? BUILTIN_SETTINGS.responseFormat;
  return {
    ...BUILTIN_SETTINGS,
    ...runtime,
    responseFormat: RESPONSE_FORMATS.has(responseFormat)
      ? responseFormat
      : BUILTIN_SETTINGS.responseFormat,
  };
}

export const DEFAULT_SETTINGS: ApiSettings = settingsWithRuntimeDefaults();

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function isResponseFormat(value: unknown): value is ApiSettings["responseFormat"] {
  return typeof value === "string" && RESPONSE_FORMATS.has(value);
}

function stringSetting(value: unknown, fallback: string): string {
  return typeof value === "string" ? value : fallback;
}

function isFiniteNonNegativeNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

function isTranscriptRaw(value: unknown): value is TranscriptRecord["raw"] {
  return typeof value === "string" || isRecord(value);
}

function normalizeHistoryRecord(value: unknown): TranscriptRecord | null {
  if (!isRecord(value)) return null;
  if (typeof value.id !== "string" || !value.id.trim()) return null;
  if (typeof value.createdAt !== "string" || Number.isNaN(Date.parse(value.createdAt))) {
    return null;
  }
  if (typeof value.sourceName !== "string" || !value.sourceName.trim()) return null;
  if (!isResponseFormat(value.format)) return null;
  if (typeof value.text !== "string") return null;
  if (value.durationSeconds !== null && !isFiniteNonNegativeNumber(value.durationSeconds)) {
    return null;
  }
  if (!isTranscriptRaw(value.raw)) return null;

  return {
    id: value.id,
    createdAt: value.createdAt,
    sourceName: value.sourceName,
    durationSeconds: value.durationSeconds,
    format: value.format,
    text: value.text,
    raw: value.raw,
  };
}

function normalizeHistory(value: unknown): TranscriptRecord[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    const record = normalizeHistoryRecord(item);
    return record ? [record] : [];
  });
}

function readJsonRecord(key: string): Record<string, unknown> {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as unknown;
    return isRecord(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function settingsForPersistence(settings: ApiSettings): Omit<ApiSettings, "apiKey"> {
  const { apiKey: _apiKey, ...persisted } = settings;
  return persisted;
}

function writeJson(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Persistence is best-effort; browser privacy/quota settings must not break the app.
  }
}

function removeStoredValue(key: string): void {
  try {
    localStorage.removeItem(key);
  } catch {
    // Ignore blocked storage during cleanup paths.
  }
}

export function loadSettings(): ApiSettings {
  const persisted = readJsonRecord(SETTINGS_KEY);
  return {
    ...DEFAULT_SETTINGS,
    apiKey: "",
    baseUrl: stringSetting(persisted.baseUrl, DEFAULT_SETTINGS.baseUrl),
    model: stringSetting(persisted.model, DEFAULT_SETTINGS.model),
    language: stringSetting(persisted.language, DEFAULT_SETTINGS.language),
    prompt: stringSetting(persisted.prompt, DEFAULT_SETTINGS.prompt),
    responseFormat: isResponseFormat(persisted.responseFormat)
      ? persisted.responseFormat
      : DEFAULT_SETTINGS.responseFormat,
  };
}

export function saveSettings(settings: ApiSettings): void {
  writeJson(SETTINGS_KEY, settingsForPersistence(settings));
}

export function loadHistory(): TranscriptRecord[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    return normalizeHistory(parsed).slice(0, HISTORY_LIMIT);
  } catch {
    return [];
  }
}

export function saveHistory(history: TranscriptRecord[]): void {
  writeJson(HISTORY_KEY, normalizeHistory(history).slice(0, HISTORY_LIMIT));
}

export function addHistory(record: TranscriptRecord): TranscriptRecord[] {
  const next = [record, ...loadHistory()].slice(0, HISTORY_LIMIT);
  saveHistory(next);
  return next;
}

export function clearHistory(): void {
  removeStoredValue(HISTORY_KEY);
}
