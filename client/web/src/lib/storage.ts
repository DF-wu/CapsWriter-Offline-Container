import type { ApiSettings, TranscriptRecord } from "../types";

const SETTINGS_KEY = "capswriter.web.settings.v1";
const HISTORY_KEY = "capswriter.web.history.v1";
const HISTORY_LIMIT = 20;
const RESPONSE_FORMATS = new Set(["json", "text", "srt", "verbose_json", "vtt"]);
const SETTING_LIMITS = {
  baseUrl: 2048,
  apiKey: 4096,
  model: 128,
  language: 32,
  prompt: 16_384,
};
const HISTORY_FIELD_LIMITS = {
  id: 128,
  sourceName: 512,
  text: 200_000,
  raw: 500_000,
};

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
  const runtimeConfig = isRecord(runtime) ? runtime : {};
  return {
    baseUrl: stringSetting(runtimeConfig.baseUrl, BUILTIN_SETTINGS.baseUrl, SETTING_LIMITS.baseUrl),
    apiKey: stringSetting(runtimeConfig.apiKey, BUILTIN_SETTINGS.apiKey, SETTING_LIMITS.apiKey),
    model: stringSetting(runtimeConfig.model, BUILTIN_SETTINGS.model, SETTING_LIMITS.model),
    language: stringSetting(runtimeConfig.language, BUILTIN_SETTINGS.language, SETTING_LIMITS.language),
    prompt: stringSetting(runtimeConfig.prompt, BUILTIN_SETTINGS.prompt, SETTING_LIMITS.prompt),
    responseFormat: isResponseFormat(runtimeConfig.responseFormat)
      ? runtimeConfig.responseFormat
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

function stringSetting(value: unknown, fallback: string, maxChars: number): string {
  if (typeof value !== "string") return fallback;
  return value.length > maxChars ? value.slice(0, maxChars) : value;
}

function boundedRequiredString(value: unknown, maxChars: number): string | null {
  if (typeof value !== "string") return null;
  if (value.length > maxChars || !value.trim()) return null;
  return value;
}

function isFiniteNonNegativeNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

function isTranscriptRaw(value: unknown): value is TranscriptRecord["raw"] {
  if (typeof value === "string") return value.length <= HISTORY_FIELD_LIMITS.raw;
  if (!isRecord(value)) return false;
  try {
    return JSON.stringify(value).length <= HISTORY_FIELD_LIMITS.raw;
  } catch {
    return false;
  }
}

function normalizeHistoryRecord(value: unknown): TranscriptRecord | null {
  if (!isRecord(value)) return null;
  const id = boundedRequiredString(value.id, HISTORY_FIELD_LIMITS.id);
  if (!id) return null;
  if (typeof value.createdAt !== "string" || Number.isNaN(Date.parse(value.createdAt))) {
    return null;
  }
  const sourceName = boundedRequiredString(value.sourceName, HISTORY_FIELD_LIMITS.sourceName);
  if (!sourceName) return null;
  if (!isResponseFormat(value.format)) return null;
  if (typeof value.text !== "string" || value.text.length > HISTORY_FIELD_LIMITS.text) return null;
  if (value.durationSeconds !== null && !isFiniteNonNegativeNumber(value.durationSeconds)) {
    return null;
  }
  if (!isTranscriptRaw(value.raw)) return null;

  return {
    id,
    createdAt: value.createdAt,
    sourceName,
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
  return {
    baseUrl: stringSetting(settings.baseUrl, DEFAULT_SETTINGS.baseUrl, SETTING_LIMITS.baseUrl),
    model: stringSetting(settings.model, DEFAULT_SETTINGS.model, SETTING_LIMITS.model),
    language: stringSetting(settings.language, DEFAULT_SETTINGS.language, SETTING_LIMITS.language),
    prompt: stringSetting(settings.prompt, DEFAULT_SETTINGS.prompt, SETTING_LIMITS.prompt),
    responseFormat: isResponseFormat(settings.responseFormat)
      ? settings.responseFormat
      : DEFAULT_SETTINGS.responseFormat,
  };
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
    baseUrl: stringSetting(persisted.baseUrl, DEFAULT_SETTINGS.baseUrl, SETTING_LIMITS.baseUrl),
    model: stringSetting(persisted.model, DEFAULT_SETTINGS.model, SETTING_LIMITS.model),
    language: stringSetting(persisted.language, DEFAULT_SETTINGS.language, SETTING_LIMITS.language),
    prompt: stringSetting(persisted.prompt, DEFAULT_SETTINGS.prompt, SETTING_LIMITS.prompt),
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
