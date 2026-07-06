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

function readJson<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return fallback;
    return { ...fallback, ...JSON.parse(raw) };
  } catch {
    return fallback;
  }
}

function settingsForPersistence(settings: ApiSettings): Omit<ApiSettings, "apiKey"> {
  const { apiKey: _apiKey, ...persisted } = settings;
  return persisted;
}

export function loadSettings(): ApiSettings {
  const persisted = readJson<Partial<ApiSettings>>(SETTINGS_KEY, {});
  const { apiKey: _apiKey, ...safePersisted } = persisted;
  return { ...DEFAULT_SETTINGS, ...safePersisted };
}

export function saveSettings(settings: ApiSettings): void {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(settingsForPersistence(settings)));
}

export function loadHistory(): TranscriptRecord[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    return raw ? (JSON.parse(raw) as TranscriptRecord[]) : [];
  } catch {
    return [];
  }
}

export function saveHistory(history: TranscriptRecord[]): void {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, HISTORY_LIMIT)));
}

export function addHistory(record: TranscriptRecord): TranscriptRecord[] {
  const next = [record, ...loadHistory()].slice(0, HISTORY_LIMIT);
  saveHistory(next);
  return next;
}

export function clearHistory(): void {
  localStorage.removeItem(HISTORY_KEY);
}
