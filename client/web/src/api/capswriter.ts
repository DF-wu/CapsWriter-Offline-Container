import type {
  ApiSettings,
  HealthResponse,
  ModelListResponse,
  ReadinessResponse,
  ResponseFormat,
  TranscriptionResult,
  VerboseTranscription,
} from "../types";

const MAX_ERROR_BODY_CHARS = 500;
const DIAGNOSTIC_TIMEOUT_MS = 10_000;
const DEFAULT_API_ROOT = "http://localhost:6017";

function stripOpenAiVersionPath(pathname: string): string {
  const trimmed = pathname.replace(/\/+$/g, "");
  if (!trimmed || trimmed === "/") return "";
  if (trimmed === "/v1") return "";
  if (trimmed.endsWith("/v1")) return trimmed.slice(0, -3).replace(/\/+$/g, "");
  return trimmed;
}

export function normalizeApiRoot(baseUrl: string): string {
  const trimmed = baseUrl.trim();
  if (!trimmed) {
    return DEFAULT_API_ROOT;
  }
  let url: URL;
  try {
    url = new URL(trimmed);
  } catch {
    throw new Error("API root must be an absolute http:// or https:// URL");
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error("API root must use http:// or https://");
  }
  if (url.username || url.password) {
    throw new Error("API root must not include username or password");
  }
  if (url.search || url.hash) {
    throw new Error("API root must not include query or fragment");
  }
  url.pathname = stripOpenAiVersionPath(url.pathname);
  return url.toString().replace(/\/$/, "");
}

function requestHeaders(settings: Pick<ApiSettings, "apiKey">): HeadersInit {
  return settings.apiKey.trim()
    ? { Authorization: `Bearer ${settings.apiKey.trim()}` }
    : {};
}

function compactErrorText(value: string): string {
  const text = value.trim().replace(/\s+/g, " ");
  if (text.length > MAX_ERROR_BODY_CHARS) {
    return `${text.slice(0, MAX_ERROR_BODY_CHARS).trimEnd()}...`;
  }
  return text;
}

export function apiErrorMessage(body: string): string {
  const text = body.trim();
  if (!text) return "";
  try {
    const payload = JSON.parse(text) as unknown;
    if (payload && typeof payload === "object") {
      const record = payload as Record<string, unknown>;
      const errorPayload = record.error;
      if (errorPayload && typeof errorPayload === "object") {
        const message = (errorPayload as Record<string, unknown>).message;
        if (message) return compactErrorText(String(message));
      }
      if (record.detail) return compactErrorText(String(record.detail));
    }
  } catch {
    return compactErrorText(text);
  }
  return compactErrorText(text);
}

function parseJsonBody<T>(body: string, path: string, status: number): T {
  try {
    return JSON.parse(body) as T;
  } catch {
    const message = apiErrorMessage(body);
    throw new Error(
      `HTTP ${status}: Expected JSON response from ${path}${message ? `: ${message}` : ""}`,
    );
  }
}

async function readJsonResponse<T>(response: Response, path: string): Promise<T> {
  return parseJsonBody<T>(await response.text(), path, response.status);
}

async function fetchWithTimeout(
  input: RequestInfo | URL,
  init: RequestInit,
  timeoutMs: number,
): Promise<Response> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(input, { ...init, signal: controller.signal });
  } catch (error) {
    if (controller.signal.aborted) {
      throw new Error(`Request timed out after ${Math.round(timeoutMs / 1000)}s`);
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}

async function checkedFetch(
  input: RequestInfo | URL,
  init?: RequestInit,
  timeoutMs?: number,
): Promise<Response> {
  const response = timeoutMs
    ? await fetchWithTimeout(input, init ?? {}, timeoutMs)
    : await fetch(input, init);
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    const message = apiErrorMessage(detail);
    throw new Error(`HTTP ${response.status}${message ? `: ${message}` : ""}`);
  }
  return response;
}

export async function fetchHealth(settings: ApiSettings): Promise<HealthResponse> {
  const root = normalizeApiRoot(settings.baseUrl);
  const response = await checkedFetch(
    `${root}/health`,
    {
      headers: requestHeaders(settings),
    },
    DIAGNOSTIC_TIMEOUT_MS,
  );
  return readJsonResponse<HealthResponse>(response, "/health");
}

export async function fetchReadiness(settings: ApiSettings): Promise<ReadinessResponse> {
  const root = normalizeApiRoot(settings.baseUrl);
  const response = await fetchWithTimeout(
    `${root}/ready`,
    {
      headers: requestHeaders(settings),
    },
    DIAGNOSTIC_TIMEOUT_MS,
  );
  const body = await response.text();
  if (response.ok || response.status === 503) {
    return parseJsonBody<ReadinessResponse>(body, "/ready", response.status);
  }
  const message = apiErrorMessage(body);
  throw new Error(`HTTP ${response.status}${message ? `: ${message}` : ""}`);
}

export async function fetchModels(settings: ApiSettings): Promise<ModelListResponse> {
  const root = normalizeApiRoot(settings.baseUrl);
  const response = await checkedFetch(
    `${root}/v1/models`,
    {
      headers: requestHeaders(settings),
    },
    DIAGNOSTIC_TIMEOUT_MS,
  );
  return readJsonResponse<ModelListResponse>(response, "/v1/models");
}

export async function parseTranscriptionResponse(
  response: Response,
  format: ResponseFormat,
): Promise<TranscriptionResult> {
  const contentType = response.headers.get("Content-Type") ?? "";
  if (format === "text" || format === "srt" || format === "vtt") {
    const text = await response.text();
    return { text, format, raw: text, contentType };
  }

  const payload = parseJsonBody<VerboseTranscription | { text?: string }>(
    await response.text(),
    "/v1/audio/transcriptions",
    response.status,
  );
  return {
    text: payload.text ?? "",
    format,
    raw: payload,
    contentType,
  };
}

export async function transcribeAudio(
  audio: Blob,
  filename: string,
  settings: ApiSettings,
  signal?: AbortSignal,
): Promise<TranscriptionResult> {
  const root = normalizeApiRoot(settings.baseUrl);
  const body = new FormData();
  body.append("file", audio, filename);
  body.append("model", settings.model || "whisper-1");
  body.append("response_format", settings.responseFormat);
  if (settings.language.trim()) {
    body.append("language", settings.language.trim());
  }
  if (settings.prompt.trim()) {
    body.append("prompt", settings.prompt.trim());
  }

  const response = await checkedFetch(`${root}/v1/audio/transcriptions`, {
    method: "POST",
    headers: requestHeaders(settings),
    body,
    signal,
  });
  return parseTranscriptionResponse(response, settings.responseFormat);
}
