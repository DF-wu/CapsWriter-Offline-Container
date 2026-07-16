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
const MAX_ERROR_SOURCE_SCAN_CHARS = MAX_ERROR_BODY_CHARS * 4;
export const MAX_RESPONSE_BODY_BYTES = 16 * 1024 * 1024;
const DIAGNOSTIC_TIMEOUT_MS = 10_000;
export const TRANSCRIPTION_TIMEOUT_MS = 600_000;
const DEFAULT_API_ROOT = "http://localhost:6017";
const REDACTED_SECRET = "[REDACTED]";
const UNSAFE_ERROR_CHARACTERS =
  /[\p{Cc}\p{Cf}\u2028\u2029]/gu;

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

function redactApiKey(value: string, apiKey: string): string {
  const secret = apiKey.trim();
  return secret ? value.split(secret).join(REDACTED_SECRET) : value;
}

function redactTruncatedSecretPrefix(value: string, secret: string): string {
  if (!secret || !value) return value;
  if (value.endsWith(secret)) return value;
  const maxPrefixLength = Math.min(secret.length - 1, value.length);
  for (let length = maxPrefixLength; length > 0; length -= 1) {
    if (value.endsWith(secret.slice(0, length))) {
      return `${value.slice(0, -length)}${REDACTED_SECRET}`;
    }
  }
  return value;
}

function compactErrorText(value: string, apiKey = ""): string {
  const secret = apiKey.trim();
  // Keep peer-controlled work bounded while retaining enough lookahead to
  // redact a key that starts near the visible preview boundary.
  const sourceLimit = MAX_ERROR_SOURCE_SCAN_CHARS + secret.length;
  const sourceWasTruncated = value.length > sourceLimit;
  const boundedSource = redactTruncatedSecretPrefix(
    value.slice(0, sourceLimit),
    sourceWasTruncated ? secret : "",
  );
  const text = redactApiKey(boundedSource, secret)
    .replace(UNSAFE_ERROR_CHARACTERS, " ")
    .trim()
    .replace(/\s+/gu, " ");
  if (sourceWasTruncated || text.length > MAX_ERROR_BODY_CHARS) {
    return `${text.slice(0, MAX_ERROR_BODY_CHARS).trimEnd()}...`;
  }
  return text;
}

function sanitizedAbortError(error: DOMException, apiKey: string): DOMException {
  return new DOMException(
    safeErrorMessage(error, apiKey, "Request aborted"),
    "AbortError",
  );
}

export function safeErrorMessage(
  reason: unknown,
  apiKey = "",
  fallback = "Request failed",
): string {
  let value = fallback;
  if (reason instanceof DOMException) {
    value = reason.message || reason.name || fallback;
  } else if (reason instanceof Error) {
    value = reason.message || reason.name || fallback;
  } else if (typeof reason === "string") {
    value = reason;
  } else if (reason !== null && reason !== undefined) {
    try {
      value = String(reason);
    } catch {
      value = fallback;
    }
  }
  return compactErrorText(value, apiKey) || compactErrorText(fallback, apiKey);
}

export function apiErrorMessage(body: string, apiKey = ""): string {
  const text = body.trim();
  if (!text) return "";
  try {
    const payload = JSON.parse(text) as unknown;
    if (payload && typeof payload === "object") {
      const record = payload as Record<string, unknown>;
      const errorPayload = record.error;
      if (errorPayload && typeof errorPayload === "object") {
        const message = (errorPayload as Record<string, unknown>).message;
        if (message) return compactErrorText(String(message), apiKey);
      }
      if (record.detail) return compactErrorText(String(record.detail), apiKey);
    }
  } catch {
    return compactErrorText(text, apiKey);
  }
  return compactErrorText(text, apiKey);
}

function parseJsonBody<T>(
  body: string,
  path: string,
  status: number,
  apiKey = "",
): T {
  try {
    return JSON.parse(body) as T;
  } catch {
    const message = apiErrorMessage(body, apiKey);
    throw new Error(
      `HTTP ${status}: Expected JSON response from ${path}${message ? `: ${message}` : ""}`,
    );
  }
}

function isJsonObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function invalidReadiness(status: number, message: string): never {
  throw new Error(`HTTP ${status}: Invalid /ready response: ${message}`);
}

function parseReadinessBody(
  body: string,
  status: number,
  apiKey = "",
): ReadinessResponse {
  const payload = parseJsonBody<unknown>(body, "/ready", status, apiKey);
  if (!isJsonObject(payload)) invalidReadiness(status, "response must be an object");

  for (const field of ["status", "model", "version"] as const) {
    if (typeof payload[field] !== "string") {
      invalidReadiness(status, `${field} must be a string`);
    }
  }

  if (!isJsonObject(payload.checks)) {
    invalidReadiness(status, "checks must be an object");
  }
  for (const field of [
    "task_router_bound",
    "recognizer_process_alive",
    "ffmpeg_available",
  ] as const) {
    if (typeof payload.checks[field] !== "boolean") {
      invalidReadiness(status, `checks.${field} must be a boolean`);
    }
  }

  if (!isJsonObject(payload.config)) {
    invalidReadiness(status, "config must be an object");
  }
  for (const field of ["auth_enabled", "cors_enabled"] as const) {
    if (typeof payload.config[field] !== "boolean") {
      invalidReadiness(status, `config.${field} must be a boolean`);
    }
  }
  for (const field of [
    "max_upload_mb",
    "task_timeout",
    "max_concurrent_requests",
    "cors_origins_count",
  ] as const) {
    const value = payload.config[field];
    if (typeof value !== "number" || !Number.isFinite(value)) {
      invalidReadiness(status, `config.${field} must be a finite number`);
    }
  }

  return payload as unknown as ReadinessResponse;
}

export async function readResponseText(
  response: Response,
  maxBytes = MAX_RESPONSE_BODY_BYTES,
): Promise<string> {
  if (!Number.isFinite(maxBytes) || maxBytes <= 0) {
    throw new Error("HTTP response body limit must be > 0");
  }

  const contentLength = response.headers.get("Content-Length");
  if (contentLength) {
    const parsedLength = Number(contentLength);
    if (Number.isFinite(parsedLength) && parsedLength > maxBytes) {
      await response.body?.cancel().catch(() => undefined);
      throw new Error(`HTTP response body exceeded ${maxBytes} bytes`);
    }
  }

  if (!response.body) {
    const text = await response.text();
    if (new TextEncoder().encode(text).byteLength > maxBytes) {
      throw new Error(`HTTP response body exceeded ${maxBytes} bytes`);
    }
    return text;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let received = 0;
  let text = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      received += value.byteLength;
      if (received > maxBytes) {
        await reader.cancel().catch(() => undefined);
        throw new Error(`HTTP response body exceeded ${maxBytes} bytes`);
      }
      text += decoder.decode(value, { stream: true });
    }
    text += decoder.decode();
    return text;
  } finally {
    reader.releaseLock();
  }
}

async function readResponseTextWithStatus(
  response: Response,
  apiKey = "",
): Promise<string> {
  try {
    return await readResponseText(response);
  } catch (error) {
    const message = safeErrorMessage(error, apiKey, "Failed to read response body");
    throw new Error(`HTTP ${response.status}: ${message}`);
  }
}

async function readJsonResponse<T>(
  response: Response,
  path: string,
  apiKey = "",
): Promise<T> {
  return parseJsonBody<T>(
    await readResponseTextWithStatus(response, apiKey),
    path,
    response.status,
    apiKey,
  );
}

async function fetchWithTimeout<T>(
  input: RequestInfo | URL,
  init: RequestInit,
  timeoutMs: number,
  apiKey = "",
  consume: (response: Response) => Promise<T>,
): Promise<T> {
  const controller = new AbortController();
  const upstreamSignal = init.signal;
  let timedOut = false;
  let completed = false;
  const abortFromUpstream = () => controller.abort();
  if (upstreamSignal?.aborted) {
    controller.abort();
  } else {
    upstreamSignal?.addEventListener("abort", abortFromUpstream, { once: true });
  }
  const timeout = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  try {
    const response = await fetch(input, {
      ...init,
      // Audio, API keys, and diagnostics must never be replayed to a redirect
      // target. The browser reports redirects as a network error instead.
      redirect: "error",
      signal: controller.signal,
    });
    // Keep the same deadline and caller-abort bridge active until the body has
    // been consumed. fetch() itself resolves as soon as response headers arrive.
    const result = await consume(response);
    completed = true;
    return result;
  } catch (error) {
    if (timedOut) {
      throw new Error(`Request timed out after ${Math.round(timeoutMs / 1000)}s`);
    }
    if (upstreamSignal?.aborted) {
      const abortError =
        error instanceof DOMException && error.name === "AbortError"
          ? error
          : new DOMException("Request aborted", "AbortError");
      throw sanitizedAbortError(abortError, apiKey);
    }
    if (error instanceof DOMException && error.name === "AbortError") {
      throw sanitizedAbortError(error, apiKey);
    }
    throw new Error(safeErrorMessage(error, apiKey, "Network request failed"));
  } finally {
    if (!completed) {
      controller.abort();
    }
    window.clearTimeout(timeout);
    upstreamSignal?.removeEventListener("abort", abortFromUpstream);
  }
}

async function checkedFetch<T>(
  input: RequestInfo | URL,
  init: RequestInit,
  timeoutMs: number,
  apiKey: string,
  consume: (response: Response) => Promise<T>,
): Promise<T> {
  return fetchWithTimeout(input, init, timeoutMs, apiKey, async (response) => {
    if (!response.ok) {
      const message = await readResponseText(response)
        .then((detail) => apiErrorMessage(detail, apiKey))
        .catch((error: unknown) => safeErrorMessage(error, apiKey, ""));
      throw new Error(`HTTP ${response.status}${message ? `: ${message}` : ""}`);
    }
    return consume(response);
  });
}

export async function fetchHealth(settings: ApiSettings, signal?: AbortSignal): Promise<HealthResponse> {
  const root = normalizeApiRoot(settings.baseUrl);
  return checkedFetch(
    `${root}/health`,
    {
      headers: requestHeaders(settings),
      signal,
    },
    DIAGNOSTIC_TIMEOUT_MS,
    settings.apiKey,
    (response) => readJsonResponse<HealthResponse>(response, "/health", settings.apiKey),
  );
}

export async function fetchReadiness(settings: ApiSettings, signal?: AbortSignal): Promise<ReadinessResponse> {
  const root = normalizeApiRoot(settings.baseUrl);
  return fetchWithTimeout(
    `${root}/ready`,
    {
      headers: requestHeaders(settings),
      signal,
    },
    DIAGNOSTIC_TIMEOUT_MS,
    settings.apiKey,
    async (response) => {
      const body = await readResponseTextWithStatus(response, settings.apiKey);
      if (response.ok || response.status === 503) {
        return parseReadinessBody(body, response.status, settings.apiKey);
      }
      const message = apiErrorMessage(body, settings.apiKey);
      throw new Error(`HTTP ${response.status}${message ? `: ${message}` : ""}`);
    },
  );
}

export async function fetchModels(settings: ApiSettings, signal?: AbortSignal): Promise<ModelListResponse> {
  const root = normalizeApiRoot(settings.baseUrl);
  return checkedFetch(
    `${root}/v1/models`,
    {
      headers: requestHeaders(settings),
      signal,
    },
    DIAGNOSTIC_TIMEOUT_MS,
    settings.apiKey,
    (response) =>
      readJsonResponse<ModelListResponse>(response, "/v1/models", settings.apiKey),
  );
}

export async function parseTranscriptionResponse(
  response: Response,
  format: ResponseFormat,
  apiKey = "",
): Promise<TranscriptionResult> {
  const contentType = response.headers.get("Content-Type") ?? "";
  if (format === "text" || format === "srt" || format === "vtt") {
    const text = await readResponseTextWithStatus(response, apiKey);
    return { text, format, raw: text, contentType };
  }

  const payload = parseJsonBody<VerboseTranscription | { text?: string }>(
    await readResponseTextWithStatus(response, apiKey),
    "/v1/audio/transcriptions",
    response.status,
    apiKey,
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

  return checkedFetch(
    `${root}/v1/audio/transcriptions`,
    {
      method: "POST",
      headers: requestHeaders(settings),
      body,
      signal,
    },
    TRANSCRIPTION_TIMEOUT_MS,
    settings.apiKey,
    (response) =>
      parseTranscriptionResponse(response, settings.responseFormat, settings.apiKey),
  );
}
