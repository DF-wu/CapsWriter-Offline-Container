import type {
  ApiSettings,
  HealthResponse,
  ModelListResponse,
  ResponseFormat,
  TranscriptionResult,
  VerboseTranscription,
} from "../types";

export function normalizeApiRoot(baseUrl: string): string {
  const trimmed = baseUrl.trim().replace(/\/+$/, "");
  if (!trimmed) {
    return "http://localhost:6017";
  }
  return trimmed.endsWith("/v1") ? trimmed.slice(0, -3) : trimmed;
}

function requestHeaders(settings: Pick<ApiSettings, "apiKey">): HeadersInit {
  return settings.apiKey.trim()
    ? { Authorization: `Bearer ${settings.apiKey.trim()}` }
    : {};
}

async function checkedFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const response = await fetch(input, init);
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(`HTTP ${response.status}${detail ? `: ${detail}` : ""}`);
  }
  return response;
}

export async function fetchHealth(settings: ApiSettings): Promise<HealthResponse> {
  const root = normalizeApiRoot(settings.baseUrl);
  const response = await checkedFetch(`${root}/health`, {
    headers: requestHeaders(settings),
  });
  return response.json() as Promise<HealthResponse>;
}

export async function fetchModels(settings: ApiSettings): Promise<ModelListResponse> {
  const root = normalizeApiRoot(settings.baseUrl);
  const response = await checkedFetch(`${root}/v1/models`, {
    headers: requestHeaders(settings),
  });
  return response.json() as Promise<ModelListResponse>;
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

  const payload = (await response.json()) as VerboseTranscription | { text?: string };
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
