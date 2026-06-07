import { File, Paths } from "expo-file-system";

import { isWeb } from "@/lib/platform";
import type {
  ApiProbe,
  AsrSettings,
  ChatMessage,
  ConversationSettings,
  TranscriptionResult,
  TtsSettings,
  UploadableAudio,
} from "@/types/client";

class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function endpoint(baseUrl: string, path: string) {
  const cleanBase = baseUrl.trim().replace(/\/+$/, "");
  const cleanPath = path.startsWith("/") ? path : `/${path}`;
  if (cleanBase.endsWith("/v1")) {
    return `${cleanBase}${cleanPath}`;
  }
  return `${cleanBase}/v1${cleanPath}`;
}

function authHeaders(apiKey: string, extra?: HeadersInit): HeadersInit {
  return {
    ...(apiKey.trim() ? { Authorization: `Bearer ${apiKey.trim()}` } : {}),
    ...extra,
  };
}

function timeoutSignal(timeoutSec: number) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), Math.max(1, timeoutSec) * 1000);
  return { signal: controller.signal, cancel: () => clearTimeout(timeout) };
}

async function readError(response: Response) {
  const body = await response.text().catch(() => "");
  try {
    const parsed = JSON.parse(body) as { error?: { message?: string }; detail?: string };
    return parsed.error?.message ?? parsed.detail ?? body;
  } catch {
    return body || response.statusText || "Request failed";
  }
}

async function ensureOk(response: Response) {
  if (!response.ok) {
    throw new ApiError(response.status, await readError(response));
  }
}

function appendAudio(form: FormData, audio: UploadableAudio) {
  if (audio.file) {
    form.append("file", audio.file, audio.name);
    return;
  }
  if (!audio.uri) {
    throw new Error("No audio file or URI available");
  }
  form.append("file", {
    uri: audio.uri,
    name: audio.name,
    type: audio.mimeType,
  } as unknown as Blob);
}

export async function transcribeAudio(
  settings: AsrSettings,
  audio: UploadableAudio,
): Promise<TranscriptionResult> {
  const form = new FormData();
  appendAudio(form, audio);
  form.append("model", settings.model);
  form.append("response_format", settings.responseFormat);
  form.append("temperature", String(settings.temperature));
  if (settings.language.trim()) {
    form.append("language", settings.language.trim());
  }
  if (settings.prompt.trim()) {
    form.append("prompt", settings.prompt.trim());
  }

  const { signal, cancel } = timeoutSignal(settings.timeoutSec);
  try {
    const response = await fetch(endpoint(settings.baseUrl, "/audio/transcriptions"), {
      method: "POST",
      headers: authHeaders(settings.apiKey),
      body: form,
      signal,
    });
    await ensureOk(response);
    const contentType = response.headers.get("content-type") ?? "";
    if (settings.responseFormat === "text" || settings.responseFormat === "srt" || settings.responseFormat === "vtt") {
      const text = await response.text();
      return { text, raw: text, contentType };
    }

    const raw = (await response.json()) as unknown;
    return { text: extractTranscriptionText(raw), raw, contentType };
  } finally {
    cancel();
  }
}

export async function runConversation(
  settings: ConversationSettings,
  messages: ChatMessage[],
  options: { onDelta?: (delta: string) => void } = {},
): Promise<string> {
  const payload =
    settings.mode === "responses"
      ? responsesPayload(settings, messages)
      : chatCompletionsPayload(settings, messages);
  const path = settings.mode === "responses" ? "/responses" : "/chat/completions";
  const { signal, cancel } = timeoutSignal(settings.timeoutSec);
  try {
    const response = await fetch(endpoint(settings.baseUrl, path), {
      method: "POST",
      headers: authHeaders(settings.apiKey, { "Content-Type": "application/json" }),
      body: JSON.stringify(payload),
      signal,
    });
    await ensureOk(response);
    if (settings.stream) {
      return readConversationStream(response, settings.mode, options.onDelta);
    }
    const json = (await response.json()) as unknown;
    return settings.mode === "responses" ? extractResponseText(json) : extractChatText(json);
  } finally {
    cancel();
  }
}

export async function synthesizeSpeech(settings: TtsSettings, input: string): Promise<string> {
  const { signal, cancel } = timeoutSignal(settings.timeoutSec);
  try {
    const response = await fetch(endpoint(settings.baseUrl, "/audio/speech"), {
      method: "POST",
      headers: authHeaders(settings.apiKey, { "Content-Type": "application/json" }),
      body: JSON.stringify({
        model: settings.model,
        voice: settings.voice,
        input,
        response_format: settings.responseFormat,
        speed: settings.speed,
        ...(settings.instructions.trim() ? { instructions: settings.instructions.trim() } : {}),
      }),
      signal,
    });
    await ensureOk(response);
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (isWeb()) {
      const blob = new Blob([bytes], { type: `audio/${settings.responseFormat}` });
      return URL.createObjectURL(blob);
    }

    const file = new File(Paths.cache, `capswriter-tts-${Date.now()}.${settings.responseFormat}`);
    file.write(bytes);
    return file.uri;
  } finally {
    cancel();
  }
}

export async function probeModels(baseUrl: string, apiKey: string): Promise<ApiProbe> {
  try {
    const response = await fetch(endpoint(baseUrl, "/models"), {
      headers: authHeaders(apiKey),
    });
    if (!response.ok) {
      return { ok: false, status: response.status, message: await readError(response) };
    }
    return { ok: true, status: response.status, message: "Model endpoint reachable" };
  } catch (error) {
    return {
      ok: false,
      message: error instanceof Error ? error.message : "Network request failed",
    };
  }
}

export function documentAssetToAudio(asset: {
  uri: string;
  name: string;
  mimeType?: string;
  size?: number;
  file?: Blob;
}): UploadableAudio {
  return {
    uri: asset.uri,
    file: asset.file,
    name: asset.name || "audio.m4a",
    mimeType: asset.mimeType || "audio/m4a",
    size: asset.size,
  };
}

export function recordingUriToAudio(uri: string): UploadableAudio {
  return {
    uri,
    name: "recording.m4a",
    mimeType: "audio/m4a",
  };
}

function chatCompletionsPayload(settings: ConversationSettings, messages: ChatMessage[]) {
  return {
    model: settings.model,
    messages: [
      ...(settings.systemPrompt.trim()
        ? [{ role: "system", content: settings.systemPrompt.trim() }]
        : []),
      ...messages.map((message) => ({
        role: message.role,
        content: message.content,
      })),
    ],
    temperature: settings.temperature,
    top_p: settings.topP,
    frequency_penalty: settings.frequencyPenalty,
    presence_penalty: settings.presencePenalty,
    max_tokens: settings.maxOutputTokens,
    stream: settings.stream,
  };
}

function responsesPayload(settings: ConversationSettings, messages: ChatMessage[]) {
  return {
    model: settings.model,
    input: [
      ...(settings.systemPrompt.trim()
        ? [{ role: "system", content: settings.systemPrompt.trim() }]
        : []),
      ...messages.map((message) => ({
        role: message.role,
        content: message.content,
      })),
    ],
    temperature: settings.temperature,
    top_p: settings.topP,
    max_output_tokens: settings.maxOutputTokens,
    stream: settings.stream,
  };
}

async function readConversationStream(
  response: Response,
  mode: ConversationSettings["mode"],
  onDelta?: (delta: string) => void,
) {
  let text = "";
  const emit = (delta: string) => {
    if (!delta) {
      return;
    }
    text += delta;
    onDelta?.(delta);
  };

  const consume = (event: SseEvent) => {
    if (event.data.trim() === "[DONE]") {
      return;
    }
    const parsed = parseJson(event.data);
    if (!parsed) {
      return;
    }
    const delta =
      mode === "responses"
        ? extractResponseStreamDelta(event.event, parsed)
        : extractChatStreamDelta(parsed);
    emit(delta);
  };

  await readSse(response, consume);
  return text;
}

type SseEvent = {
  event: string;
  data: string;
};

async function readSse(response: Response, onEvent: (event: SseEvent) => void) {
  const body = response.body;
  if (!body || !("getReader" in body) || typeof TextDecoder === "undefined") {
    parseSseText(await response.text(), onEvent);
    return;
  }

  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    buffer = drainSseBuffer(buffer, onEvent);
  }

  buffer += decoder.decode();
  if (buffer.trim()) {
    parseSseText(buffer, onEvent);
  }
}

function drainSseBuffer(buffer: string, onEvent: (event: SseEvent) => void) {
  const normalized = buffer.replace(/\r\n/g, "\n");
  const blocks = normalized.split("\n\n");
  const rest = blocks.pop() ?? "";
  for (const block of blocks) {
    parseSseBlock(block, onEvent);
  }
  return rest;
}

function parseSseText(text: string, onEvent: (event: SseEvent) => void) {
  for (const block of text.replace(/\r\n/g, "\n").split("\n\n")) {
    parseSseBlock(block, onEvent);
  }
}

function parseSseBlock(block: string, onEvent: (event: SseEvent) => void) {
  if (!block.trim()) {
    return;
  }
  let event = "message";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (!line || line.startsWith(":")) {
      continue;
    }
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      data.push(line.slice("data:".length).trimStart());
    }
  }
  if (data.length) {
    onEvent({ event, data: data.join("\n") });
  }
}

function parseJson(data: string): unknown | null {
  try {
    return JSON.parse(data);
  } catch {
    return null;
  }
}

function extractChatStreamDelta(raw: unknown): string {
  const value = raw as {
    choices?: Array<{ delta?: { content?: string | Array<{ text?: string }> } }>;
  };
  const content = value.choices?.[0]?.delta?.content;
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content.map((part) => part.text ?? "").join("");
  }
  return "";
}

function extractResponseStreamDelta(event: string, raw: unknown): string {
  const value = raw as {
    type?: string;
    delta?: string;
    output_text?: string;
  };
  const type = value.type ?? event;
  if (type === "response.output_text.delta" && typeof value.delta === "string") {
    return value.delta;
  }
  if (type === "response.completed" && typeof value.output_text === "string") {
    return value.output_text;
  }
  return "";
}

function extractTranscriptionText(raw: unknown): string {
  if (typeof raw === "string") {
    return raw;
  }
  if (raw && typeof raw === "object" && "text" in raw && typeof raw.text === "string") {
    return raw.text;
  }
  return JSON.stringify(raw, null, 2);
}

function extractChatText(raw: unknown): string {
  const value = raw as {
    choices?: Array<{ message?: { content?: string | Array<{ text?: string }> } }>;
  };
  const content = value.choices?.[0]?.message?.content;
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content.map((part) => part.text ?? "").join("");
  }
  return JSON.stringify(raw, null, 2);
}

function extractResponseText(raw: unknown): string {
  const value = raw as {
    output_text?: string;
    output?: Array<{ content?: Array<{ text?: string; type?: string }> }>;
  };
  if (value.output_text) {
    return value.output_text;
  }
  const text = value.output
    ?.flatMap((item) => item.content ?? [])
    .map((part) => part.text ?? "")
    .join("");
  return text || JSON.stringify(raw, null, 2);
}
