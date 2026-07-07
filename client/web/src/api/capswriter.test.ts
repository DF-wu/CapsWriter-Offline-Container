import { afterEach, describe, expect, it, vi } from "vitest";
import {
  MAX_RESPONSE_BODY_BYTES,
  apiErrorMessage,
  fetchHealth,
  fetchReadiness,
  normalizeApiRoot,
  parseTranscriptionResponse,
  readResponseText,
  transcribeAudio,
} from "./capswriter";
import type { ApiSettings } from "../types";

const settings: ApiSettings = {
  baseUrl: "http://localhost:6017",
  apiKey: "",
  model: "whisper-1",
  language: "",
  prompt: "",
  responseFormat: "text",
};

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("normalizeApiRoot", () => {
  it("keeps a root API URL unchanged", () => {
    expect(normalizeApiRoot("http://localhost:6017")).toBe("http://localhost:6017");
  });

  it("accepts OpenAI-style /v1 base URLs", () => {
    expect(normalizeApiRoot("http://localhost:6017/v1/")).toBe("http://localhost:6017");
  });

  it("keeps a path prefix while stripping OpenAI-style /v1", () => {
    expect(normalizeApiRoot("https://asr.example.test/capswriter/v1/")).toBe(
      "https://asr.example.test/capswriter",
    );
  });

  it("falls back to the local HTTP API", () => {
    expect(normalizeApiRoot(" ")).toBe("http://localhost:6017");
  });

  it("rejects unsupported URL schemes", () => {
    expect(() => normalizeApiRoot("ftp://asr.example.test")).toThrow(
      "API root must use http:// or https://",
    );
  });

  it("rejects URL credentials", () => {
    expect(() => normalizeApiRoot("https://user:secret@asr.example.test")).toThrow(
      "API root must not include username or password",
    );
  });

  it("rejects query strings and fragments", () => {
    expect(() => normalizeApiRoot("https://asr.example.test/v1?token=secret")).toThrow(
      "API root must not include query or fragment",
    );
    expect(() => normalizeApiRoot("https://asr.example.test/v1#ready")).toThrow(
      "API root must not include query or fragment",
    );
  });
});

describe("parseTranscriptionResponse", () => {
  it("rejects oversized response bodies while reading", async () => {
    const response = new Response("abcd");

    await expect(readResponseText(response, 3)).rejects.toThrow(
      "HTTP response body exceeded 3 bytes",
    );
  });

  it("parses plain text responses", async () => {
    const response = new Response("hello world", {
      headers: { "Content-Type": "text/plain;charset=utf-8" },
    });
    await expect(parseTranscriptionResponse(response, "text")).resolves.toEqual({
      text: "hello world",
      format: "text",
      raw: "hello world",
      contentType: "text/plain;charset=utf-8",
    });
  });

  it("extracts text from json responses", async () => {
    const response = new Response(JSON.stringify({ text: "你好" }), {
      headers: { "Content-Type": "application/json" },
    });
    const result = await parseTranscriptionResponse(response, "json");
    expect(result.text).toBe("你好");
    expect(result.raw).toEqual({ text: "你好" });
  });

  it("throws bounded diagnostics for invalid json transcription responses", async () => {
    const response = new Response(`<html>${"x".repeat(700)}</html>`, {
      status: 200,
      headers: { "Content-Type": "text/html" },
    });

    await expect(parseTranscriptionResponse(response, "json")).rejects.toThrow(
      /^HTTP 200: Expected JSON response from \/v1\/audio\/transcriptions: <html>x+.*\.\.\.$/,
    );
  });

  it("rejects oversized json transcription responses before parsing", async () => {
    const response = new Response("{}", {
      status: 200,
      headers: {
        "Content-Length": String(MAX_RESPONSE_BODY_BYTES + 1),
        "Content-Type": "application/json",
      },
    });

    await expect(parseTranscriptionResponse(response, "json")).rejects.toThrow(
      `HTTP 200: HTTP response body exceeded ${MAX_RESPONSE_BODY_BYTES} bytes`,
    );
  });

  it("preserves verbose json payloads", async () => {
    const payload = {
      text: "capswriter",
      duration: 1.5,
      words: [{ word: "capswriter", start: 0, end: 1.5 }],
    };
    const response = new Response(JSON.stringify(payload), {
      headers: { "Content-Type": "application/json" },
    });
    const result = await parseTranscriptionResponse(response, "verbose_json");
    expect(result.text).toBe("capswriter");
    expect(result.raw).toEqual(payload);
  });
});

describe("apiErrorMessage", () => {
  it("extracts OpenAI-style error messages", () => {
    expect(
      apiErrorMessage(
        JSON.stringify({
          error: {
            message: "Invalid API key",
            type: "authentication_error",
            param: null,
            code: null,
          },
        }),
      ),
    ).toBe("Invalid API key");
  });

  it("falls back to FastAPI detail payloads", () => {
    expect(apiErrorMessage(JSON.stringify({ detail: "Not Found" }))).toBe("Not Found");
  });

  it("bounds non-json error body previews", () => {
    const body = `<html>
      <body>${"x".repeat(700)}</body>
    </html>`;

    const message = apiErrorMessage(body);

    expect(message).toHaveLength(503);
    expect(message).toMatch(/^<html> <body>x+/);
    expect(message.endsWith("...")).toBe(true);
    expect(message).not.toContain("\n");
  });
});

describe("fetchReadiness", () => {
  it("returns degraded readiness diagnostics from HTTP 503", async () => {
    const payload = {
      status: "degraded",
      model: "mock_asr",
      version: "dev",
      checks: {
        task_router_bound: false,
        ffmpeg_available: true,
      },
      config: {
        auth_enabled: false,
        max_upload_mb: 100,
        task_timeout: 600,
        max_concurrent_requests: 2,
        cors_enabled: true,
        cors_origins_count: 1,
      },
    };
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify(payload), { status: 503 })),
    );

    await expect(fetchReadiness(settings)).resolves.toEqual(payload);
  });

  it("throws on unrelated readiness HTTP errors", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify({ detail: "missing" }), { status: 404 })),
    );

    await expect(fetchReadiness(settings)).rejects.toThrow("HTTP 404");
  });

  it("throws bounded diagnostics when readiness returns invalid json", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(`<html>${"x".repeat(700)}</html>`, { status: 503 })),
    );

    await expect(fetchReadiness(settings)).rejects.toThrow(
      /^HTTP 503: Expected JSON response from \/ready: <html>x+.*\.\.\.$/,
    );
  });
});

describe("diagnostic fetches", () => {
  it("rejects invalid API roots before sending diagnostics", async () => {
    const fetch = vi.fn();
    vi.stubGlobal("fetch", fetch);

    await expect(
      fetchHealth({ ...settings, baseUrl: "ftp://asr.example.test" }),
    ).rejects.toThrow("API root must use http:// or https://");
    expect(fetch).not.toHaveBeenCalled();
  });

  it("times out health checks that never resolve", async () => {
    vi.useFakeTimers();
    vi.stubGlobal(
      "fetch",
      vi.fn((_input: RequestInfo | URL, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => {
            reject(new DOMException("aborted", "AbortError"));
          });
        }),
      ),
    );

    const health = expect(fetchHealth(settings)).rejects.toThrow(
      "Request timed out after 10s",
    );
    await vi.advanceTimersByTimeAsync(10_000);

    await health;
  });

  it("aborts diagnostic requests with the caller signal", async () => {
    const controller = new AbortController();
    vi.stubGlobal(
      "fetch",
      vi.fn((_input: RequestInfo | URL, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => {
            reject(new DOMException("aborted", "AbortError"));
          });
        }),
      ),
    );

    const health = expect(fetchHealth(settings, controller.signal)).rejects.toThrow("aborted");
    controller.abort();

    await health;
  });
});

describe("transcribeAudio", () => {
  it("throws concise OpenAI-style server error messages", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(
          JSON.stringify({
            error: {
              message: "File too large (>100 MB)",
              type: "invalid_request_error",
              param: null,
              code: null,
            },
          }),
          { status: 413 },
        ),
      ),
    );

    await expect(
      transcribeAudio(new Blob(["RIFF"]), "sample.wav", settings),
    ).rejects.toThrow("HTTP 413: File too large (>100 MB)");
  });

  it("throws bounded diagnostics for oversized server error bodies", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response("error", {
          status: 502,
          headers: { "Content-Length": String(MAX_RESPONSE_BODY_BYTES + 1) },
        }),
      ),
    );

    await expect(
      transcribeAudio(new Blob(["RIFF"]), "sample.wav", settings),
    ).rejects.toThrow(
      `HTTP 502: HTTP response body exceeded ${MAX_RESPONSE_BODY_BYTES} bytes`,
    );
  });
});
