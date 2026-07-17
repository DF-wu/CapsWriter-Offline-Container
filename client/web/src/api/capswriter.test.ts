import { afterEach, describe, expect, it, vi } from "vitest";
import {
  MAX_RESPONSE_BODY_BYTES,
  TRANSCRIPTION_TIMEOUT_MS,
  apiErrorMessage,
  fetchHealth,
  fetchReadiness,
  normalizeApiRoot,
  parseTranscriptionResponse,
  readResponseText,
  safeErrorMessage,
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

async function rejectionMessage(promise: Promise<unknown>): Promise<string> {
  try {
    await promise;
  } catch (error) {
    if (error instanceof Error) return error.message;
    throw new Error("Promise rejected without an Error instance");
  }
  throw new Error("Expected promise to reject");
}

function stalledBodyResponse(
  signal: AbortSignal | null | undefined,
  init?: ResponseInit,
): Response {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      const abort = () => {
        controller.error(new DOMException("body aborted", "AbortError"));
      };
      if (signal?.aborted) {
        abort();
      } else {
        signal?.addEventListener("abort", abort, { once: true });
      }
    },
  });
  return new Response(body, init);
}

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

  it("redacts reflected keys from malformed json transcription diagnostics", async () => {
    const apiKey = "sk-web-malformed-secret";
    const response = new Response(`{broken Bearer ${apiKey}\u0000\u001b[31m`, {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });

    const message = await rejectionMessage(
      parseTranscriptionResponse(response, "json", apiKey),
    );

    expect(message).toContain("Bearer [REDACTED]");
    expect(message).not.toContain(apiKey);
    expect(message).not.toMatch(/[\u0000-\u001f\u007f-\u009f]/u);
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

  it("redacts keys and control characters from every peer error envelope", () => {
    const apiKey = "sk-peer-reflection-secret";
    const bodies = [
      JSON.stringify({
        error: { message: `Denied Authorization: Bearer ${apiKey}\u0000\u001b[31m` },
      }),
      JSON.stringify({ detail: `Legacy detail reflected ${apiKey}\u0007` }),
      `{malformed proxy body Bearer ${apiKey}\u0000\u200b\u202e`,
    ];

    for (const body of bodies) {
      const message = apiErrorMessage(body, apiKey);
      expect(message).toContain("[REDACTED]");
      expect(message).not.toContain(apiKey);
      expect(message).not.toMatch(/[\u0000-\u001f\u007f-\u009f\u202a-\u202e]/u);
      expect(message).not.toContain("\u200b");
      expect(message.length).toBeLessThanOrEqual(503);
    }
  });

  it("redacts a key spanning the bounded preview boundary", () => {
    const apiKey = `sk-${"s".repeat(64)}`;
    const message = apiErrorMessage(
      `${"x".repeat(490)}Bearer ${apiKey}${"y".repeat(700)}`,
      apiKey,
    );

    expect(message).not.toContain(apiKey);
    expect(message).toContain("[RE");
    expect(message).toHaveLength(503);
  });

  it("redacts a key cut by the bounded source scan after collapsible text", () => {
    const apiKey = `sk-${"s".repeat(64)}`;
    const exposedPrefix = apiKey.slice(0, -1);
    const message = apiErrorMessage(
      JSON.stringify({
        error: { message: `Denied${" ".repeat(1995)}${apiKey}` },
      }),
      apiKey,
    );

    expect(message).toContain("[REDACTED]");
    expect(message).not.toContain(exposedPrefix);
    expect(message).not.toContain(apiKey);
  });

  it("redacts and bounds network error strings", () => {
    const apiKey = "sk-network-reflection-secret";
    const message = safeErrorMessage(
      new TypeError(`Network failed for Bearer ${apiKey}\u0000${"x".repeat(700)}`),
      apiKey,
    );

    expect(message).toContain("Bearer [REDACTED]");
    expect(message).not.toContain(apiKey);
    expect(message).not.toMatch(/[\u0000-\u001f\u007f-\u009f]/u);
    expect(message.length).toBe(503);
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
        recognizer_process_alive: false,
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

  it("redacts a reflected key from readiness detail errors", async () => {
    const apiKey = "sk-readiness-reflection-secret";
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(
          JSON.stringify({ detail: `Rejected Bearer ${apiKey}\u001b[31m` }),
          { status: 401 },
        ),
      ),
    );

    const message = await rejectionMessage(
      fetchReadiness({ ...settings, apiKey }),
    );

    expect(message).toBe("HTTP 401: Rejected Bearer [REDACTED] [31m");
    expect(message).not.toContain(apiKey);
    expect(message).not.toContain("\u001b");
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

  it("keeps the HTTP status for a valid-json readiness schema error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(
          JSON.stringify({ status: "degraded", model: "mock_asr", version: "dev" }),
          { status: 503 },
        ),
      ),
    );

    await expect(fetchReadiness(settings)).rejects.toThrow(
      "HTTP 503: Invalid /ready response: checks must be an object",
    );
  });
});

describe("diagnostic fetches", () => {
  it("rejects redirects for diagnostic requests", async () => {
    const fetch = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) =>
        new Response(JSON.stringify({ status: "ok" }), { status: 200 }),
    );
    vi.stubGlobal("fetch", fetch);

    await fetchHealth(settings);

    expect(fetch).toHaveBeenCalledOnce();
    expect(fetch.mock.calls[0]?.[1]).toMatchObject({ redirect: "error" });
  });

  it("rejects invalid API roots before sending diagnostics", async () => {
    const fetch = vi.fn();
    vi.stubGlobal("fetch", fetch);

    await expect(
      fetchHealth({ ...settings, baseUrl: "ftp://asr.example.test" }),
    ).rejects.toThrow("API root must use http:// or https://");
    expect(fetch).not.toHaveBeenCalled();
  });

  it("redacts reflected keys from non-json HTTP errors", async () => {
    const apiKey = "sk-proxy-reflection-secret";
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(`Proxy rejected Bearer ${apiKey}\u0000\u0007`, { status: 502 }),
      ),
    );

    const message = await rejectionMessage(fetchHealth({ ...settings, apiKey }));

    expect(message).toBe("HTTP 502: Proxy rejected Bearer [REDACTED]");
    expect(message).not.toContain(apiKey);
    expect(message).not.toMatch(/[\u0000-\u001f\u007f-\u009f]/u);
  });

  it("redacts reflected keys from malformed successful responses", async () => {
    const apiKey = "sk-json-reflection-secret";
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(`not-json Bearer ${apiKey}\u0000`, { status: 200 })),
    );

    const message = await rejectionMessage(fetchHealth({ ...settings, apiKey }));

    expect(message).toContain("Expected JSON response from /health");
    expect(message).toContain("Bearer [REDACTED]");
    expect(message).not.toContain(apiKey);
    expect(message).not.toMatch(/[\u0000-\u001f\u007f-\u009f]/u);
  });

  it("cancels an oversized declared body and aborts its request scope", async () => {
    let cancelled = false;
    let requestSignal: AbortSignal | null | undefined;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
        requestSignal = init?.signal;
        const body = new ReadableStream<Uint8Array>({
          cancel() {
            cancelled = true;
          },
        });
        return new Response(body, {
          status: 502,
          headers: { "Content-Length": String(MAX_RESPONSE_BODY_BYTES + 1) },
        });
      }),
    );

    await expect(fetchHealth(settings)).rejects.toThrow(
      `HTTP 502: HTTP response body exceeded ${MAX_RESPONSE_BODY_BYTES} bytes`,
    );
    expect(cancelled).toBe(true);
    expect(requestSignal?.aborted).toBe(true);
  });

  it("redacts reflected keys from network failures", async () => {
    const apiKey = "sk-fetch-reflection-secret";
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError(`Failed to fetch with Bearer ${apiKey}\u0000\u001b[31m`);
      }),
    );

    const message = await rejectionMessage(fetchHealth({ ...settings, apiKey }));

    expect(message).toBe("Failed to fetch with Bearer [REDACTED] [31m");
    expect(message).not.toContain(apiKey);
    expect(message).not.toContain("\u001b");
  });

  it("preserves abort semantics while redacting abort error strings", async () => {
    const apiKey = "sk-abort-reflection-secret";
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new DOMException(`Aborted Bearer ${apiKey}\u0000`, "AbortError");
      }),
    );

    let rejection: unknown;
    try {
      await fetchHealth({ ...settings, apiKey });
    } catch (error) {
      rejection = error;
    }

    expect(rejection).toBeInstanceOf(DOMException);
    expect((rejection as DOMException).name).toBe("AbortError");
    expect((rejection as DOMException).message).toBe("Aborted Bearer [REDACTED]");
    expect((rejection as DOMException).message).not.toContain(apiKey);
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

  it("keeps the diagnostic timeout active while reading the response body", async () => {
    vi.useFakeTimers();
    let requestSignal: AbortSignal | null | undefined;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
        requestSignal = init?.signal;
        return stalledBodyResponse(requestSignal, {
          headers: { "Content-Type": "application/json" },
        });
      }),
    );

    const health = expect(fetchHealth(settings)).rejects.toThrow(
      "Request timed out after 10s",
    );
    await vi.advanceTimersByTimeAsync(10_000);

    await health;
    expect(requestSignal?.aborted).toBe(true);
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

  it("aborts a stalled diagnostic response body with the caller signal", async () => {
    const controller = new AbortController();
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) =>
        stalledBodyResponse(init?.signal, {
          headers: { "Content-Type": "application/json" },
        }),
      ),
    );

    const health = fetchHealth(settings, controller.signal);
    controller.abort();

    await expect(health).rejects.toMatchObject({ name: "AbortError" });
  });
});

describe("transcribeAudio", () => {
  it("does not allow private multipart audio to be replayed across redirects", async () => {
    const fetch = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) =>
        new Response("ok", { status: 200 }),
    );
    vi.stubGlobal("fetch", fetch);

    await transcribeAudio(new Blob(["private audio"]), "private.wav", settings);

    expect(fetch).toHaveBeenCalledOnce();
    const init = fetch.mock.calls[0]?.[1];
    expect(init).toMatchObject({ method: "POST", redirect: "error" });
    expect(init?.body).toBeInstanceOf(FormData);
  });

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

  it("redacts a full reflected Bearer token from transcription errors", async () => {
    const apiKey = "sk-transcription-reflection-secret";
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(
          JSON.stringify({
            error: {
              message: `Invalid Authorization: Bearer ${apiKey}\u0000`,
              type: "authentication_error",
            },
          }),
          { status: 401 },
        ),
      ),
    );

    const message = await rejectionMessage(
      transcribeAudio(
        new Blob(["RIFF"]),
        "sample.wav",
        { ...settings, apiKey },
      ),
    );

    expect(message).toBe("HTTP 401: Invalid Authorization: Bearer [REDACTED]");
    expect(message).not.toContain(apiKey);
    expect(message).not.toMatch(/[\u0000-\u001f\u007f-\u009f]/u);
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

  it("times out transcription requests that never resolve", async () => {
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

    const transcription = expect(
      transcribeAudio(new Blob(["RIFF"]), "sample.wav", settings),
    ).rejects.toThrow("Request timed out after 600s");
    await vi.advanceTimersByTimeAsync(TRANSCRIPTION_TIMEOUT_MS);

    await transcription;
  });

  it("keeps the transcription timeout active while reading the response body", async () => {
    vi.useFakeTimers();
    let requestSignal: AbortSignal | null | undefined;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
        requestSignal = init?.signal;
        return stalledBodyResponse(requestSignal, {
          headers: { "Content-Type": "text/plain" },
        });
      }),
    );

    const transcription = expect(
      transcribeAudio(new Blob(["RIFF"]), "sample.wav", settings),
    ).rejects.toThrow("Request timed out after 600s");
    await vi.advanceTimersByTimeAsync(TRANSCRIPTION_TIMEOUT_MS);

    await transcription;
    expect(requestSignal?.aborted).toBe(true);
  });

  it("aborts transcription requests with the caller signal", async () => {
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

    const transcription = expect(
      transcribeAudio(new Blob(["RIFF"]), "sample.wav", settings, controller.signal),
    ).rejects.toThrow("aborted");
    controller.abort();

    await transcription;
  });

  it("aborts a stalled transcription response body with the caller signal", async () => {
    const controller = new AbortController();
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) =>
        stalledBodyResponse(init?.signal, {
          headers: { "Content-Type": "text/plain" },
        }),
      ),
    );

    const transcription = transcribeAudio(
      new Blob(["RIFF"]),
      "sample.wav",
      settings,
      controller.signal,
    );
    controller.abort();

    await expect(transcription).rejects.toMatchObject({ name: "AbortError" });
  });
});
