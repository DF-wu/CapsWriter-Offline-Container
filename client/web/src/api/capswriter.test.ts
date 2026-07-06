import { afterEach, describe, expect, it, vi } from "vitest";
import {
  apiErrorMessage,
  fetchReadiness,
  normalizeApiRoot,
  parseTranscriptionResponse,
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
});

describe("normalizeApiRoot", () => {
  it("keeps a root API URL unchanged", () => {
    expect(normalizeApiRoot("http://localhost:6017")).toBe("http://localhost:6017");
  });

  it("accepts OpenAI-style /v1 base URLs", () => {
    expect(normalizeApiRoot("http://localhost:6017/v1/")).toBe("http://localhost:6017");
  });

  it("falls back to the local HTTP API", () => {
    expect(normalizeApiRoot(" ")).toBe("http://localhost:6017");
  });
});

describe("parseTranscriptionResponse", () => {
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
});
