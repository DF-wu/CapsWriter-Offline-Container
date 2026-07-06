import { describe, expect, it } from "vitest";
import { normalizeApiRoot, parseTranscriptionResponse } from "./capswriter";

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
