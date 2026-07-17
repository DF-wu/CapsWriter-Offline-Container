import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  addHistory,
  clearHistory,
  loadHistory,
  loadSettings,
  saveHistory,
  saveSettings,
  settingsWithRuntimeDefaults,
} from "./storage";
import type { TranscriptRecord } from "../types";

const SETTINGS_KEY = "capswriter.web.settings.v1";
const HISTORY_KEY = "capswriter.web.history.v1";

beforeEach(() => {
  localStorage.clear();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("settingsWithRuntimeDefaults", () => {
  it("uses built-in defaults without runtime config", () => {
    expect(settingsWithRuntimeDefaults(undefined)).toMatchObject({
      baseUrl: "http://localhost:6017",
      model: "whisper-1",
      responseFormat: "verbose_json",
    });
  });

  it("applies deploy-time runtime config", () => {
    expect(
      settingsWithRuntimeDefaults({
        baseUrl: "https://asr.example.test",
        apiKey: "sk-private-deploy",
        model: "whisper-1",
        language: "zh",
        prompt: "terms",
        responseFormat: "text",
      }),
    ).toMatchObject({
      baseUrl: "https://asr.example.test",
      apiKey: "sk-private-deploy",
      language: "zh",
      prompt: "terms",
      model: "whisper-1",
      responseFormat: "text",
    });
  });

  it("ignores non-object runtime config", () => {
    expect(settingsWithRuntimeDefaults(["bad", "shape"] as never)).toMatchObject({
      baseUrl: "http://localhost:6017",
      apiKey: "",
      model: "whisper-1",
      language: "",
      prompt: "",
      responseFormat: "verbose_json",
    });
  });

  it("falls back when runtime config fields have invalid types", () => {
    expect(
      settingsWithRuntimeDefaults({
        baseUrl: 6017,
        apiKey: ["sk"],
        model: false,
        language: ["zh"],
        prompt: { text: "terms" },
        responseFormat: "xml",
      } as never),
    ).toMatchObject({
      baseUrl: "http://localhost:6017",
      apiKey: "",
      model: "whisper-1",
      language: "",
      prompt: "",
      responseFormat: "verbose_json",
    });
  });

  it("bounds runtime config strings", () => {
    const settings = settingsWithRuntimeDefaults({
      baseUrl: "https://".padEnd(3000, "a"),
      apiKey: "k".repeat(5000),
      model: "m".repeat(300),
      language: "l".repeat(100),
      prompt: "p".repeat(20_000),
    });

    expect(settings.baseUrl).toHaveLength(2048);
    expect(settings.apiKey).toHaveLength(4096);
    expect(settings.model).toHaveLength(128);
    expect(settings.language).toHaveLength(32);
    expect(settings.prompt).toHaveLength(16_384);
  });

  it("falls back when runtime response format is invalid", () => {
    expect(
      settingsWithRuntimeDefaults({
        responseFormat: "xml" as never,
      }).responseFormat,
    ).toBe("verbose_json");
  });

  it("does not persist API keys in localStorage", () => {
    saveSettings({
      ...settingsWithRuntimeDefaults(undefined),
      apiKey: "sk-local-secret",
      baseUrl: "https://asr.example.test",
    });

    const raw = localStorage.getItem(SETTINGS_KEY) ?? "";
    expect(raw).not.toContain("sk-local-secret");
    expect(JSON.parse(raw)).not.toHaveProperty("apiKey");
    expect(loadSettings()).toMatchObject({
      apiKey: "",
      baseUrl: "https://asr.example.test",
    });
  });

  it("ignores API keys from older persisted settings", () => {
    localStorage.setItem(
      SETTINGS_KEY,
      JSON.stringify({
        apiKey: "sk-old-secret",
        baseUrl: "https://saved.example.test",
        responseFormat: "text",
      }),
    );

    expect(loadSettings()).toMatchObject({
      apiKey: "",
      baseUrl: "https://saved.example.test",
      responseFormat: "text",
    });
  });

  it("ignores non-object persisted settings", () => {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(["bad", "shape"]));

    expect(loadSettings()).toMatchObject({
      apiKey: "",
      baseUrl: "http://localhost:6017",
      model: "whisper-1",
      responseFormat: "verbose_json",
    });
  });

  it("falls back when persisted settings have invalid field types", () => {
    localStorage.setItem(
      SETTINGS_KEY,
      JSON.stringify({
        baseUrl: 6017,
        model: false,
        language: ["zh"],
        prompt: { text: "terms" },
        responseFormat: "xml",
      }),
    );

    expect(loadSettings()).toMatchObject({
      baseUrl: "http://localhost:6017",
      model: "whisper-1",
      language: "",
      prompt: "",
      responseFormat: "verbose_json",
    });
  });

  it("bounds persisted settings strings", () => {
    localStorage.setItem(
      SETTINGS_KEY,
      JSON.stringify({
        baseUrl: "https://".padEnd(3000, "a"),
        model: "m".repeat(300),
        language: "l".repeat(100),
        prompt: "p".repeat(20_000),
        responseFormat: "text",
      }),
    );

    const settings = loadSettings();

    expect(settings.baseUrl).toHaveLength(2048);
    expect(settings.model).toHaveLength(128);
    expect(settings.language).toHaveLength(32);
    expect(settings.prompt).toHaveLength(16_384);
    expect(settings.responseFormat).toBe("text");
  });

  it("bounds settings before saving", () => {
    saveSettings({
      ...settingsWithRuntimeDefaults(undefined),
      baseUrl: "https://".padEnd(3000, "a"),
      model: "m".repeat(300),
      language: "l".repeat(100),
      prompt: "p".repeat(20_000),
    });

    const persisted = JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? "{}") as Record<string, string>;
    expect(persisted.baseUrl).toHaveLength(2048);
    expect(persisted.model).toHaveLength(128);
    expect(persisted.language).toHaveLength(32);
    expect(persisted.prompt).toHaveLength(16_384);
  });

  it("ignores blocked storage when saving settings", () => {
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("storage blocked");
    });

    expect(() =>
      saveSettings({
        ...settingsWithRuntimeDefaults(undefined),
        baseUrl: "https://asr.example.test",
      }),
    ).not.toThrow();
  });

  it("returns new history when persistence is blocked", () => {
    const record: TranscriptRecord = {
      id: "record-1",
      createdAt: "2026-07-07T00:00:00.000Z",
      sourceName: "sample.wav",
      durationSeconds: 1,
      format: "text",
      text: "hello",
      raw: "hello",
    };
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("quota exceeded");
    });

    expect(addHistory(record)).toEqual([record]);
  });

  it("ignores non-array persisted history", () => {
    const record: TranscriptRecord = {
      id: "record-1",
      createdAt: "2026-07-07T00:00:00.000Z",
      sourceName: "sample.wav",
      durationSeconds: 1,
      format: "text",
      text: "hello",
      raw: "hello",
    };
    localStorage.setItem(HISTORY_KEY, JSON.stringify({ bad: "shape" }));

    expect(loadHistory()).toEqual([]);
    expect(addHistory(record)).toEqual([record]);
  });

  it("filters malformed persisted history records", () => {
    const valid: TranscriptRecord = {
      id: "record-1",
      createdAt: "2026-07-07T00:00:00.000Z",
      sourceName: "sample.wav",
      durationSeconds: null,
      format: "verbose_json",
      text: "hello",
      raw: { text: "hello" },
    };
    localStorage.setItem(
      HISTORY_KEY,
      JSON.stringify([
        valid,
        { ...valid, id: "" },
        { ...valid, createdAt: "not-a-date" },
        { ...valid, sourceName: "" },
        { ...valid, durationSeconds: -1 },
        { ...valid, format: "xml" },
        { ...valid, text: 123 },
        { ...valid, id: "x".repeat(129) },
        { ...valid, sourceName: "x".repeat(513) },
        { ...valid, text: "x".repeat(200_001) },
        { ...valid, raw: "x".repeat(500_001) },
        { ...valid, raw: { text: "x".repeat(500_000) } },
        { ...valid, raw: ["bad"] },
      ]),
    );

    expect(loadHistory()).toEqual([valid]);
  });

  it("filters malformed history before saving", () => {
    const valid: TranscriptRecord = {
      id: "record-1",
      createdAt: "2026-07-07T00:00:00.000Z",
      sourceName: "sample.wav",
      durationSeconds: 1,
      format: "text",
      text: "hello",
      raw: "hello",
    };

    saveHistory([valid, { ...valid, raw: ["bad"] } as never]);

    expect(JSON.parse(localStorage.getItem(HISTORY_KEY) ?? "[]")).toEqual([valid]);
  });

  it("filters oversized history before saving", () => {
    const valid: TranscriptRecord = {
      id: "record-1",
      createdAt: "2026-07-07T00:00:00.000Z",
      sourceName: "sample.wav",
      durationSeconds: 1,
      format: "text",
      text: "hello",
      raw: "hello",
    };

    saveHistory([
      valid,
      { ...valid, id: "x".repeat(129) },
      { ...valid, sourceName: "x".repeat(513) },
      { ...valid, text: "x".repeat(200_001), raw: "x".repeat(200_001) },
      { ...valid, raw: "x".repeat(500_001) },
    ]);

    expect(JSON.parse(localStorage.getItem(HISTORY_KEY) ?? "[]")).toEqual([valid]);
  });

  it("ignores blocked storage when clearing history", () => {
    vi.spyOn(Storage.prototype, "removeItem").mockImplementation(() => {
      throw new Error("storage blocked");
    });

    expect(() => clearHistory()).not.toThrow();
  });
});
