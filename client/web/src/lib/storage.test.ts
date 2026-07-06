import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  addHistory,
  clearHistory,
  loadSettings,
  saveSettings,
  settingsWithRuntimeDefaults,
} from "./storage";
import type { TranscriptRecord } from "../types";

const SETTINGS_KEY = "capswriter.web.settings.v1";

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
        model: "whisper-1",
        responseFormat: "text",
      }),
    ).toMatchObject({
      baseUrl: "https://asr.example.test",
      model: "whisper-1",
      responseFormat: "text",
    });
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

  it("ignores blocked storage when clearing history", () => {
    vi.spyOn(Storage.prototype, "removeItem").mockImplementation(() => {
      throw new Error("storage blocked");
    });

    expect(() => clearHistory()).not.toThrow();
  });
});
