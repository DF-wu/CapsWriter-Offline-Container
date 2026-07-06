import { describe, expect, it } from "vitest";
import { settingsWithRuntimeDefaults } from "./storage";

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
});
