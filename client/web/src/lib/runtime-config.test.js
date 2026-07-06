import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { runInNewContext } from "node:vm";
import { afterEach, describe, expect, it } from "vitest";

const scriptPath = join(process.cwd(), "deploy", "write-config.sh");
const tempDirs = [];

afterEach(() => {
  for (const dir of tempDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

describe("write-config.sh", () => {
  it("writes deploy-time config as valid JavaScript string literals", () => {
    const dir = mkdtempSync(join(tmpdir(), "capswriter-web-config-"));
    tempDirs.push(dir);
    const configPath = join(dir, "config.js");
    const prompt = 'first line\n"quoted" \\ path\rsecond line';

    const result = spawnSync("sh", [scriptPath], {
      env: {
        ...process.env,
        CAPSWRITER_WEB_CONFIG_PATH: configPath,
        CAPSWRITER_WEB_API_BASE: "https://asr.example.test",
        CAPSWRITER_WEB_API_KEY: 'sk-"quoted"',
        CAPSWRITER_WEB_MODEL: "whisper-1",
        CAPSWRITER_WEB_LANGUAGE: "zh",
        CAPSWRITER_WEB_PROMPT: prompt,
        CAPSWRITER_WEB_RESPONSE_FORMAT: "text",
      },
      encoding: "utf8",
    });

    expect(result.status, result.stderr).toBe(0);
    const content = readFileSync(configPath, "utf8");
    const sandbox = { window: {} };
    runInNewContext(content, sandbox);

    expect(sandbox.window.__CAPSWRITER_WEB_CONFIG__).toMatchObject({
      baseUrl: "https://asr.example.test",
      apiKey: 'sk-"quoted"',
      language: "zh",
      prompt,
      responseFormat: "text",
    });
  });
});
