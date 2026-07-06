import { createServer } from "node:net";
import { spawn, spawnSync } from "node:child_process";
import { mkdir, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const expectedText = "mock transcript from CapsWriter Web Console";
const npx = process.platform === "win32" ? "npx.cmd" : "npx";
const session = `capswriter-web-smoke-${process.pid}`;
const children = [];

function freePort() {
  return new Promise((resolve, reject) => {
    const server = createServer();
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      const port = typeof address === "object" && address ? address.port : 0;
      server.close(() => resolve(port));
    });
  });
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForHttp(url, timeoutMs = 25000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch {
      // Keep polling until the local dev server starts.
    }
    await sleep(250);
  }
  throw new Error(`Timed out waiting for ${url}`);
}

function runAgent(args, { allowFailure = false } = {}) {
  const result = spawnSync(npx, ["agent-browser", "--session", session, ...args], {
    cwd: root,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
  if (result.status !== 0 && !allowFailure) {
    throw new Error(
      `agent-browser ${args.join(" ")} failed\n${result.stdout}${result.stderr}`,
    );
  }
  return `${result.stdout}${result.stderr}`;
}

async function writeSmokeWav(path) {
  const sampleRate = 16000;
  const durationSeconds = 0.2;
  const samples = Math.floor(sampleRate * durationSeconds);
  const dataSize = samples * 2;
  const buffer = Buffer.alloc(44 + dataSize);

  buffer.write("RIFF", 0);
  buffer.writeUInt32LE(36 + dataSize, 4);
  buffer.write("WAVE", 8);
  buffer.write("fmt ", 12);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20);
  buffer.writeUInt16LE(1, 22);
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(sampleRate * 2, 28);
  buffer.writeUInt16LE(2, 32);
  buffer.writeUInt16LE(16, 34);
  buffer.write("data", 36);
  buffer.writeUInt32LE(dataSize, 40);

  for (let i = 0; i < samples; i += 1) {
    const value = Math.round(Math.sin((2 * Math.PI * 440 * i) / sampleRate) * 12000);
    buffer.writeInt16LE(value, 44 + i * 2);
  }

  await writeFile(path, buffer);
}

function startChild(command, args, options = {}) {
  const child = spawn(command, args, {
    cwd: root,
    stdio: ["ignore", "pipe", "pipe"],
    ...options,
  });
  children.push(child);
  child.stdout.on("data", (chunk) => process.stdout.write(chunk));
  child.stderr.on("data", (chunk) => process.stderr.write(chunk));
  return child;
}

async function pollTranscript() {
  const started = Date.now();
  while (Date.now() - started < 25000) {
    const value = runAgent(["get", "value", "textarea.transcript-output"], {
      allowFailure: true,
    }).trim();
    if (value.includes(expectedText)) {
      return;
    }
    await sleep(500);
  }
  throw new Error("Transcript textarea did not contain mock transcription");
}

async function main() {
  const [apiPort, webPort] = await Promise.all([freePort(), freePort()]);
  const apiRoot = `http://127.0.0.1:${apiPort}`;
  const webRoot = `http://127.0.0.1:${webPort}`;
  const tmpDir = join(root, ".tmp");
  const audioPath = join(tmpDir, "browser-smoke.wav");
  await mkdir(tmpDir, { recursive: true });
  await writeSmokeWav(audioPath);

  startChild(process.execPath, ["scripts/mock-api.mjs"], {
    env: { ...process.env, CAPSWRITER_MOCK_API_PORT: String(apiPort) },
  });
  startChild(npx, ["vite", "--host", "127.0.0.1", "--port", String(webPort), "--strictPort"]);
  await Promise.all([waitForHttp(`${apiRoot}/health`), waitForHttp(webRoot)]);

  runAgent(["close"], { allowFailure: true });
  runAgent(["open", webRoot]);
  runAgent(["find", "label", "API root", "fill", apiRoot]);
  runAgent(["select", "select", "text"]);
  runAgent(["find", "role", "button", "click", "--name", "檢查服務"]);
  runAgent(["wait", "--text", "mock_asr"]);
  runAgent(["upload", "input[type=file]", audioPath]);
  runAgent(["wait", "--text", "browser-smoke.wav"]);
  runAgent(["find", "role", "button", "click", "--name", "轉錄"]);
  await pollTranscript();
  console.log("Browser smoke passed");
}

try {
  await main();
} finally {
  runAgent(["close"], { allowFailure: true });
  await rm(join(root, ".tmp"), { recursive: true, force: true });
  await Promise.all(
    children.map(
      (child) =>
        new Promise((resolve) => {
          if (child.exitCode !== null) {
            resolve();
            return;
          }
          child.once("exit", resolve);
          child.kill();
        }),
    ),
  );
}
