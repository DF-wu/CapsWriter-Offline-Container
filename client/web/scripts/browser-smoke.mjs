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
const AGENT_BROWSER_TIMEOUT_ENV = "CAPSWRITER_WEB_BROWSER_AGENT_TIMEOUT_MS";
const CHILD_SHUTDOWN_TIMEOUT_ENV = "CAPSWRITER_WEB_BROWSER_CHILD_SHUTDOWN_TIMEOUT_MS";
const HTTP_PROBE_TIMEOUT_ENV = "CAPSWRITER_WEB_BROWSER_HTTP_PROBE_TIMEOUT_MS";
const AGENT_BROWSER_TIMEOUT_MS = readPositiveMilliseconds(AGENT_BROWSER_TIMEOUT_ENV, 30000);
const CHILD_SHUTDOWN_TIMEOUT_MS = readPositiveMilliseconds(CHILD_SHUTDOWN_TIMEOUT_ENV, 5000);
const HTTP_PROBE_TIMEOUT_MS = readPositiveMilliseconds(HTTP_PROBE_TIMEOUT_ENV, 2000);
const CHILD_KILL_TIMEOUT_MS = 1000;

function readPositiveMilliseconds(name, defaultValue) {
  const raw = (process.env[name] ?? "").trim();
  if (!raw) return defaultValue;

  const value = Number(raw);
  if (!Number.isFinite(value)) {
    throw new Error(`${name} must be a number`);
  }
  if (value <= 0) {
    throw new Error(`${name} must be > 0`);
  }
  return Math.max(1, Math.ceil(value));
}

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

async function fetchWithTimeout(url, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function waitForHttp(url, timeoutMs = 25000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    try {
      const remainingMs = Math.max(1, timeoutMs - (Date.now() - started));
      const response = await fetchWithTimeout(url, Math.min(HTTP_PROBE_TIMEOUT_MS, remainingMs));
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
    timeout: AGENT_BROWSER_TIMEOUT_MS,
  });
  const output = `${result.stdout ?? ""}${result.stderr ?? ""}`;
  if (result.error?.code === "ETIMEDOUT") {
    if (!allowFailure) {
      throw new Error(
        `agent-browser ${args.join(" ")} timed out after ${AGENT_BROWSER_TIMEOUT_MS}ms\n${output}`,
      );
    }
    return output;
  }
  if (result.error) {
    if (!allowFailure) {
      throw new Error(
        `agent-browser ${args.join(" ")} failed to start: ${result.error.message}\n${output}`,
      );
    }
    return output;
  }
  if (result.status !== 0 && !allowFailure) {
    throw new Error(
      `agent-browser ${args.join(" ")} failed\n${output}`,
    );
  }
  return output;
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

function childHasExited(child) {
  return child.exitCode !== null || child.signalCode !== null;
}

function waitForChildExit(child, timeoutMs) {
  return new Promise((resolve) => {
    let settled = false;
    let timer;
    const finish = (exited) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      child.off("exit", onExit);
      resolve(exited);
    };
    const onExit = () => finish(true);
    timer = setTimeout(() => finish(false), timeoutMs);
    child.once("exit", onExit);
    if (childHasExited(child)) {
      finish(true);
    }
  });
}

async function stopChild(child) {
  if (childHasExited(child)) {
    return;
  }

  const label = child.spawnargs?.join(" ") || `pid ${child.pid ?? "unknown"}`;
  try {
    child.kill();
  } catch (error) {
    console.warn(`Failed to stop ${label}: ${error.message}`);
    return;
  }

  if (await waitForChildExit(child, CHILD_SHUTDOWN_TIMEOUT_MS)) {
    return;
  }

  console.warn(
    `Timed out after ${CHILD_SHUTDOWN_TIMEOUT_MS}ms waiting for ${label}; sending SIGKILL`,
  );
  try {
    child.kill("SIGKILL");
  } catch (error) {
    console.warn(`Failed to force-stop ${label}: ${error.message}`);
    return;
  }

  if (!(await waitForChildExit(child, CHILD_KILL_TIMEOUT_MS))) {
    console.warn(`Timed out after ${CHILD_KILL_TIMEOUT_MS}ms waiting for ${label} after SIGKILL`);
  }
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
  runAgent(["wait", "--text", "100 MB / 2 slots"]);
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
  await Promise.all(children.map(stopChild));
}
