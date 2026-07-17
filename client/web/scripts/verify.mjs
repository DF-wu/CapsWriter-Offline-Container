import { spawnSync } from "node:child_process";

const WEB_VERIFY_STEP_TIMEOUT_ENV = "CAPSWRITER_WEB_VERIFY_STEP_TIMEOUT";
const DEFAULT_WEB_VERIFY_STEP_TIMEOUT_SECONDS = 600;
const TIMEOUT_EXIT_CODE = 124;

function verifyStepTimeoutMs() {
  const raw = (process.env[WEB_VERIFY_STEP_TIMEOUT_ENV] ?? "").trim();
  if (!raw) return DEFAULT_WEB_VERIFY_STEP_TIMEOUT_SECONDS * 1000;

  const seconds = Number(raw);
  if (!Number.isFinite(seconds)) {
    throw new Error(`${WEB_VERIFY_STEP_TIMEOUT_ENV} must be a number`);
  }
  if (seconds <= 0) {
    throw new Error(`${WEB_VERIFY_STEP_TIMEOUT_ENV} must be > 0`);
  }
  return Math.max(1, Math.ceil(seconds * 1000));
}

function formatSeconds(timeoutMs) {
  return String(timeoutMs / 1000);
}

function run(args) {
  const command = ["npm", ...args].join(" ");
  const result = spawnSync("npm", args, {
    shell: process.platform === "win32",
    stdio: "inherit",
    timeout: stepTimeoutMs,
  });
  if (result.error?.code === "ETIMEDOUT") {
    console.error(`Command timed out after ${formatSeconds(stepTimeoutMs)}s: ${command}`);
    return TIMEOUT_EXIT_CODE;
  }
  if (result.error) {
    console.error(`Command failed to start: ${command}: ${result.error.message}`);
    return 1;
  }
  return result.status ?? 1;
}

let stepTimeoutMs;
try {
  stepTimeoutMs = verifyStepTimeoutMs();
} catch (error) {
  console.error(error.message);
  process.exit(1);
}

let status = run(["run", "test"]);
if (status === 0) {
  status = run(["run", "build"]);
}

const cleanStatus = run(["run", "clean"]);
process.exit(status || cleanStatus);
