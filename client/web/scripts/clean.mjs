import { rm } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const paths = [
  "dist",
  "coverage",
  ".vite",
  "node_modules/.vite",
  "playwright-report",
  "test-results",
  ".tmp",
  "tsconfig.tsbuildinfo",
  "tsconfig.node.tsbuildinfo",
  "vite.config.js",
  "vite.config.d.ts",
];

await Promise.all(
paths.map((item) =>
    rm(join(root, item), { recursive: true, force: true }),
  ),
);
