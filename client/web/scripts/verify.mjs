import { spawnSync } from "node:child_process";

function run(args) {
  const result = spawnSync("npm", args, {
    shell: process.platform === "win32",
    stdio: "inherit",
  });
  return result.status ?? 1;
}

let status = run(["run", "test"]);
if (status === 0) {
  status = run(["run", "build"]);
}

const cleanStatus = run(["run", "clean"]);
process.exit(status || cleanStatus);
