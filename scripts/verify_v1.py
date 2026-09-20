"""Run v1 branch, core, HTTP API and container regression suites."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SUITES = ("tests", "scripts/tests", "fork_server/http_api/tests", "docker/server/tests")

def main():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    for suite in SUITES:
        print(f"\nRunning {suite}", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", suite, "-p", "test_*.py"],
            cwd=ROOT, env=env,
        )
        if result.returncode:
            return result.returncode
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
