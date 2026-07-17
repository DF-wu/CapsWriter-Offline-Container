"""
Lightweight configured-backend probe.

Tries to instantiate the configured ASR engine via EngineFactory.
If construction fails (e.g. CUDA/Vulkan libraries missing, model files missing,
GPU OOM), exit non-zero so entrypoint.sh can switch every backend to CPU and
validate the fallback before starting the long-running server.

Used whenever any CUDA or Vulkan path is selected and once more after a failed
GPU probe has selected the full CPU path. The main server has the same engine
construction behavior, but this probe lets the entrypoint replace backend
runtime files before the long-running supervisor starts.
"""
import os
import math
import signal
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if ROOT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, ROOT_DIR.as_posix())

from config_server import ServerConfig


BACKEND_PROBE_TIMEOUT_ENV = "CAPSWRITER_BACKEND_PROBE_TIMEOUT"
DEFAULT_BACKEND_PROBE_TIMEOUT_SECONDS = 300.0
MAX_BACKEND_PROBE_TIMEOUT_SECONDS = 1800.0
BACKEND_PROBE_KILL_GRACE_SECONDS = 5.0


def backend_probe_timeout_seconds() -> float:
    raw = os.environ.get(BACKEND_PROBE_TIMEOUT_ENV, "").strip()
    if not raw:
        return DEFAULT_BACKEND_PROBE_TIMEOUT_SECONDS
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise ValueError(f"{BACKEND_PROBE_TIMEOUT_ENV} must be a number") from exc
    if (
        not math.isfinite(timeout)
        or timeout <= 0
        or timeout > MAX_BACKEND_PROBE_TIMEOUT_SECONDS
    ):
        raise ValueError(
            f"{BACKEND_PROBE_TIMEOUT_ENV} must be > 0 and <= "
            f"{MAX_BACKEND_PROBE_TIMEOUT_SECONDS:g}"
        )
    return timeout


def main() -> int:
    # entrypoint.sh 在 start_server_docker.py 之前跑 probe，所以必須在 import
    # EngineFactory 前套用完整 env contract。只讀 model type 會讓 CUDA/Vulkan
    # probe 沿用 upstream 的 CPU/default Args，無法驗證實際要啟動的 backend。
    try:
        from fork_server.env_config import apply as apply_env_config

        apply_env_config()
    except Exception as error:
        print(f"[capswriter] configured backend probe failed: {error}", file=sys.stderr)
        return 1

    model_type = str(ServerConfig.model_type).strip().lower()

    # Skip probe for non-GGUF engines (sensevoice/paraformer don't load llama.cpp).
    if model_type not in {"qwen_asr", "fun_asr_nano"}:
        print(f"[capswriter] probe skipped: {model_type} does not use llama.cpp")
        return 0

    try:
        from core.server.engines.factory import EngineFactory
        engine = EngineFactory.create_asr_engine(model_type)
    except Exception as error:
        print(f"[capswriter] configured backend probe failed: {error}", file=sys.stderr)
        return 1

    try:
        engine.cleanup()
    except Exception:
        pass

    backend = os.environ.get("CAPSWRITER_LLAMA_BACKEND", "cpu")
    print(
        f"[capswriter] configured backend probe passed for {model_type} "
        f"(llama={backend})"
    )
    return 0


def terminate_probe_process_tree(process: subprocess.Popen[bytes]) -> None:
    """Kill and reap the isolated probe session, including descendants."""

    try:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        try:
            process.kill()
        except (OSError, ProcessLookupError):
            pass
    try:
        process.wait(timeout=BACKEND_PROBE_KILL_GRACE_SECONDS)
    except (OSError, subprocess.TimeoutExpired):
        pass


def supervised_main() -> int:
    """Run native backend construction in a killable, deadline-bounded child."""

    try:
        timeout = backend_probe_timeout_seconds()
    except ValueError as error:
        print(f"[capswriter] configured backend probe failed: {error}", file=sys.stderr)
        return 1
    try:
        process = subprocess.Popen(
            [sys.executable, Path(__file__).resolve().as_posix(), "--worker"],
            start_new_session=os.name != "nt",
        )
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        terminate_probe_process_tree(process)
        print(
            f"[capswriter] configured backend probe timed out after {timeout:g}s",
            file=sys.stderr,
        )
        return 1
    except OSError as error:
        print(
            f"[capswriter] configured backend probe failed to start: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        sys.exit(main())
    if sys.argv[1:]:
        print("usage: probe_backend.py", file=sys.stderr)
        sys.exit(2)
    sys.exit(supervised_main())
