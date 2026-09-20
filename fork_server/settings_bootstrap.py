"""Resolve saved daily settings before Docker hardware probing and downloads."""
from __future__ import annotations
import os
import sys
from fork_server.settings import SettingsStore
from fork_server.http_api.runtime_config import ConfigError


def bootstrap_environment(environ):
    store = SettingsStore(environ)
    merged = store.environment()
    injected = [key for key in merged if merged[key] != environ.get(key)]
    merged["CAPSWRITER_SETTINGS_INJECTED_KEYS"] = ",".join(injected)
    merged["CAPSWRITER_SETTINGS_BOOTSTRAPPED"] = "1"
    return merged


def main():
    try:
        environment = bootstrap_environment(os.environ)
    except ConfigError as exc:
        print(f"CapsWriter configuration error: {exc}", file=sys.stderr)
        return 2
    os.execvpe("bash", ["bash", *sys.argv[1:]], environment)


if __name__ == "__main__":
    raise SystemExit(main())
