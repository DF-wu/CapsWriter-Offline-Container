import json
import os
import socket
import sys
from http.client import HTTPConnection, HTTPException


TRUE_VALUES = {"1", "true", "yes", "on"}


def env_port(name: str, default: int) -> int | None:
    raw = os.getenv(name, str(default)).strip()
    try:
        port = int(raw)
    except ValueError:
        return None
    if port < 1 or port > 65535:
        return None
    return port


def normalize_loopback_host(host: str) -> str:
    if host in ("", "0.0.0.0", "::", "[::]"):
        return "127.0.0.1"
    return host


def env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUE_VALUES


def check_websocket_port() -> bool:
    """
    對 WebSocket port 發送一個合法的 HTTP/1.1 GET, 預期 server 回 426 Upgrade Required。

    舊版只開裸 TCP 後立即 close, 會讓 websockets server 端 raise EOFError
    並寫整段 traceback 到 log; 每 30s healthcheck 一次累積大量 noise。
    送出合法 HTTP request 之後 server 走正常 reject 路徑, 不寫 traceback。
    """
    host = normalize_loopback_host(os.getenv("CAPSWRITER_SERVER_ADDR", "127.0.0.1"))
    port = env_port("CAPSWRITER_SERVER_PORT", 6016)
    if port is None:
        return False

    request = (
        "GET / HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "User-Agent: capswriter-healthcheck\r\n"
        "Connection: close\r\n"
        "\r\n"
    ).encode("ascii")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(3)
        try:
            sock.connect((host, port))
            sock.sendall(request)
            # Read a small chunk so server completes its response; ignore content.
            sock.recv(64)
        except OSError:
            return False

    return True


def check_http_readiness() -> bool:
    host = normalize_loopback_host(os.getenv("CAPSWRITER_HTTP_API_BIND", "127.0.0.1"))
    port = env_port("CAPSWRITER_HTTP_API_PORT", 6017)
    if port is None:
        return False
    conn = HTTPConnection(host, port, timeout=3)
    try:
        conn.request("GET", "/ready", headers={"User-Agent": "capswriter-healthcheck"})
        response = conn.getresponse()
        body = response.read(4096)
    except (HTTPException, OSError):
        return False
    finally:
        conn.close()

    if response.status != 200:
        return False
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return payload.get("status") == "ok"


def main() -> int:
    if not check_websocket_port():
        return 1
    if env_enabled("CAPSWRITER_HTTP_API_ENABLE") and not check_http_readiness():
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
