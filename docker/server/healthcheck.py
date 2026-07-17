import base64
import hashlib
import json
import os
import socket
import sys
import time
from http.client import HTTPConnection, HTTPException


TRUE_VALUES = {"1", "true", "yes", "on"}
WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
# Base64 for the required 16-byte nonce ``capswriter-probe``.
WEBSOCKET_KEY = "Y2Fwc3dyaXRlci1wcm9iZQ=="
MAX_WEBSOCKET_RESPONSE_HEADERS = 4096
WEBSOCKET_PROBE_TIMEOUT_SECONDS = 3.0
WEBSOCKET_CLOSE_ACK_TIMEOUT_SECONDS = 0.25


def env_port(name: str, default: int) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    raw = raw.strip()
    try:
        port = int(raw)
    except ValueError:
        return None
    if port < 1 or port > 65535:
        return None
    return port


def normalize_loopback_host(host: str) -> str:
    host = host.strip()
    if host in ("", "0.0.0.0"):
        return "127.0.0.1"
    if host in ("::", "[::]"):
        return "::1"
    if host.startswith("[") and host.endswith("]"):
        return host[1:-1]
    return host


def format_host_header(host: str, port: int) -> str:
    if ":" in host:
        return f"[{host}]:{port}"
    return f"{host}:{port}"


def env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUE_VALUES


def check_websocket_port() -> bool:
    """
    對 WebSocket port 完成 bounded RFC 6455 handshake，再送 masked close frame。

    裸 TCP 或一般 HTTP GET 都會讓 websockets 16 記錄 InvalidUpgrade traceback；
    healthcheck 每 30s 跑一次時會造成大量 production log noise。完整 handshake
    同時證明 port 上確實是 WebSocket server，close frame 則讓 app 正常回收连接。
    """
    host = normalize_loopback_host(os.getenv("CAPSWRITER_SERVER_ADDR", "127.0.0.1"))
    port = env_port("CAPSWRITER_SERVER_PORT", 6016)
    if port is None:
        return False

    request = (
        "GET / HTTP/1.1\r\n"
        f"Host: {format_host_header(host, port)}\r\n"
        "User-Agent: capswriter-healthcheck\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {WEBSOCKET_KEY}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "Sec-WebSocket-Protocol: binary\r\n"
        "\r\n"
    ).encode("ascii")
    expected_accept = base64.b64encode(
        hashlib.sha1(f"{WEBSOCKET_KEY}{WEBSOCKET_GUID}".encode("ascii")).digest()
    ).decode("ascii")
    deadline = time.monotonic() + WEBSOCKET_PROBE_TIMEOUT_SECONDS

    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        with socket.create_connection((host, port), timeout=remaining) as sock:
            sock.sendall(request)
            response = bytearray()
            while b"\r\n\r\n" not in response:
                if len(response) >= MAX_WEBSOCKET_RESPONSE_HEADERS:
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                sock.settimeout(remaining)
                chunk = sock.recv(
                    min(512, MAX_WEBSOCKET_RESPONSE_HEADERS - len(response))
                )
                if not chunk:
                    return False
                response.extend(chunk)

            header_block = bytes(response).split(b"\r\n\r\n", 1)[0]
            try:
                header_lines = header_block.decode("ascii").split("\r\n")
                status_parts = header_lines[0].split(" ", 2)
                status_code = int(status_parts[1])
                headers = {}
                for line in header_lines[1:]:
                    name, value = line.split(":", 1)
                    headers[name.strip().lower()] = value.strip()
            except (UnicodeDecodeError, ValueError, IndexError):
                return False

            if (
                status_code != 101
                or headers.get("upgrade", "").lower() != "websocket"
                or "upgrade"
                not in {
                    token.strip().lower()
                    for token in headers.get("connection", "").split(",")
                }
                or headers.get("sec-websocket-accept") != expected_accept
                or headers.get("sec-websocket-protocol", "").lower() != "binary"
            ):
                return False

            try:
                # Client frames must be masked. An empty close payload with a
                # zero masking key is valid and avoids server-side EOF noise.
                sock.sendall(b"\x88\x80\x00\x00\x00\x00")
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    sock.settimeout(
                        min(remaining, WEBSOCKET_CLOSE_ACK_TIMEOUT_SECONDS)
                    )
                    sock.recv(64)
            except OSError:
                # The 101 response already proves health; a raced close ACK must
                # not turn a healthy server unhealthy.
                pass
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
