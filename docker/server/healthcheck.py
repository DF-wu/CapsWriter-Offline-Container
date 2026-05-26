import os
import socket
import sys


def main() -> int:
    """
    對 WebSocket port 發送一個合法的 HTTP/1.1 GET, 預期 server 回 426 Upgrade Required。

    舊版只開裸 TCP 後立即 close, 會讓 websockets server 端 raise EOFError
    並寫整段 traceback 到 log; 每 30s healthcheck 一次累積大量 noise。
    送出合法 HTTP request 之後 server 走正常 reject 路徑, 不寫 traceback。
    """
    host = os.getenv("CAPSWRITER_SERVER_ADDR", "127.0.0.1")
    if host == "0.0.0.0":
        host = "127.0.0.1"
    port = int(os.getenv("CAPSWRITER_SERVER_PORT", "6016"))

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
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
