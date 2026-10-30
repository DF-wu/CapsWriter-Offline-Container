"""以作業系統檔案鎖限制每個使用者的麥克風客戶端只能執行一份。"""
from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import sys
import time

from .settings import settings_path


RESTART_WAIT_ARGUMENT = "--wait-for-client-exit"
RESTART_WAIT_SECONDS = 10.0


class ClientInstance:
    def __init__(self, path: Path | None = None):
        self.path = path if path is not None else settings_path().with_name("client-instance.lock")
        self._file = None

    def acquire(self, timeout: float = 0) -> bool:
        if self._file is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 鎖檔不可刪除或 replace，否則舊、新 inode 可被分別上鎖而失去互斥。
        stream = self.path.open("a+b")
        deadline = time.monotonic() + timeout
        while True:
            try:
                self._lock(stream)
                break
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    stream.close()
                    raise
                if time.monotonic() >= deadline:
                    stream.close()
                    return False
                time.sleep(min(0.05, max(0, deadline - time.monotonic())))
        self._file = stream
        try:
            if stream.seek(0, os.SEEK_END) == 0:
                stream.write(b"\0")
            stream.seek(1)
            stream.truncate()
            stream.write(json.dumps({"pid": os.getpid(), "executable": sys.executable}).encode("utf-8"))
            stream.flush()
        except BaseException:
            self.release()
            raise
        return True

    @staticmethod
    def _lock(stream) -> None:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def owner(self) -> dict:
        try:
            with self.path.open("rb") as stream:
                # Windows 的位元組鎖只涵蓋第一個 byte，其他程序可讀取其後的 PID。
                stream.seek(1)
                value = json.loads(stream.read(4096).decode("utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    def release(self) -> None:
        stream, self._file = self._file, None
        if stream is not None:
            # 關閉 handle 會釋放鎖；程序被強制結束時也由作業系統自動釋放。
            stream.close()


def report_status() -> int:
    instance = ClientInstance()
    if instance.acquire():
        instance.release()
        print("CapsWriter Client 未執行。")
        return 1
    print(f"CapsWriter Client 正在執行，PID={instance.owner().get('pid', '未知')}；Windows 程序名稱為 start_client.exe。")
    return 0
