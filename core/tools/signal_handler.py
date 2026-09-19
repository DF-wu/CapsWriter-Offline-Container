import signal
import sys
import time


class SignalHandler:
    def __init__(self, callback):
        self.last_time = None
        self.callback = callback
        self._stopping = False

    def __call__(self, signum, frame):
        if self._stopping:
            return
        # Only a person pressing Ctrl+C needs the confirmation window.
        # Service-manager termination must always request cleanup immediately.
        interactive = False
        if signum == signal.SIGINT:
            try:
                interactive = sys.stdin is not None and sys.stdin.isatty()
            except (OSError, ValueError):
                pass
        if interactive:
            now = time.monotonic()
            if self.last_time is None or now - self.last_time > 1.0:
                self.last_time = now
                print(f"\n收到 {signal.Signals(signum).name}，1秒内再次按下将会退出...")
                return

        self._stopping = True
        try:
            print(f"\n收到 {signal.Signals(signum).name}，确认退出...\n")
        except (OSError, ValueError):
            # A closed terminal or log pipe must never prevent cleanup.
            pass
        self.callback()


def register_signal(callback):
    handler = SignalHandler(callback)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
