import time
from types import SimpleNamespace
import unittest

from util.server.server_classes import Task
from util.server.server_recognize import clear_results_by_socket_id, recognize


class FakeStream:
    def __init__(self, text):
        self.result = SimpleNamespace(text=text, tokens=[], timestamps=[])

    def accept_waveform(self, samplerate, samples):
        del samplerate, samples


class FakeRecognizer:
    def __init__(self, texts):
        self._texts = iter(texts)

    def create_stream(self):
        return FakeStream(next(self._texts))

    def decode_stream(self, stream, **kwargs):
        del stream, kwargs


def audio_task(socket_id, *, is_final):
    now = time.time()
    return Task(
        source="file",
        data=b"\x00" * (1600 * 4),
        offset=0.0,
        overlap=0.0,
        task_id="shared-client-id",
        socket_id=socket_id,
        is_final=is_final,
        time_start=now,
        time_submit=now,
    )


class RecognitionStateIsolationTests(unittest.TestCase):
    def tearDown(self):
        clear_results_by_socket_id("socket-a")
        clear_results_by_socket_id("socket-b")

    def test_same_task_id_on_different_sockets_does_not_merge_or_misroute(self):
        recognizer = FakeRecognizer(("alpha", "beta"))

        first = recognize(
            recognizer,
            None,
            audio_task("socket-a", is_final=False),
        )
        second = recognize(
            recognizer,
            None,
            audio_task("socket-b", is_final=True),
        )

        self.assertEqual(first.socket_id, "socket-a")
        self.assertEqual(second.socket_id, "socket-b")
        self.assertEqual(second.task_id, "shared-client-id")
        self.assertEqual(second.text, "beta")


if __name__ == "__main__":
    unittest.main()
