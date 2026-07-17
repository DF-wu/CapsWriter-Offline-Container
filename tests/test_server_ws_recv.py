import base64
import unittest
from unittest.mock import patch

from util.constants import AudioFormat
from util.server.server_cosmic import Cosmic
from util.server.server_ws_recv import (
    AudioCache,
    InvalidAudioMessage,
    MAX_AUDIO_CHUNK_BYTES,
    message_handler,
    validate_audio_message,
)


class FakeQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


class FakeWebSocket:
    id = "socket-1"


class QuietStatus:
    on = False

    def start(self):
        self.on = True

    def stop(self):
        self.on = False


def audio_message(raw: bytes = b"", **overrides):
    message = {
        "task_id": "task-1",
        "source": "mic",
        "data": base64.b64encode(raw).decode("ascii"),
        "is_final": False,
        "time_start": 123.0,
        "seg_duration": 60.0,
        "seg_overlap": 4.0,
        "context": "",
    }
    message.update(overrides)
    return message


class AudioMessageValidationTests(unittest.TestCase):
    def test_official_client_parameters_and_chunk_size_are_accepted(self):
        raw = b"\x00" * AudioFormat.seconds_to_bytes(60)

        self.assertEqual(validate_audio_message(audio_message(raw)), raw)

    def test_zero_segment_duration_is_rejected(self):
        with self.assertRaisesRegex(InvalidAudioMessage, "seg_duration"):
            validate_audio_message(audio_message(seg_duration=0))

    def test_sub_sample_and_unaligned_segment_geometry_are_rejected(self):
        with self.assertRaisesRegex(InvalidAudioMessage, "one audio sample"):
            validate_audio_message(audio_message(seg_duration=0.000001))
        with self.assertRaisesRegex(InvalidAudioMessage, "sample-aligned"):
            validate_audio_message(audio_message(seg_duration=5 / 64000))

    def test_non_finite_and_negative_segment_values_are_rejected(self):
        with self.assertRaisesRegex(InvalidAudioMessage, "seg_duration"):
            validate_audio_message(audio_message(seg_duration=float("nan")))
        with self.assertRaisesRegex(InvalidAudioMessage, "seg_overlap"):
            validate_audio_message(audio_message(seg_overlap=-1))

    def test_invalid_base64_and_unaligned_pcm_are_rejected(self):
        with self.assertRaisesRegex(InvalidAudioMessage, "Base64"):
            validate_audio_message(audio_message(data="%%%"))
        with self.assertRaisesRegex(InvalidAudioMessage, "sample-aligned"):
            validate_audio_message(audio_message(b"abc"))

    def test_oversized_decoded_chunk_is_rejected(self):
        raw = b"\x00" * (MAX_AUDIO_CHUNK_BYTES + AudioFormat.BYTES_PER_SAMPLE)

        with self.assertRaisesRegex(InvalidAudioMessage, "exceeds"):
            validate_audio_message(audio_message(raw))

    def test_message_shape_and_bounded_metadata_are_enforced(self):
        with self.assertRaisesRegex(InvalidAudioMessage, "source"):
            validate_audio_message(audio_message(source="network"))
        with self.assertRaisesRegex(InvalidAudioMessage, "is_final"):
            validate_audio_message(audio_message(is_final="false"))
        with self.assertRaisesRegex(InvalidAudioMessage, "task_id"):
            validate_audio_message(audio_message(task_id=""))
        with self.assertRaisesRegex(InvalidAudioMessage, "control"):
            validate_audio_message(audio_message(task_id="task\nforged-log"))
        with self.assertRaisesRegex(InvalidAudioMessage, "context"):
            validate_audio_message(audio_message(context="x" * 8193))


class MessageHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.original_queue = Cosmic.queue_in
        self.queue = FakeQueue()
        Cosmic.queue_in = self.queue
        self.status_patch = patch(
            "util.server.server_ws_recv.status_mic", QuietStatus()
        )
        self.status_patch.start()

    async def asyncTearDown(self):
        self.status_patch.stop()
        Cosmic.queue_in = self.original_queue

    async def test_valid_audio_is_segmented_and_finalized_in_order(self):
        cache = AudioCache()
        two_seconds = b"\x00" * AudioFormat.seconds_to_bytes(2.0)

        await message_handler(
            FakeWebSocket(),
            audio_message(two_seconds, seg_duration=1.0, seg_overlap=0.25),
            cache,
        )
        await message_handler(
            FakeWebSocket(),
            audio_message(
                is_final=True,
                seg_duration=1.0,
                seg_overlap=0.25,
            ),
            cache,
        )

        self.assertEqual(len(self.queue.items), 2)
        self.assertFalse(self.queue.items[0].is_final)
        self.assertTrue(self.queue.items[1].is_final)
        self.assertEqual(self.queue.items[0].offset, 0.0)
        self.assertEqual(self.queue.items[1].offset, 1.0)
        self.assertIsNone(cache.task_id)

    async def test_standard_mic_final_packet_may_use_legacy_15_2_settings(self):
        cache = AudioCache()
        raw = b"\x00" * AudioFormat.seconds_to_bytes(0.5)
        await message_handler(FakeWebSocket(), audio_message(raw), cache)

        await message_handler(
            FakeWebSocket(),
            audio_message(is_final=True, seg_duration=15.0, seg_overlap=2.0),
            cache,
        )

        self.assertTrue(self.queue.items[-1].is_final)

    async def test_task_cannot_change_mid_stream(self):
        cache = AudioCache()
        raw = b"\x00" * AudioFormat.seconds_to_bytes(0.5)
        await message_handler(FakeWebSocket(), audio_message(raw), cache)

        with self.assertRaisesRegex(InvalidAudioMessage, "changed"):
            await message_handler(
                FakeWebSocket(), audio_message(raw, task_id="task-2"), cache
            )


if __name__ == "__main__":
    unittest.main()
