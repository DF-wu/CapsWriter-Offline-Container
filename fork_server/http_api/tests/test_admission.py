# coding: utf-8

from __future__ import annotations

import asyncio
import unittest

from fork_server.http_api.admission import (
    AdmissionController,
    AdmissionQueueFullError,
    ReplayableReceive,
)


class AdmissionControllerTest(unittest.TestCase):
    def test_disconnect_probe_replays_body_events_and_remembers_disconnect(self) -> None:
        async def scenario() -> None:
            messages = iter(
                (
                    {"type": "http.request", "body": b"first", "more_body": True},
                    {"type": "http.request", "body": b"second", "more_body": False},
                    {"type": "http.disconnect"},
                )
            )

            async def receive():
                return next(messages)

            replay = ReplayableReceive(receive)
            try:
                self.assertFalse(await replay.probe_disconnect())
                self.assertFalse(await replay.probe_disconnect())
                self.assertTrue(await replay.probe_disconnect())
                self.assertEqual(
                    await replay(),
                    {
                        "type": "http.request",
                        "body": b"firstsecond",
                        "more_body": False,
                    },
                )
                self.assertEqual(await replay(), {"type": "http.disconnect"})
                self.assertEqual(await replay(), {"type": "http.disconnect"})
            finally:
                replay.close()

        asyncio.run(scenario())

    def test_many_empty_probe_events_coalesce_to_one_replay_record(self) -> None:
        async def scenario() -> None:
            remaining = 10_000

            async def receive():
                nonlocal remaining
                if remaining:
                    remaining -= 1
                    return {
                        "type": "http.request",
                        "body": b"",
                        "more_body": True,
                    }
                return {"type": "http.disconnect"}

            replay = ReplayableReceive(receive)
            try:
                await replay.wait_for_disconnect()
                self.assertEqual(replay.buffered_event_count, 1)
                self.assertEqual(
                    await replay(),
                    {"type": "http.request", "body": b"", "more_body": True},
                )
                self.assertEqual(await replay(), {"type": "http.disconnect"})
            finally:
                replay.close()

        asyncio.run(scenario())

    def test_queue_is_bounded_and_releases_in_fifo_order(self) -> None:
        async def scenario() -> None:
            controller = AdmissionController(max_active=1, max_pending=2)
            order: list[str] = []

            async def queued(name: str) -> None:
                async with controller.slot():
                    order.append(name)

            async with controller.slot():
                first = asyncio.create_task(queued("first"))
                second = asyncio.create_task(queued("second"))
                while controller.waiting < 2:
                    await asyncio.sleep(0)
                with self.assertRaises(AdmissionQueueFullError):
                    async with controller.slot():
                        self.fail("a full admission queue granted a slot")

            await asyncio.gather(first, second)
            self.assertEqual(order, ["first", "second"])
            self.assertEqual(controller.active, 0)
            self.assertEqual(controller.waiting, 0)

        asyncio.run(scenario())

    def test_cancelled_waiter_does_not_leak_capacity(self) -> None:
        async def scenario() -> None:
            controller = AdmissionController(max_active=1, max_pending=1)

            async def queued() -> None:
                async with controller.slot():
                    self.fail("cancelled waiter entered the active section")

            async with controller.slot():
                task = asyncio.create_task(queued())
                while controller.waiting < 1:
                    await asyncio.sleep(0)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertEqual(controller.waiting, 0)

            async with controller.slot():
                self.assertEqual(controller.active, 1)
            self.assertEqual(controller.active, 0)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
