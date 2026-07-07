# coding: utf-8

from __future__ import annotations

import asyncio
from types import SimpleNamespace
import unittest

from fork_server.http_api.task_router import TaskRouter


class TaskRouterTest(unittest.TestCase):
    def test_register_and_cancel_manage_synthetic_socket(self) -> None:
        async def scenario() -> None:
            state = SimpleNamespace(sockets_id=[], queue_in=object())
            router = TaskRouter()
            router.bind(state, asyncio.get_running_loop())

            fut = router.register("abc")

            self.assertFalse(fut.done())
            self.assertEqual(state.sockets_id, ["http:abc"])
            router.cancel("abc")
            self.assertTrue(fut.cancelled())
            self.assertEqual(state.sockets_id, [])

        asyncio.run(scenario())

    def test_late_final_result_after_cancel_is_absorbed(self) -> None:
        async def scenario() -> None:
            state = SimpleNamespace(sockets_id=[], queue_in=object())
            router = TaskRouter()
            router.bind(state, asyncio.get_running_loop())
            fut = router.register("abc")

            router.cancel("abc")
            handled = router.try_resolve(
                SimpleNamespace(
                    task_id="abc",
                    socket_id="http:abc",
                    is_final=True,
                )
            )

            self.assertTrue(handled)
            self.assertTrue(fut.cancelled())
            self.assertEqual(state.sockets_id, [])

        asyncio.run(scenario())

    def test_late_intermediate_result_after_cancel_keeps_tombstone(self) -> None:
        async def scenario() -> None:
            state = SimpleNamespace(sockets_id=[], queue_in=object())
            router = TaskRouter()
            router.bind(state, asyncio.get_running_loop())
            fut = router.register("abc")

            router.cancel("abc")
            intermediate_handled = router.try_resolve(
                SimpleNamespace(
                    task_id="abc",
                    socket_id="http:abc",
                    is_final=False,
                )
            )
            final_handled = router.try_resolve(
                SimpleNamespace(
                    task_id="abc",
                    socket_id="http:abc",
                    is_final=True,
                )
            )

            self.assertTrue(intermediate_handled)
            self.assertTrue(final_handled)
            self.assertTrue(fut.cancelled())
            self.assertEqual(state.sockets_id, [])

        asyncio.run(scenario())

    def test_cancel_without_pending_task_does_not_absorb_unknown_result(self) -> None:
        router = TaskRouter()

        router.cancel("abc")

        self.assertFalse(
            router.try_resolve(
                SimpleNamespace(
                    task_id="abc",
                    socket_id="http:abc",
                    is_final=True,
                )
            )
        )

    def test_cancelled_task_does_not_absorb_non_http_socket_result(self) -> None:
        async def scenario() -> None:
            state = SimpleNamespace(sockets_id=[], queue_in=object())
            router = TaskRouter()
            router.bind(state, asyncio.get_running_loop())
            router.register("abc")

            router.cancel("abc")

            self.assertFalse(
                router.try_resolve(
                    SimpleNamespace(
                        task_id="abc",
                        socket_id="websocket-1",
                        is_final=True,
                    )
                )
            )

        asyncio.run(scenario())

    def test_try_resolve_ignores_non_http_results(self) -> None:
        router = TaskRouter()
        result = SimpleNamespace(task_id="missing", is_final=True)

        self.assertFalse(router.try_resolve(result))

    def test_try_resolve_absorbs_intermediate_result_without_cleanup(self) -> None:
        async def scenario() -> None:
            state = SimpleNamespace(sockets_id=[], queue_in=object())
            router = TaskRouter()
            router.bind(state, asyncio.get_running_loop())
            fut = router.register("abc")

            handled = router.try_resolve(SimpleNamespace(task_id="abc", is_final=False))

            self.assertTrue(handled)
            self.assertFalse(fut.done())
            self.assertEqual(state.sockets_id, ["http:abc"])

        asyncio.run(scenario())

    def test_try_resolve_final_result_sets_future_and_cleans_socket(self) -> None:
        async def scenario() -> None:
            state = SimpleNamespace(sockets_id=[], queue_in=object())
            router = TaskRouter()
            router.bind(state, asyncio.get_running_loop())
            fut = router.register("abc")
            result = SimpleNamespace(task_id="abc", is_final=True, text="done")

            handled = router.try_resolve(result)
            await asyncio.sleep(0)

            self.assertTrue(handled)
            self.assertIs(fut.result(), result)
            self.assertEqual(state.sockets_id, [])

        asyncio.run(scenario())

    def test_cancelled_tombstones_expire(self) -> None:
        now = {"value": 100.0}

        async def scenario() -> None:
            def monotonic() -> float:
                return now["value"]

            state = SimpleNamespace(sockets_id=[], queue_in=object())
            router = TaskRouter(
                cancelled_tombstone_ttl_seconds=10.0,
                monotonic=monotonic,
            )
            router.bind(state, asyncio.get_running_loop())
            router.register("abc")
            router.cancel("abc")
            now["value"] = 111.0

            self.assertFalse(
                router.try_resolve(
                    SimpleNamespace(
                        task_id="abc",
                        socket_id="http:abc",
                        is_final=True,
                    )
                )
            )

        asyncio.run(scenario())

    def test_cancelled_tombstones_are_size_bounded(self) -> None:
        async def scenario() -> None:
            now = {"value": 100.0}

            def monotonic() -> float:
                return now["value"]

            state = SimpleNamespace(sockets_id=[], queue_in=object())
            router = TaskRouter(
                max_cancelled_tombstones=2,
                monotonic=monotonic,
            )
            router.bind(state, asyncio.get_running_loop())
            for task_id in ("a", "b", "c"):
                router.register(task_id)
                router.cancel(task_id)
                now["value"] += 1.0

            self.assertFalse(
                router.try_resolve(
                    SimpleNamespace(
                        task_id="a",
                        socket_id="http:a",
                        is_final=True,
                    )
                )
            )
            self.assertTrue(
                router.try_resolve(
                    SimpleNamespace(
                        task_id="b",
                        socket_id="http:b",
                        is_final=True,
                    )
                )
            )
            self.assertTrue(
                router.try_resolve(
                    SimpleNamespace(
                        task_id="c",
                        socket_id="http:c",
                        is_final=True,
                    )
                )
            )

        asyncio.run(scenario())

    def test_set_result_ignores_already_cancelled_future(self) -> None:
        async def scenario() -> None:
            fut = asyncio.get_running_loop().create_future()
            fut.cancel()
            TaskRouter._set_result(fut, object())
            self.assertTrue(fut.cancelled())

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
