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

    def test_set_result_ignores_already_cancelled_future(self) -> None:
        async def scenario() -> None:
            fut = asyncio.get_running_loop().create_future()
            fut.cancel()
            TaskRouter._set_result(fut, object())
            self.assertTrue(fut.cancelled())

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
