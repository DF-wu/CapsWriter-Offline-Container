import asyncio
import unittest

from util.server.server_classes import Result
from util.server.server_cosmic import Cosmic
from util.server.task_router import TaskRouter


class TaskRouterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.original_socket_ids = Cosmic.sockets_id
        Cosmic.sockets_id = []
        self.router = TaskRouter()
        self.router.bind_loop(asyncio.get_running_loop())

    async def asyncTearDown(self):
        for task_id in list(self.router._pending):
            self.router.cancel(task_id)
        Cosmic.sockets_id = self.original_socket_ids

    async def test_final_result_resolves_and_cleans_synthetic_socket(self):
        future = self.router.register("task-1")
        result = Result("task-1", "http:task-1", "file", is_final=True)

        self.assertTrue(self.router.try_resolve(result))
        await asyncio.sleep(0)

        self.assertIs(await future, result)
        self.assertEqual(Cosmic.sockets_id, [])

    async def test_intermediate_result_is_absorbed_until_final(self):
        future = self.router.register("task-2")

        handled = self.router.try_resolve(
            Result("task-2", "http:task-2", "file", is_final=False)
        )

        self.assertTrue(handled)
        self.assertFalse(future.done())
        self.assertEqual(Cosmic.sockets_id, ["http:task-2"])

    async def test_cancel_removes_all_registration_state(self):
        future = self.router.register("task-3")

        self.router.cancel("task-3")

        self.assertTrue(future.cancelled())
        self.assertEqual(Cosmic.sockets_id, [])

    async def test_duplicate_task_id_is_rejected_without_leaking_socket(self):
        self.router.register("duplicate")

        with self.assertRaises(ValueError):
            self.router.register("duplicate")

        self.assertEqual(Cosmic.sockets_id, ["http:duplicate"])

    async def test_unknown_result_remains_available_to_websocket_path(self):
        result = Result("unknown", "socket", "mic", is_final=True)

        self.assertFalse(self.router.try_resolve(result))


if __name__ == "__main__":
    unittest.main()
