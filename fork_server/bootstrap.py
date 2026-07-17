# coding: utf-8
"""
bootstrap — fork_server 啟動編排

工作流:
1. apply_env_config()    把 env 變數塞回 ServerConfig 等 class 屬性
                         (必須在 import core.server.* 之前)
2. ForkedCapsWriterServer 子類化上游 CapsWriterServer:
   - 在 start() 中, 若 http_api_enable=true,
     用 _run_fork_services() 監督 WebSocket、HTTP API 與 shutdown event；
     任一先完成就進入 bounded cleanup，避免留下另一個 listener。
   - 同時 monkey-patch `core.server.connection.server_manager.ws_send`
     為 ws_send_with_http (HTTP-aware 版本)。
"""

from __future__ import annotations
import asyncio


HTTP_SERVER_SHUTDOWN_WAIT_SECONDS = 8.0


async def _await_cleanup_task(cleanup_task: asyncio.Task):
    """Join an independent cleanup task despite repeated caller cancellation."""

    cancellation = None
    while True:
        try:
            result = await asyncio.shield(cleanup_task)
            break
        except asyncio.CancelledError as exc:
            cancellation = cancellation or exc
            if cleanup_task.cancelled():
                raise
            if cleanup_task.done():
                result = cleanup_task.result()
                break

    if cancellation is not None:
        raise cancellation
    return result


def _stop_http_enabled_server(server) -> None:
    """Synchronously request component shutdown without stopping the loop."""
    if not server.is_alive:
        return
    server.is_alive = False
    server._http_stop_requested = True

    from core.server import logger, console

    logger.info("=" * 50)
    logger.info("開始清理 Server 資源...")
    http_server = getattr(server, "_http_server", None)
    if http_server is not None:
        http_server.should_exit = True
    server.socket_manager.stop()
    server.process_manager.stop()
    server.tray_manager.stop()

    shutdown_event = getattr(server, "_http_shutdown_event", None)
    if shutdown_event is not None:
        try:
            server.loop.call_soon_threadsafe(shutdown_event.set)
        except RuntimeError:
            shutdown_event.set()
    logger.info("Server 資源清理訊號已送出")
    console.print('[green4]再見！')


async def _cleanup_fork_service_tasks(
    server,
    websocket_task: asyncio.Task,
    http_task: asyncio.Task,
    stop_task: asyncio.Task,
):
    """Signal, bound, and reap every cotask; safe on caller cancellation."""
    if server.is_alive:
        _stop_http_enabled_server(server)
    http_server = getattr(server, "_http_server", None)
    if http_server is not None:
        http_server.should_exit = True
    server.socket_manager.stop()

    http_error = None
    if not http_task.done():
        try:
            await asyncio.wait_for(
                asyncio.shield(http_task),
                timeout=HTTP_SERVER_SHUTDOWN_WAIT_SECONDS,
            )
        except asyncio.TimeoutError:
            http_server = getattr(server, "_http_server", None)
            if http_server is not None:
                http_server.force_exit = True
            http_task.cancel()
        except asyncio.CancelledError:
            if not http_task.cancelled():
                raise
        except Exception as exc:
            http_error = exc
    if http_task.done() and not http_task.cancelled():
        try:
            http_task.result()
        except Exception as exc:
            http_error = exc

    if not websocket_task.done():
        websocket_task.cancel()
    stop_task.cancel()
    await asyncio.gather(
        websocket_task,
        http_task,
        stop_task,
        return_exceptions=True,
    )
    return http_error


async def _run_fork_services(server, run_http_server) -> None:
    """Run both listeners and await bounded HTTP teardown on server.stop()."""
    websocket_task = asyncio.create_task(
        server.socket_manager.start(),
        name="capswriter-websocket-server",
    )
    http_task = asyncio.create_task(
        run_http_server(server),
        name="capswriter-http-server",
    )
    stop_task = asyncio.create_task(
        server._http_shutdown_event.wait(),
        name="capswriter-server-shutdown",
    )
    service_tasks = (websocket_task, http_task)
    external_stop = False
    failed_task = None
    http_error = None

    try:
        done, _pending = await asyncio.wait(
            (*service_tasks, stop_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        external_stop = stop_task in done
        failed_task = next(
            (task for task in service_tasks if task in done),
            None,
        )
        if not external_stop and server.is_alive:
            _stop_http_enabled_server(server)
    finally:
        cleanup_task = asyncio.create_task(
            _cleanup_fork_service_tasks(
                server,
                websocket_task,
                http_task,
                stop_task,
            ),
            name="capswriter-service-cleanup",
        )
        http_error = await _await_cleanup_task(cleanup_task)

    if not external_stop:
        if failed_task is not None and not failed_task.cancelled():
            failure = failed_task.exception()
            if failure is not None:
                raise failure
        if http_error is not None:
            raise http_error


def apply_env_config() -> None:
    """套用 env 覆寫到 ServerConfig 與引擎 Args classes (idempotent)。"""
    from . import env_config
    env_config.apply()


def _install_ws_send_hook() -> None:
    """
    用 ws_send_with_http 取代 server_manager 已 import 的 ws_send 符號。

    這比子類化 SocketManager 更小: 只動 module attribute, 不複製 setup 邏輯。
    """
    from core.server.connection import server_manager
    from .http_api.ws_send_with_http import ws_send_with_http
    server_manager.ws_send = ws_send_with_http


def create_server():
    """
    回傳已配置的 ForkedCapsWriterServer 實例。

    Note: 必須在 apply_env_config() 之後呼叫, 否則 core.server.* 的
    log_level 等 import-time snapshot 會錯誤。
    """
    from core.server.app import CapsWriterServer
    from config_server import ServerConfig as Config

    class ForkedCapsWriterServer(CapsWriterServer):
        """
        對 CapsWriterServer 的最小覆寫:
        - 若 http_api_enable=true, 以 supervised cotasks 同時跑 WebSocket
          send 迴圈與 HTTP API uvicorn server，並做 bounded teardown。
        - 否則完全等同上游, 跑 socket_manager.start()。
        """

        def start(self):
            # 防重入
            if self.is_alive:
                return
            self.is_alive = True
            self._http_stop_requested = False
            self._http_server = None

            # 沿用上游 setup
            from core.tools.signal_handler import register_signal
            register_signal(self.stop)
            self.tray_manager.start()
            self._print_banner()
            self.process_manager.start()
            if not self.is_alive:
                return

            # 分支點: HTTP API enable 與否
            if getattr(Config, "http_api_enable", False):
                from .http_api.serve import run_http_server

                # ws_send → ws_send_with_http (在啟動 socket_manager 之前 patch)
                _install_ws_send_hook()
                self._http_shutdown_event = asyncio.Event()

                async def _main():
                    await _run_fork_services(
                        self,
                        run_http_server,
                    )

                try:
                    self.loop.run_until_complete(_main())
                except RuntimeError:
                    pass
            else:
                # 上游原樣
                try:
                    self.loop.run_until_complete(self.socket_manager.start())
                except RuntimeError:
                    pass

        def stop(self):
            if getattr(Config, "http_api_enable", False):
                _stop_http_enabled_server(self)
                return
            super().stop()

    return ForkedCapsWriterServer()
