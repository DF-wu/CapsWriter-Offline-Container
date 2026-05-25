# coding: utf-8
"""
bootstrap — fork_server 啟動編排

工作流:
1. apply_env_config()    把 env 變數塞回 ServerConfig 等 class 屬性
                         (必須在 import core.server.* 之前)
2. ForkedCapsWriterServer 子類化上游 CapsWriterServer:
   - 在 start() 中, 若 http_api_enable=true,
     用 asyncio.gather(socket_manager.start(), run_http_server())
     讓 WebSocket 與 HTTP API 並存於同一 loop。
   - 同時 monkey-patch `core.server.connection.server_manager.ws_send`
     為 ws_send_with_http (HTTP-aware 版本)。
"""

from __future__ import annotations
import asyncio


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
        - 若 http_api_enable=true, 用 asyncio.gather 同時跑 WebSocket
          send 迴圈與 HTTP API uvicorn server。
        - 否則完全等同上游, 跑 socket_manager.start()。
        """

        def start(self):
            # 防重入
            if self.is_alive:
                return
            self.is_alive = True

            # 沿用上游 setup
            from core.tools.signal_handler import register_signal
            register_signal(self.stop)
            self.tray_manager.start()
            self._print_banner()
            self.process_manager.start()

            # 分支點: HTTP API enable 與否
            if getattr(Config, "http_api_enable", False):
                from .http_api.serve import run_http_server

                # ws_send → ws_send_with_http (在啟動 socket_manager 之前 patch)
                _install_ws_send_hook()

                async def _main():
                    await asyncio.gather(
                        self.socket_manager.start(),
                        run_http_server(self),
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

    return ForkedCapsWriterServer()
