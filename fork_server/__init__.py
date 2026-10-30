# coding: utf-8
"""
fork_server — Linux container 化貼皮層 (sidecar)

本套件提供上游 CapsWriter-Offline 的容器、設定與 HTTP 整合層：

- env_config:  把 env 變數塞回上游 config_server.* 的 class 屬性, 讓
               docker-compose 設定能直接驅動上游邏輯。
- bootstrap:   ForkedCapsWriterServer — 子類化上游 CapsWriterServer,
               在 HTTP API 啟用時並行運行 ws_send 與 uvicorn。
- http_api/:   OpenAI Whisper API 相容的 REST 端點。共用識別子進程,
               透過 sentinel socket_id="http:<task_id>" 路由結果。

上游 core 的必要修改由 scripts/check_upstream_divergence.py 與回歸測試追蹤；
同步時參見 docs/upstream-sync-guide.md。
"""

__version__ = "2.1.0"

__all__ = ["bootstrap"]
