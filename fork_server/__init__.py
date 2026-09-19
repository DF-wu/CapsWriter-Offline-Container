# coding: utf-8
"""
fork_server — Linux container 化貼皮層 (sidecar)

本套件是上游 CapsWriter-Offline 的純加值層, 不修改任何 upstream 檔案:

- env_config:  把 env 變數塞回上游 config_server.* 的 class 屬性, 讓
               docker-compose 設定能直接驅動上游邏輯。
- bootstrap:   ForkedCapsWriterServer — 子類化上游 CapsWriterServer,
               在 HTTP API 啟用時並行運行 ws_send 與 uvicorn。
- http_api/:   OpenAI Whisper API 相容的 REST 端點。共用識別子進程,
               透過 sentinel socket_id="http:<task_id>" 路由結果。

設計原則 (參見 docs/architecture.md):
- 修改上游檔案數 = 0。
- 唯一漂移點: http_api/ws_send_with_http.py 內嵌了上游 ws_send 邏輯複本,
  上游若改 ws_send 簽名需手動 re-port (參見 docs/upstream-sync-guide.md)。
"""

__all__ = ["bootstrap"]
