# coding: utf-8
"""
ws_send_with_http — 上游 ws_send 的 fork 版本

差異:
- 從 queue_out 取到 Result 後, 先呼叫 task_router.try_resolve()
  — 若是 HTTP 任務則 resolve future 並 continue, 跳過原 ws 派發。

此檔的主體必須與上游 `core/server/connection/ws_send.py` 保持邏輯一致。
上游 ws_send 若變更 (新欄位、新訊號處理…), 需 re-port 到此檔。
請參考 docs/upstream-sync-guide.md §關鍵漂移點 章節。

最後 re-port 自上游版本: origin/master @ 7d7fac3 (upstream v2.6)
"""

from __future__ import annotations

from core.server.schema import Result
from core.protocol import RecognitionMessage
from core.server.state import console
from core.server import logger
from core.server.connection.result_dispatcher import (
    AsyncResultQueueReader,
    WebSocketResultDispatcher,
)

from .task_router import router as task_router


async def ws_send_with_http(app):
    state = app.state
    queue_out = state.queue_out
    sockets = state.sockets
    reader = AsyncResultQueueReader(queue_out)
    dispatcher = WebSocketResultDispatcher(state, logger)

    logger.info("WebSocket 發送任務已啟動 (HTTP-aware)")

    try:
        while True:
            try:
                result: Result = await reader.get()

                # 退出訊號
                if result is None:
                    logger.info("收到退出通知，停止發送任務")
                    return

                # === FORK ADDITION: HTTP first chance ===
                # try_resolve 對 HTTP 任務的 final result 會 set_result Future 並返回 True,
                # 對 HTTP 任務的中間結果也返回 True (吞掉), 對 ws 任務返回 False。
                if task_router.try_resolve(result):
                    continue
                # === END FORK ADDITION ===

                # 1. 轉為協議訊息
                msg = RecognitionMessage(
                    task_id=result.task_id,
                    is_final=result.is_final,
                    duration=result.duration,
                    time_start=result.time_start,
                    time_submit=result.time_submit,
                    time_complete=result.time_complete,
                    text=result.text,
                    text_accu=result.text_accu,
                    tokens=result.tokens,
                    timestamps=result.timestamps,
                    error_code=result.error_code,
                    error_message=result.error_message
                )

                # 找對應 websocket
                websocket = next(
                    (ws for ws in sockets.values() if str(ws.id) == result.socket_id),
                    None,
                )

                if not websocket:
                    logger.warning(f"客戶端 {result.socket_id} 不存在，跳過發送結果，任務ID: {result.task_id}")
                    continue

                # Per-peer send work is bounded and isolated from this shared
                # queue/HTTP dispatcher.
                dispatcher.submit(
                    websocket,
                    msg.to_json(),
                    socket_id=result.socket_id,
                    task_id=result.task_id,
                    is_final=result.is_final,
                )
                logger.debug(f"排隊發送識別結果，任務ID: {result.task_id}, 文本長度: {len(result.text)}")

                if result.type == 'mic':
                    logger.info(f"麥克風識別結果: {result.text}")
                elif result.type == 'file':
                    console.print(f'    轉錄進度：{result.duration:.2f}s', end='\r')
                    logger.debug(f"檔案轉錄進度: {result.duration:.2f}s")
                    if result.is_final:
                        console.print('\n    [green]轉錄完成')
                        logger.info(f"檔案轉錄完成，任務ID: {result.task_id}, 總時長: {result.duration:.2f}s")

            except Exception as e:
                logger.error(f"發送結果時發生錯誤: {e}", exc_info=True)
                print(e)
    finally:
        await reader.aclose()
        await dispatcher.aclose()
