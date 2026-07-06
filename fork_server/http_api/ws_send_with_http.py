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
from core.tools.asyncio_to_thread import to_thread
from core.server.state import console
from core.server import logger

from .task_router import router as task_router


async def ws_send_with_http(app):
    state = app.state
    queue_out = state.queue_out
    sockets = state.sockets

    logger.info("WebSocket 發送任務已啟動 (HTTP-aware)")

    while True:
        try:
            # 取結果 (blocking — 推到 thread)
            result: Result = await to_thread(queue_out.get)

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
                timestamps=result.timestamps
            )

            # 找對應 websocket
            websocket = next(
                (ws for ws in sockets.values() if str(ws.id) == result.socket_id),
                None,
            )

            if not websocket:
                logger.warning(f"客戶端 {result.socket_id} 不存在，跳過發送結果，任務ID: {result.task_id}")
                continue

            # 發送訊息
            await websocket.send(msg.to_json())
            logger.debug(f"發送識別結果，任務ID: {result.task_id}, 文本長度: {len(result.text)}")

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
