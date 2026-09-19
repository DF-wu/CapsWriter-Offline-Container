# 2026-09-18 上游同步紀錄

本次以 `HaujetZhao/CapsWriter-Offline` 的 `84912d5` 為基準，吸收自
`7d7fac3` 之後 13 個提交。v2 的 merge commit 為 `a1cdd45`。
同步在新分支上完成；沒有升級 axolotl 的既有服務。

## 使用行為

- 跨分片 token 拼接遇到 token 內部切點時，保留剩餘字元與時間戳；
  `chalk?` 接 `? Just` 不再遺失空格，較早的歷史片段也不會被截掉。
- 特殊按鍵由 pynput 的 `keyboard.Key` enum 產生，涵蓋 Insert、Home、
  End、方向鍵等；維持 fork 的延遲 import，避免 Linux/headless import 失敗。
- 取消錄音後清除全域 ingress queue。保留 fork 已有的 callback reservation
  釋放、檔案收尾、重複取消防護、依序傳送與 server partial stream 取消機制。
- 修正合併檢查時發現的錄音資料缺漏：跨過快捷鍵錄音門檻的當前區塊，
  現在會與門檻前快取一併送出，不會遺失一個音訊區塊。
- 托盤增加開啟當月日記目錄；使用既有跨平台預設程式 opener，
  啟動失敗會記錄訊息，避免回呼直接拋出錯誤。
- 中文數字正規化保留「点一下」、「万一」、「万三」等非普通數值語意，
  一般數值如「一万三」及「三点一四」仍正常轉换。
- 字幕採兩階段分行，保留縮寫、小數與較短片語；由下一行起點決定行尾，
  控制字幕之間間隙及長靜默；提高 aligner context 大小。
- 保留上游 process name 日誌原始大小寫。

## 相容性取捨

上游新 LLM ctypes binding 的 b10621 struct layout 與 penalty sampler
五參數簽名，不能用於 fork 現有的 b7798 native libraries。
fork downloader 目前以 SHA-256 manifest 驗證 b7798，並提供同一版本給
LLM、Qwen ASR、FunASR、force aligner；後三者的 bindings 仍使用 b7798 ABI。

因此本次同步保留 **b7798 struct layout 與四參數 penalty sampler**，
同時吸收上游「模型載入失敗立即拋出例外」的修正。下載指引也維持 b7798。
這是刻意適配，不能只替換 LLM 綁定或只更新共享二進位。後續升級 native
runtime 時，需一起更新所有 bindings、下載雜湊及模型推論驗證。

上游 `pyproject.toml` 直接要求 Python 3.14，且未限制 Windows-only
packages。fork 維持 Python **3.10–3.12** 相容範圍，依平台選擇依賴；
開發工具在本次 PR 以分角色安裝與 lock 流程整理。上游 Python 3.14
的 `uv.lock` 不適用，不能不經解析就直接重用。

## 回歸驗證

`python -m unittest scripts.tests.test_upstream_refresh`
驗證 token 邊界字元／歷史保留、中文數字、字幕分行與時間、共享 b7798
struct layout、penalty sampler 的四參數呼叫契約，以及錄音門檻區塊保留。
錄音門檻測試在修正前明確失敗，修正後通過。

另外重跑既有 recorder cleanup、backpressure、desktop portability、
tray process launch 測試，確認 fork 既有保障仍有效。完整隔離容器、
API、Web 及 Windows 建置的結果由本次 PR 驗證紀錄統一記載；
Linux 容器測試不代表已完成 Windows 實機快捷鍵／麥克風驗收。
