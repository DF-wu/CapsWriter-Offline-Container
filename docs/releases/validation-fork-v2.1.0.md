# fork-v2.1.0 候選驗證紀錄（2026-10-30）

本紀錄對應本次 feature branch 的候選內容；最終提交 SHA 由草稿 PR 的 head
與 CI 記錄識別。上游基準仍是 `84912d5`，本次工作起點為
`b8baa933bef3cbca5176792848ddfc817cdd1270`。尚未建立正式 tag 或發布映像。

## 自動驗證

- `.venv-dev/api/bin/python scripts/verify_all.py --skip-web`：通過。
  CLI 63、HTTP/API 191、Docker bootstrap 64、scripts/worker 434 項；最後一組
  有 13 項因顯示環境或未安裝完整 server／字幕依賴而略過。
- 前一輪完整 `scripts/verify_all.py --no-web-install` 已跑過同一份未變動的
  Web 程式：Vitest 8 個檔案、118 項通過，TypeScript 檢查與 Vite build 通過。
- 嚴格 API contract runner 使用 hash-locked API 環境，檢查依賴版本並拒絕
  skipped tests；191 項通過、沒有略過。
- `.venv-dev/server/bin/python -m unittest scripts.tests.test_startup_shutdown_races
  scripts.tests.test_lifecycle_review -q`：21 項通過，沒有略過；使用測試 worker
  與隔離 listener，不連線正式部署。
- 新增真正的 `multiprocessing.get_context("spawn")` 回歸：fresh interpreter
  載入原始預設，再以 fake model worker 執行 production entrypoint，涵蓋
  啟用的 argv 與停用的 legacy 字串命令；另檢查父程序確實傳送設定 snapshot。
- GPU 設定、CPU-only／完整 CPU fallback、low-VRAM CUDA、單調時間閒置、
  abandoned warmup、失敗重試／還原、正常退出及 API 取消／deadline 均有回歸。
  硬體命令以測試替身取代；程序逾時測試只執行合成 Python 子程序。
- 模型載入後再檢查 SenseVoice／Qwen／Fun-ASR 的實際 ORT providers，
  涵蓋要求 CUDA／TRT／TENSORRT 卻載入 CPU-only sessions 的情況，並在
  ready 前更新資格。GGUF 實際 offload 層數與 Sherpa 內部 session 無可用
  查詢介面，仍依載入實例設定；這不構成完整硬體 placement 驗證。
- default／GPU／GPU+boost／settings Compose 結構驗證與 `git diff --check` 通過。
  已確認 default 不新增 root／capability、boost 保持關閉；選用 override
  才有 root／capabilities 與 150 秒 stop grace。
- myServices 的 base／boost override 只做 `docker compose config` 靜態驗證；
  本機缺少 `DF_API_KEY` 的插值警告不代表正式 Portainer 環境已驗證。

## 獨立審查

第一輪沒有候選準備阻擋項；提出 CPU fallback 不應鎖頻及 spawn 測試未跨程序
兩點，已補上實作與回歸。第二輪提出 TensorRT selector 漏列，已按引擎接受的
`TRT`／`TENSORRT` 補齊與回歸。第三輪指出 ORT 的 runtime CPU fallback
仍會沿用配置資格，已新增載入後判斷與回歸。第四輪審查實作提交
`312a8866135707855138034846f8e620e9c88675`，沒有新增問題；獨立跑過
64 項 focused tests，全部通過、沒有略過。配對 recipe 審查提交為
`9184841f5adef68fef32b855c62af9aa275b07e8`。後續文件提交與最終 head
記錄於[原始碼草稿 PR #10](https://github.com/DF-wu/CapsWriter-Offline-Container/pull/10)，
recipe 位於[草稿 PR #18](https://github.com/DF-wu/myServices/pull/18)。

## 尚未驗證／未執行

- 沒有執行 NVIDIA 實際鎖頻、量測效能或宣稱特定 GPU／driver 支援。
- 沒有做 Windows 實機權限、完整 Windows spawn 模型推論、容器停止或 root
  模式切回非 root 的實機演練。POSIX 上的 spawn 測試不等於 Windows 實機驗證。
- 沒有操作 Portainer、正式容器、遠端掛載或舊 appdata 模型資料。
- 沒有合併或推送 master、建立正式 tag、發布 Release 或移動 latest。
- CI、最終 merge SHA、映像 digest、Windows artifact 與適用的硬體證據，必須
  在正式發布前填入[發布檢查表](fork-v2.1.0.md)。

原始本機輸出保留於 `/tmp/capswriter-v2.1-verify-runtime-final.log`、
`/tmp/capswriter-v2.1-api-contract-runtime-final.log`、
`/tmp/capswriter-v2.1-lifecycle-runtime-final.log`；前輪 Web 輸出位於
`/tmp/capswriter-v2.1-verify-final.log`。
這些路徑是本次工作紀錄，不是對其他主機的可攜式 artifact 連結。
