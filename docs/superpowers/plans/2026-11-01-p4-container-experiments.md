# P4 隔離容器實驗計畫

**Goal:** 依 Issue #14 研究，在本機 P4 執行數組可重現實驗，找出下一步值得整合的配置。

**Authorization:** 使用者已授權設計並在本機 test container 執行，完成後清理垃圾。追蹤 Issue #15；沿用 `research/tesla-p4-asr-model-quantization`。

**Architecture:** 正式映像以 image ID 固定，另建無服務 port 的容器；正式模型 volume 只讀。原始碼、替代 library、模型與音訊在本次專用暫存目錄。GPU 實驗串行，source 編譯與下載可並行。

## 比較

- [x] 驗證目前模型 hash、ONNX 真正 dtype 與 GPU 身分。
- [x] 固定公開自然音訊與來源／reference；短樣本用於 latency，較長中文／混語補充品質。不使用私人錄音。
- [x] A：現有 FP16 split encoder + Q5_K + Vulkan（發現缺 EGL 的 CPU fallback，另列第六組 baseline）。
- [x] B：同模型／b7798 ABI，decoder 改為 CUDA SM61。
- [x] C：將現有 FP16 encoder graph／weights 提升 FP32，保留已經四捨五入的相同權重，固定 decoder。此項稱「FP32 promoted」，不冒稱原始 checkpoint 原生重新匯出，也不恢復精度。
- [x] D：保留現有 encoder，僅換上游現成 Q4_K decoder；查清 archive 與 tensor metadata。它是發佈方模型對照，不冒稱已驗得完全相同浮點源頭。
- [x] 另測完整上游 W4 encoder／Q4 decoder 套件，以 ORT profiling 確認 MatMulNBits 在 CUDA。
- [x] 通過 pilot 後，以多 block 交錯執行候選，每組至少30次短樣本；失敗、空白、重試與退出獨立列出。

## 固定與量測

- 先使用 greedy temperature=0；同樣的4個 CPU threads、n_ctx=2048、n_batch=4096、n_ubatch=512、flash attention設定、prompt及language。必要時追加原服務 temperature=0.4，以固定種子作有限回歸。
- 執行服務實際使用的 `QwenASREngine.decode_stream()`，不使用會補齊80秒的獨立 `asr()` 捷徑。
- 每次用 `perf_counter` 記 engine wall、encoder frontend/backend、prefill、generation、tokens、重試與輸出。這不是 client release→上屏或 HTTP端到端。
- 公開 reference 的 CER/WER、相對 baseline 輸出差異分開；沒有 reference 的片段只作延遲與一致性，不計準確率。
- 記錄 GPU 溫度、時脈、thermal slowdown及記憶體；不鎖頻、不停其他服務。共用GPU結果保留實際環境限制。
- 新 FP32 ONNX 必須通過 checker、有效shape與embedding差異檢查；CUDA須確認實際offload，不能以能載入library當GPU成功。

## 保存與清理

- [x] 保存實驗腳本、去識別公開樣本manifest、逐次JSONL、摘要、模型／library hash、環境與失敗記錄。
- [x] 結果寫 `docs/reports/p4-container-experiments-20261101.md` 並回連 Issue #15/#14。
- [x] 移除本次test／compiler容器、新拉取的devel image與專用暫存模型／build／音訊；保留既有映像與正式volume。
- [x] 檢查正式服務仍在原容器ID、沒有本次標籤殘留，資料分析通過；提交推送紀錄由 git history／Issue #15 回報連結核對。

本計畫是使用者已授權的實驗執行範圍，不代表任何尚未量得的效能結果。

實際執行：6 配置×30 latency＋6×5 quality＋9 temperature=0.4 smoke。latency 是同一來源裁切的3個unique clips；quality為3段英語、1段中文數字、1段英法義西多語，無台灣口音／中英混語。GPU latency最高91°C，Q4組出現HW thermal slowdown，不能把小差異視為純模型效果。正式服務未改；清理證據見報告。
