# Tesla P4／Qwen3-ASR 短句上屏延遲優化計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Checkboxes distinguish completed evidence from planned work.

**Goal:** 保留 Qwen3-ASR 1.7B／目前 Q5_K 與既有功能，優先降低「放開錄音鍵 → 最終可使用文字上屏」的 p50／p95。

**Architecture:** 先建立同一 client clock 上的端到端 baseline，再以 server 階段耗時定位瓶頸。保留 CapsWriter 的引擎介面、傳輸、排程、格式化與對齊功能，依序比較資料搬移、Pascal CUDA 解碼器與錄音期間增量處理；獨立 runtime 是有條件的後續候選。

**Tech Stack:** Python、ONNX Runtime 1.18.0、llama.cpp b7798、CUDA／Vulkan；目前 Docker CUDA 11.8／cuDNN 8；既有桌面用戶端與 Web HTTP client。

**Spec:** [一手來源研究與使用者確認的短句優先情境](../../reports/tesla-p4-qwen3-asr-runtime-20261031.md)。

**Tracking:** [Issue #12](https://github.com/DF-wu/CapsWriter-Offline-Container/issues/12)。

**Branch:** `perf/p4-qwen3-asr-dictation-latency`。

**Base:** `a674e40cb58c5c51991b0e20600a575b7db27944`。

## Global Constraints

- 所有研究、計畫、工具、實作與結果留在這個專用分支；不直接推送或合併 master。
- 本次交付範圍是分支、計畫與 issue 紀錄，不代表已執行以下硬體實驗。
- 不操作現有 Portainer 部署、正式容器、模型 volume 或 GPU 時脈。實機 benchmark 使用另行指定的隔離環境；鎖頻維持關閉。
- 維持同一個 1.7B checkpoint、現有 Q5_K、tokenizer、音訊前處理、語言與 prompt 條件；量化改變另立實驗，不混入 runtime 對比。
- 保留 HTTP／WebSocket、桌面 mic/file、長檔分段、字幕與時間戳語意、取消、deadline、隱私與佇列界限。
- 不以縮成 0.6B、關閉校正／字幕／LLM功能或縮短 context 作為等價加速。
- 不新增未量得的加速倍數、不把 GPU 設定當作有效後端證據、不把磁碟模型大小當作 VRAM。
- 跨主機 clock 不直接相減；用戶端主指標的起終點使用同一個 monotonic clock。
- 外部 runtime 的支援狀態沿用研究的版本邊界；實際建置前重新核對精確版本與架構。

## 開發紀錄規則

- 每個獨立實作或實驗先建立子 issue，主 issue 加入連結與階段摘要。
- Commit／PR 使用 `Refs #12` 與對應子 issue 編號；主 issue 不因單一階段 PR 自動關閉。
- 每次紀錄包含假設、唯一主要變因、完整 commit、環境、重現命令、樣本 ID、結果與保留／淘汰原因。
- 追加進度留言保存過程，主 issue 的 checklist 只摘要狀態。負結果、相容性失敗與品質回歸也要保留。
- Issue 不貼 API key、私密錄音或原始逐字稿；記錄可分享的去識別樣本 ID 與 artifact。

## 檔案與責任邊界

- `docs/reports/tesla-p4-qwen3-asr-runtime-20261031.md`：研究與來源，不當成實測結果。
- 本檔：實作順序、決策門檻與工作追蹤。
- `scripts/bench_dictation_latency.py`（階段一新增）：分析去識別 trace、產出樣本數與 p50／p95；不自行碰觸部署。
- `scripts/tests/test_bench_dictation_latency.py`（階段一新增）：驗證缺失事件、無效樣本與百分位計算，不測文件字串。
- `core/client/audio/recorder.py`、`core/client/output/result_processor.py`、`core/client/output/text_output.py`：桌面 release、接收與上屏量測接點。
- `core/server/worker/task_handler.py`、`core/server/worker/pipeline.py`：排隊與推論階段，不改 transport 契約。
- `core/server/engines/qwen_asr_gguf/inference/{encoder.py,asr.py}`：編碼、prefill 與生成測量／優化接點。
- `docker/server/{Dockerfile,download_models.py,probe_backend.py}`、`core/server/engines/llama/llama.py`：僅在階段二需要 CUDA 封裝時修改；ABI 與 bootstrap manifest 必須一起驗證。

## Task 1：建立短句基準與可信量測

**Consumes:** 同一 client 的錄音 release／最終 ASR 接收／最終文字 commit 事件，以及獨立的 server 階段耗時。

**Produces:** 去識別 JSONL trace 與 JSON summary。Client 記錄至少包含 `run_id`、`sample_id`、`event`、`monotonic_ns`；server 記錄自己的 durations，不跨主機相減。

- [ ] 建立 baseline 子 issue，記錄使用中的映像、模型 hash、量化與樣本集；實際 ORT sessions／decoder 裝置與 CPU fallback 需獨立核對。
- [ ] 先寫分析工具的失敗測試：不同 `run_id` 不可混算；缺 release／commit 的樣本不可當作零延遲；錯誤或取消請求另列，不混成成功樣本。
- [ ] 定義 summary：成功／失敗樣本數、release→final_received、release→text_committed、final_received→text_committed，全部列 p50／p95；p95 採 nearest-rank 並標示樣本數。
- [ ] 實作最少量的事件接點與分析器。只記時間、ID 與必要配置，不記 prompt／逐字稿；預設關閉 trace。
- [ ] 用合成 trace 驗證分析器與同 client clock 邊界，再用隔離環境建立 2／5／10 秒固定短句資料。每組至少 30 次成功樣本，另列冷啟動與閒置後第一句。
- [ ] 對照桌面 mic 與 Web HTTP：Web 目前停錄音不自動提交，手動操作間隔另列；確認用戶端校正／LLM 與真正上屏是否位於主路徑。
- [ ] 記錄 mel／frontend／backend／prefill／decode／alignment／queue 與上屏耗時；現有 RTF log 不能代替完整字幕端到端時間。
- [ ] 將命令、樣本／環境 ID、baseline summary、品質結果與完整 commit 貼回子 issue，主 issue 留摘要。此階段不承諾任何加速數字。

## Task 2：一次一個變因，保留目前模型與服務能力

**Consumes:** Task 1 的固定樣本、可重現 baseline 與 trace schema。

**Produces:** 各候選相對同組 baseline 的延遲、品質及記憶體結果；只保留驗證有效的改動。

- [ ] 依 baseline 最大耗時建立優化子 issue，先選一個主要變因；不把 I/O Binding、batch、後端、量化一次混改。
- [ ] 若 frontend 的搬移／launch 開銷值得優化，先寫同輸入下有效長度、mask、embedding 與結果語意回歸，再比較 I/O Binding 或固定 frontend CUDA Graph。Graph 只對符合固定形狀／位址與 CUDA 節點條件的路徑啟用。
- [ ] 若 decoder 是瓶頸，比較同一 b7798 ABI、同一 GGUF 的 Pascal SM61 CUDA 與 Vulkan。建置產物、執行映像與 CUDA 工具鏈需匹配；更新下載來源、checksum、probe 與 marker，不能隨意替換 `.so`。
- [ ] 保留溫度／重試與 prompt 設定；任何數值或解碼差異必須以固定音訊驗品質，不以文字差異一律視為無害。
- [ ] 若 HTTP 純文字路徑有實際對齊成本，獨立設計需求旗標，只在不要求時間戳的請求跳過對齊；字幕請求仍走原流程。桌面 mic 已跳過對齊，不計為桌面收益。
- [ ] 候選的主指標須有可重現 p95 改善、p50 不退化，且通過同組品質、功能與 VRAM 驗收；無改善或代價過大就淘汰並保留紀錄。
- [ ] 每個候選附上測試、baseline 與 candidate 的完整 commit／命令／summary。通過後才將候選納入專用分支，不發布映像或部署正式服務。

## Task 3：評估錄音期間前移運算與尾段收斂

**Consumes:** Task 1 的計時與品質基準，以及 Task 2 的有效改動；保留現有引擎／服務的契約。

**Produces:** 增量方案的獨立決策與原型結果，不以暫時性 token 代替最終文字。

- [ ] 先開增量處理子 issue，描述錄音期間哪些運算可重用、放開時還需完成哪些工作，以及狀態／記憶體上限。
- [ ] 比較目前短句在放開後主要推論的方式，與增量 encoder／decoder cache 的設計；不得只縮短分段就宣稱完成真串流。
- [ ] 先建立 final 收斂、回滾、斷詞／混語／專有名詞、取消後狀態清理與長錄音有界記憶體的測試，再做最小原型。
- [ ] 原型比較相同 2／5／10 秒短句的 release→最終上屏 p50／p95，另列錄音期間額外運算與峰值 VRAM；不得犧牲最終穩定性換取提前預覽。
- [ ] 若現有架構仍不能達到驗收條件，另開 runtime 子 issue，逐個評測 `antirez/qwen-asr`、`transcribe.cpp` 或 sherpa-onnx。先通過 P4 建置／模型尺寸／能力檢查，再進效能比較。
- [ ] 獨立 runtime 必須常駐模型並接回既有 ASR adapter、prompt／語言與需要的 aligner；不能拿載入成本排除條件不同的 CLI benchmark 當作勝出證據。

## 最終驗證與交付

- [ ] 各階段執行與改動相稱的 focused tests，原有 deadline／取消／隱私與 transport 回歸保持有效。
- [ ] 程式整合完成後跑 root verification、strict API contract；有 Web 修改才加跑其測試／build。文件交付階段只需文件與 diff 檢查。
- [ ] 以中文／混語、數字、專有名詞、噪音／靜音及長段邊界驗品質；保留字幕時間戳語意，不把估算時間戳標成模型對齊。
- [ ] 提交總結：有效與淘汰候選、精確環境／SHA、主指標改善、功能／品質回歸與尚未驗證限制。
- [ ] 將分支、相關 PR、子 issue 與驗證 artifact 回連 Issue #12；整體目標驗收完成前不關閉主 issue。

## 目前進度

- [x] 完成來源研究與 runtime 相容性比較。
- [x] 確認「放開後最終上屏」為優先情境。
- [x] 建立專用分支與 Issue #12。
- [ ] Task 1 baseline／工具實作與實機量測。
- [ ] Task 2 單變因效能候選。
- [ ] Task 3 增量處理與必要的替代 runtime 評測。

此計畫的 checkbox 只代表已取得的證據；未完成項目不視為已實作或已跑過。
