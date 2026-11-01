# Qwen3-ASR prefill、音訊長度與顯存調校

適用於目前 Server 的 split ONNX encoder＋GGUF decoder。所有數值都需連同
runtime、模型、輸入長度與實際 native log 判讀；單一 Compose 與 P4 flag
不會自動選出最快的 prefill 參數。設定入口見[單檔部署](compose.md)，環境變數
總表見 [Docker Server](docker-server.md)。本頁沒有變更預設值或部署現行服務。

## Prefill 處理什麼

Server 的路徑為音訊 → mel → ONNX frontend／backend → audio embeddings →
llama.cpp prefill → 逐 token 生成文字。Prefill 一次讀入音訊 embeddings、system／
context、語言提示；生成階段再逐 token 解碼。`N_BATCH`、`N_UBATCH` 是 decoder 的
token／embedding 位置數，不是秒數、音檔數或 HTTP 同時請求數，也不改 ONNX 精度。

## 參數與取捨

| 環境變數 | 未覆寫時的現行值 | 調整後的影響 |
|---|---|---|
| `CAPSWRITER_QWEN_LLAMA_N_BATCH` | 留空；adapter 請求 `4096`，P4 的 `n_ctx=2048` 下有效值 `2048` | 單次 native decode 接受的輸入位置上限。現行 adapter 整段送入；太小可能讓長音訊／長 prompt 觸發 assertion，並非自動多跑幾批 |
| `CAPSWRITER_QWEN_LLAMA_N_UBATCH` | 留空；`512` | 實際運算的 micro-batch 上限。較小通常減少部分暫存需求、增加批次／排程開銷；較大可能增加暫存或改善吞吐，也可能更慢／OOM，須實測 |
| `CAPSWRITER_QWEN_N_CTX` | `2048` | KV context 容量，需容納輸入與後續生成。加大通常增加 KV／工作記憶體；不會自動延長模型的有效能力，也不會解除音訊截斷或固定生成上限 |
| `CAPSWRITER_QWEN_CHUNK_SIZE` | `80` 秒 | **目前 Server adapter 的單次音訊保留上限**；超過就只取前面這段。調小會丟掉尾端，不是無損分段加速 |
| `CAPSWRITER_QWEN_MEMORY_NUM` | `1` | 內部 file-transcribe 分段路徑的歷史段數；目前 Server `decode_stream()` 不使用，不會替跨請求保留記憶或補回截斷部分 |
| `CAPSWRITER_QWEN_LLAMA_FLASH_ATTN` | 留空；adapter 請求 `true` | 改變 attention 實作，可能影響速度／暫存，需 backend／模型支援。依 native log 確認是否真正啟用，不能從設定值保證加速 |
| `CAPSWRITER_QWEN_LLAMA_OFFLOAD_KQV` | 留空；`true` | K/Q/V 相關運算的 GPU offload 控制；關閉可能改變 CPU／GPU 負擔，但不等於把全部 decoder 權重移回 CPU。需要整體低顯存模式時，另評估 `low_vram_gpu` |
| `CAPSWRITER_QWEN_PAD_TO` | `30` 秒 | 此 encoder 僅在 DirectML 路徑做固定形狀 padding。P4 CUDA 不會因設成 `0` 而省下「補到30秒」的成本 |

整數 batch／ubatch／context 覆寫須為正整數；空白代表保留既有值。布林覆寫接受
`true/false`、`1/0`、`yes/no`、`on/off`。env 層檢查值域，**不會驗證某份音訊是否
裝得下，也不會自動選擇安全 batch**。在此固定 b7798 causal decoder 中，有效
`n_batch` 不超過 `n_ctx`，有效 `n_ubatch` 不超過 `n_batch`。因此 `N_CTX=2048`
時把 batch 從 `2048` 設成 `4096`，不代表運算批量加倍。

## 與語音長度的關係

此版 encoder 每一整秒約產生 **13 個 audio embeddings**；零碎尾段依 frontend
取整規則計算。以下只估算音訊部分，**不是允許長度或品質保證**：

| 單次保留音訊 | 音訊輸入位置約數 | 還要加上什麼 |
|---|---:|---|
| 2／5／10 秒 | 26／65／130 | system、context、語言提示及特殊 tokens |
| 30 秒 | 390 | 同上；長 context 可能占用比音訊更多的位置 |
| 60 秒 | 780 | 同上，且 context 尚需保留生成空間 |
| 80 秒 | 1040 | 同上；這是目前預設截斷點，不是已驗證長音訊品質上限 |

調整時分別檢查：

1. `完整 prefill 位置數 <= 有效 n_batch`。例如 `N_BATCH=512` 不能解讀成
   「任意長音訊，每次只算512個」；60秒光音訊就約780個，尚未加入 prompt。
2. `完整 prefill 位置數 + 實際生成位置數 <= 有效 n_ctx`。目前內部生成迴圈每次
   最多512輪；調大 context 不會同步增加這個上限。語速、語言及標點會影響輸出
   token 數，不能把「2048÷13秒」當作可完整轉錄的秒數。
3. Server adapter 先套用 `CHUNK_SIZE` 截斷。即使 batch／context 足夠，超過80秒
   的單次輸入在預設設定下仍不完整。長檔應由已驗證的上層流程切段並合併文字，
   或另外驗證較大的保留上限；不要直接調低 `CHUNK_SIZE` 來量加速。

Server worker 呼叫的是外層 adapter，並非內部 `transcribe()`／`asr()` 分段函式。
後者會使用 chunk／memory，不能把它的歷史視窗行為套到目前 HTTP／WebSocket
recognition call。WebSocket task 的總秒數限制也不是單次 Qwen context 容量。

## P4 可比較的 prefill 組合

以下是**試驗矩陣**，不是新 preset，也沒有宣稱每組都已在 P4 測過：

| 目的 | `N_CTX` | `N_BATCH` 請求值 | `N_UBATCH` | 使用限制與取捨 |
|---|---:|---:|---:|---|
| 保留既有基準 | 2048 | 留空（4096，有效2048） | 留空（512） | 既有 P4 短音訊實測採此組；先確認 GPU 確實 offload |
| 較低 prefill 暫存試驗 | 2048 | 4096 | 256，必要時128 | 保留相同輸入容量，只縮 micro-batch；可能增加延遲，未保證固定 MiB 節省 |
| 較大 micro-batch 試驗 | 2048 | 4096 | 1024 | 不延長可保留音訊；短輸入可能裝不滿，也可能增大工作 buffer。未證明比512快 |
| 額外 context 試驗 | 4096 | 4096 | 512 | 主要給較長 prompt／輸入與生成餘裕；KV成本增加，80秒截斷及生成上限仍存在，未完成長音訊品質驗收 |

例如要試256，只覆寫一項，其他維持原配置：

```dotenv
CAPSWRITER_QWEN_LLAMA_N_UBATCH=256
```

將設定寫入原 Portainer Stack 或 Compose `.env`，檢查 `docker compose config --quiet`
後，在維護時段以原 Stack／同一 Compose 重新部署以套用 env；普通 `docker restart`
不會改既有注入值。保留既有 env／secrets、GPU 區塊、ports、network 與 volumes。
回復時移除該覆寫或設空，再重新部署。P4 flag 及其嚴格 CUDA 啟動驗證不變。

### 速度與辨識效果怎麼判讀

micro-batch 改的是運算切法，**不是提高量化精度或辨識品質的開關**。不能保證文字
逐字相同：浮點運算路徑、取樣與熔斷重試都可能改變輸出。目前 Server adapter 預設
temperature=0.4，既有 HTTP worker 也不會把 API 的 temperature 值轉交此 adapter；
不能把填了 API temperature=0 當作完成 greedy 控制。縮短／截斷音訊則直接改變
辨識內容，速度即使變快也不能視為同品質加速。

比較時固定相同權重、runtime、音訊 bytes、context、language、取樣與其餘參數，
用代表性的2／5／10秒短句與30／60／接近80秒樣本，另驗證超過上限的處理。記錄
整體延遲 p50／p95、prefill／encoder／generation 分項（可取得時）、native 有效
batch／ubatch／context、輸出／重試／失敗數、CER／WER、GPU常駐與峰值。每組暖機、
重複測量；共用GPU與散熱條件要一致。engine wall、HTTP與放開按鍵到上屏是不同邊界。

### 既有證據與沒有測過的部分

- [獨占 P4 矩陣](reports/p4-exclusive-container-experiments-20261101.md)的 CUDA Q5
  基準為 context2048、requested batch4096／effective2048、ubatch512；2／5／10秒
  engine p50 約0.133／0.303／0.583秒。該矩陣比較模型／runtime，**不是 prefill
  batch sweep**；不能用它宣稱256或1024更快，也未驗收60／80秒品質。
- 同表 decoder-only Q6_K 比 Q5 多約150–160MiB的呼叫前後顯存讀值，10秒延遲增加
  約12.5%，5份品質樣本未見額外收益。換量化格式與調 prefill 是不同變因；這些
  顯存讀值不是瞬間峰值，也不是各 ubatch 的顯存比較。
- Server 路徑依據：[adapter](../core/server/engines/qwen_asr_gguf/asr_engine.py)、
  [prefill／生成與內部分段](../core/server/engines/qwen_asr_gguf/inference/asr.py)、
  [encoder與輸出長度](../core/server/engines/qwen_asr_gguf/inference/encoder.py)、
  [預設 schema](../core/server/engines/qwen_asr_gguf/inference/schema.py)、
  [env覆寫](../fork_server/env_config.py)。
- native 邊界依據：固定 [b7798 llama-context.cpp](https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/src/llama-context.cpp)，
  context constructor限制有效batch／ubatch，`decode()`要求整批位置數不超過batch，
  再由memory context按ubatch處理。此說明不假定其他版本有相同行為。

## English operator summary

`N_BATCH` limits positions accepted by one native decoder call; the current
server adapter submits the full prompt at once. A too-small value can abort the
worker instead of processing a long recording in several logical batches.
`N_UBATCH` controls physical micro-batches: try 256 versus the existing 512 while
holding input capacity fixed, and measure memory, latency and transcripts.
These alternatives have not been benchmarked as a P4 parameter sweep.

`N_CTX` budgets input plus generated positions. With the pinned causal decoder,
batch is capped by context and ubatch by batch. The existing baseline requests
4096/512 with context2048, producing effective batch2048. Audio contributes about
13 positions per second, plus prompt overhead; generation also needs context.

The server's `CHUNK_SIZE=80` **truncates** a recognition call rather than splitting
it. `MEMORY_NUM` applies to the separate internal file-transcription path, not
server calls. Increasing context or batch does not remove truncation or the fixed
generation limit. Padding is DirectML-specific here and does not pad CUDA input
to30 seconds. Persist env changes in the original Stack and recreate using its
existing settings; a restart alone does not update env. A smaller ubatch is not a
quality upgrade, and temperature/API forwarding limits must be considered when
comparing text.
