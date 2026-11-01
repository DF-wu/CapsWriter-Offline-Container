# Tesla P4：Qwen3-ASR 本機隔離容器實驗

日期：2026-11-01（Asia/Taipei）。追蹤 [研究 #14](https://github.com/DF-wu/CapsWriter-Offline-Container/issues/14)、[實驗 #15](https://github.com/DF-wu/CapsWriter-Offline-Container/issues/15)。分支：`research/tesla-p4-asr-model-quantization`。

**後續更新：**[13組獨占GPU實測](p4-exclusive-container-experiments-20261101.md)已完成，包含Q6／Q8／IQ4與完整transcribe.cpp引擎；品質較原Q5差者排除推薦。下文為第一輪共用、熱降頻環境紀錄。

## 結論與建議

**這台機器首先值得修的是 decoder 的執行路徑。** 正式映像缺少 `libEGL.so.1`，現有 Vulkan decoder 找不到 P4，實際只枚舉 CPU。在 test container 僅安裝 `libegl1` 就能恢復 Vulkan GPU；同模型改用 b7798 的 CUDA SM61 build，這輪測得更低的 engine 延遲。這次沒有修改正式映像、容器、模型或 GPU stack。

優先整合候選是 **原 FP16 ONNX encoder＋原 Q5_K decoder＋CUDA SM61**。它不是每格中位數都最小，但這輪三種長度的尾延遲比 Q4 穩定。CUDA 相對本輪 CPU decoder fallback，2／5／10 秒片段的 engine p50 分別改善約 **6.23／5.30／4.41 倍**；相對已修好的 Vulkan 約 **1.50／1.76／1.54 倍**。這些是共用且熱降頻中的本機實測比值，不能外推為 P4 峰值或 client 上屏速度。

模型端也實測了兩條路：FP32 promoted 沒有全面加速；現成 W4 encoder＋Q4 decoder 明顯節省記憶體，但 encoder 本身較慢。只換 Q4 decoder 有部分中位數優勢，仍需在改善散熱後重測尾延遲與更廣的語料，不能據此宣布最高性能或品質等價。

## 量測設計與邊界

使用專用 test／compiler containers，正式模型 volume 在 test container 掛載為唯讀 `/models-original`。專用 scratch 掛載 `/bench`；6 CPU、16 GiB RAM、512 pids、無對外 port。正式服務繼續運作；GPU 原有約 1,253 MiB 使用量。沒有停服務、改風扇、鎖時脈或改功耗上限。

- Tesla P4，SM 6.1，7,680 MiB，driver 580.196.07；CUDA 11.8、cuDNN 8、ORT 1.18.0、Python 3.10.12。
- 程式固定為 `d3b334e95c4d958f6d7812cb94c14a33fff427d3` 的 archive；runtime 固定既有 image ID。完整版本與模型／library SHA256 見 [environment.json](experiments/p4-20261101/results/environment.json)。
- 呼叫服務使用的 `QwenASREngine.decode_stream()`；沒有使用會補齊 80 秒的獨立 `asr()`。
- `temperature=0`、NumPy seed `1729+repeat`、`n_threads=n_threads_batch=4`、`n_ctx=2048`、`n_batch=4096`、`n_ubatch=512`、flash attention 開啟；無 context／aligner。ORT 選項沿用現有 encoder，沒有額外設定 ORT intra-op thread 數；4 threads 是 decoder 與 OMP／OpenBLAS 設定。
- 5 個 GPU 配置串行，3 blocks 的 repeats 為 4／3／3，各 block 輪轉配置順序。每個 process 先做 3 次暖機，排除於統計。CPU fallback 先單獨測 10 repeats、2 次暖機，沒有交錯，環境差異須保留。
- 每個 repeat 依序處理 2／5／10 秒片段；每配置 30 次，但僅 **3 個 unique clips，且是同一段 Mandarin noise2 的巢狀裁切**。不能視為 30 個獨立語料。裁切段無對齊 reference，不計 CER。
- 每配置另跑 5 個完整 reference 品質樣本一次。總計 180 次 measured latency、30 次 measured quality；另 9 次 temperature=0.4 固定種子 smoke。pilot、載入與暖機全部分開。
- wall 從 `decode_stream()` 開始到返回，含 encoder／decoder／該路徑輸出處理；不含模型載入、音檔讀取、client、網路與實際 release→上屏。stage 是內部計時，不能把各 stage 的獨立中位數直接相加當 wall。
- p50 是中位數；p95 為 nearest-rank。每 clip 只有 10 次，所以 p95 就是該 clip 的最大值；沒有提供置信區間或統計顯著性主張。

## 六組配置

| ID | Encoder | Decoder／backend | 用途 |
|---|---|---|---|
| shipped-cpu-fallback | 原 FP16 ONNX，CUDA EP | 原 Q5_K，CPU | 重現缺 EGL 的實際 fallback |
| vulkan-f16-q5 | 原 FP16 ONNX，CUDA EP | 原 Q5_K，Vulkan P4 | 測試容器補 libegl1 |
| cuda-f16-q5 | 原 FP16 ONNX，CUDA EP | 原 Q5_K，CUDA SM61 | 固定模型比較 backend |
| cuda-f32-q5 | FP32 promoted，CUDA EP | 原 Q5_K，CUDA SM61 | 固定 decoder，比 encoder graph dtype |
| cuda-f16-q4 | 原 FP16 ONNX，CUDA EP | 上游 Q4_K，CUDA SM61 | 固定 encoder，比發佈方 decoder artifact |
| cuda-int4-q4 | 上游 W4 ONNX，CUDA EP | 上游 Q4_K，CUDA SM61 | 完整現成 Q4 套件 |

Q4 來自 [上游 release archive](https://github.com/HaujetZhao/CapsWriter-Offline/releases/download/models/Qwen3-ASR-1.7B-q4_k.zip)。ZIP SHA256、大小及下載記錄見 [q4-archive.json](experiments/p4-20261101/results/q4-archive.json)。Q4／Q5 均為 `qwen3vl`，28 layers、embedding 2048、311 tensors、tensor 名稱及形狀相符；這不能證明兩者來自完全相同的原始浮點權重，故不是嚴格的同 checkpoint 量化精度 A/B。

W4 frontend／backend 分別含 1／146 個 `MatMulNBits`，bits=4、block_size=128、activation 為 FLOAT；它是 weight-only INT4，不是已證明的 INT8 DP4A encoder。FP32 promoted 是把既有 half initializer、圖中 type／Cast／tensor constant 提升為 FLOAT；**不會恢復 FP16 前已損失的精度，也不是原始 checkpoint 原生重新匯出**。

## 延遲結果

每格為 engine 秒數 **p50／p95**，各 10 次；完整 [逐次資料與 summary](experiments/p4-20261101/results/summary.json) 保留失敗、輸出、stage 與 GPU snapshots。

| 配置 | 2 秒片段 | 5 秒片段 | 10 秒片段 |
|---|---:|---:|---:|
| CPU decoder fallback | 2.599／3.143 | 5.375／5.805 | 8.603／9.899 |
| Vulkan＋FP16＋Q5 | 0.626／0.646 | 1.788／1.793 | 3.000／3.023 |
| CUDA＋FP16＋Q5 | 0.417／0.427 | 1.015／1.028 | 1.951／1.963 |
| CUDA＋FP32 promoted＋Q5 | 0.410／0.420 | 1.051／1.059 | 1.960／1.978 |
| CUDA＋FP16＋Q4 | 0.409／0.540 | 1.031／1.278 | 1.890／2.732 |
| CUDA＋W4＋Q4 | 0.499／0.508 | 1.128／1.147 | 2.018／2.033 |

三個 block 的各 clip median 另存 [audit.json](experiments/p4-20261101/results/audit.json)，不是只保留最快一次。所有 210 次主要 measured calls 都有非空結果，無 exception、最終 abort 或額外 decode retry。這裡的成功只表示呼叫與輸出有效，**不表示轉錄正確**。

在 10 秒片段，CUDA FP16／Q5 的 encoder／prefill／generation 中位數約為 0.289／0.011／1.637 秒；CPU fallback 的 prefill 約 6.533 秒。GPU 路徑解決的是這個大段瓶頸；不能只看 encoder dtype。

### 散熱與 GPU 記憶體

下表是主要 latency 每次前後的 60 個 snapshots；memory 是含其他服務的 GPU **總使用量**，不是單 process 用量或完整瞬間峰值。200ms telemetry 另存 [gpu-telemetry.csv](experiments/p4-20261101/results/gpu-telemetry.csv)，只涵蓋交錯 GPU matrix 與品質階段，未涵蓋最初 CPU baseline／後續 profiling／smoke。

| 配置 | 溫度 °C | graphics MHz | SW thermal Active／60 | HW thermal Active／60 | total MiB 範圍 |
|---|---:|---:|---:|---:|---:|
| CPU fallback | 86–88 | 885–999 | 0 | 0 | 2503 |
| Vulkan FP16 Q5 | 88–89 | 455–1113 | 37 | 0 | 4234–4244 |
| CUDA FP16 Q5 | 89–90 | 455 | 60 | 0 | 4225–4227 |
| CUDA FP32 Q5 | 88–90 | 455–885 | 57 | 0 | 5283–5291 |
| CUDA FP16 Q4 | 88–91 | 455 | 60 | 7 | 4085–4093 |
| CUDA W4 Q4 | 88–90 | 455 | 60 | 0 | 3351–3411 |

Q4 decoder 在 455MHz 下仍有部分 median 優勢，但該組也出現 HW thermal slowdown；目前較差的 p95 不能全歸因於 Q4。CUDA Q5 三個 clip 的 generation 中位數約 0.316／0.826／1.637 秒；Q4 約 0.305／0.841／1.572 秒，且 5 秒輸出 token 數不同（Q5 11、Q4 12），所以不是純 kernel speedup。

W4 的 encoder median 為 0.174／0.265／0.427 秒，相較原 encoder 0.078／0.165／0.289 秒較慢；全 engine 仍要一起看。FP32 total memory 比原 FP16 Q5 約多 1 GiB。W4/Q4 約少 0.8 GiB，但本輪不構成性能優先選擇。

## Vulkan 根因及 CUDA 相容性

[vulkan-diagnosis.json](experiments/p4-20261101/results/vulkan-diagnosis.json) 與 [strace](experiments/p4-20261101/logs/vulkan-loader.strace) 記錄了以下鏈條：

1. 正式映像的 test clone：`find_library('EGL') == None`；NVIDIA `vk_icdGetInstanceProcAddr(NULL, 'vkCreateInstance')` 為 NULL；GGML devices 僅 `CPU`。
2. 先單獨複製 host NVIDIA graphics DSOs 到 scratch，未解決。正常 container driver injection 本身已具備需要的 driver 路徑；測試比較不使用那批副本，也未覆寫 `LD_LIBRARY_PATH`。
3. 僅在 test container 安裝 `libegl1`（及依賴 `libglvnd0`）後，EGL 可解析，pointer 非空，GGML 枚舉 `Vulkan0`、`CPU`；P4 的 `fp16:0`、`int dot:1`、`matrix cores:none`。
4. 正式容器唯讀 probe 也確認 EGL 缺失、pointer false；既有 log 有 `ggml_vulkan: No devices found`。沒有修改正式容器。單看 `offloaded 29/29 layers` 不足以判定用了 GPU。

CUDA build 來源固定 [llama.cpp c301172](https://github.com/ggml-org/llama.cpp/tree/c301172f660a1fe0b42023da990bf7385d69adb4)（b7798），`GGML_CUDA=ON`、`CMAKE_CUDA_ARCHITECTURES=61`，`cuobjdump --list-elf` 證實有 SM61 cubins，runner 枚舉 `CUDA0` 並核對實際 offload log。CUDA graphs 因架構停用，沒有套用其他新卡的 graphs／tensor-core 預期。

無 GPU 的 compiler container 最後 link 曾因找不到 stub `libcuda.so.1` 失敗；只在 compiler 建立 stub symlink，補上該步後 build 成功。stub 的 `LD_LIBRARY_PATH` 只用於編譯，runtime 使用真實注入的 driver。保留失敗與重試 log，沒有把首次失敗隱藏為全部一次成功。

## Encoder 驗證

獨立於 latency 的 [check_encoders.py](experiments/p4-20261101/check_encoders.py) 開 ORT profiling，輸出 [encoder-diagnostics.json](experiments/p4-20261101/results/encoder-diagnostics.json)。profiling durations 含額外開銷，只用來確認 provider／operator，不當作正式 benchmark。

- 原 backend 因 Cast 排在 consumer 後方而不通過 ONNX checker；ORT 仍能接受。FP32 promoted 作 topological sort 後 frontend／backend 都通過 checker。
- 原 encoder 的 Conv／MatMul／FusedMatMul 在 CUDA EP；少量 shape 組裝的 Concat 在 CPU，不是整個 encoder CPU fallback。
- W4 的 frontend／backend `MatMulNBits` **都在 CUDA EP**。backend profile 共 584 次（146×4，含 encoder 初始化暖機），沒有證據可把這輪變慢解釋為 MatMulNBits 全落 CPU。
- ORT profile 只揭示 provider、operator 與輸入輸出 dtype；無法單靠它斷言內部 CUDA kernel 採原生 half arithmetic 或 INT8 DP4A。也不能套用「FP16 必然慢 64 倍」。
- 三個 clip 的 embedding shape 都是 `[26/65/130, 2048]` 且 finite。FP32 vs 原 encoder 整體 cosine 為 0.9999968／0.9999956／0.9999641，relative L2 為 0.00255／0.00295／0.00848；數值接近但非完全相同。
- W4 vs 原 encoder 整體 cosine 約 0.97331／0.97023／0.96947；此比較還含發佈 artifact 來源不完全可證同源的限制，不能把差異全歸為單一量化操作。
- 保存的 promotion 腳本重新產生 frontend／backend 並比對 SHA256；實際結果見 environment 中 `fp32_promotion_reproduction`。權重本體不提交。

## 品質與 sampling 回歸

公開樣本與 reference 出自 [LibriSpeech dummy](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/tree/5be91486e11a2d616f4ec5db8d3fd248585ac07a) 前三筆，以及 [公開 Qwen3-ASR 範例 test_wavs](https://huggingface.co/csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/tree/68818b2313fe77bd06f6a7c5068ff3ef59d02b8a/test_wavs) 的 `noise2`、`codeswitch`。hash、revision、重採樣與裁切方法見 [samples.json](experiments/p4-20261101/samples.json)。`codeswitch` 是英／法／義／西語，**不是中英混語**；這組沒有台灣口音與實際使用者語料。

CER 正規化僅 NFKC、lowercase、移除標點／空白，英文 `Mr.`→`mister`；不轉繁簡或數字讀法。英文 WER 保留 apostrophe 詞形。完整逐句轉錄、edit count、reference 長度及相對 CPU baseline 的原文／正規化差異都在 audit。

| 品質樣本 | 全部配置結果 |
|---|---|
| 3 段 LibriSpeech | 每段正規化 CER／WER 均 0；僅三筆，不能推廣為品質保證 |
| Mandarin noise2，22.833s | 每組 CER 2/53≈3.77%，正規化後相同；含「您→你」與多一個「五」 |
| 多語 codeswitch，6.230s | Q5 各 backend／FP32 CER 10%；兩組 Q4 CER 0%；僅此一筆，不能認定 Q4 普遍較準 |

所有配置在 2 秒裁切片段都轉成「王浩。」；5 秒片段的輸出也隨模型／backend 改變。這說明截斷片段與輸出 token 數會影響延遲，不能拿它們做品質證明。

另以原 Q5 的 CPU fallback／Vulkan／CUDA 各跑 3 個裁切 clip，temperature=0.4、同 seed=1729；9 次 smoke 全部非空、無失敗或 retry，10 秒文字三組相同，但 5 秒仍有「何事／什么」差異。只是一個 seed、每 clip 一次；期間溫度／時脈與 matrix 不同且 Vulkan 有延遲離群，**不與 greedy matrix 合併，不以此給 p50／p95 或可重現加速倍數**。

## 保存、重現與清理

完整 [重現說明](experiments/p4-20261101/README.md) 保留實際 runner、matrix、分析、promotion、provider 診斷與環境驗證腳本；results 保留原始 JSONL、schedule、telemetry、hash、統計與 audit，logs 保留編譯及推論證據。未提交模型、原始音訊、第三方 source tree、編譯產物或帶簽名的音檔 URL。

已移除本次兩個 containers、新拉取的 devel image、約 6.10 GB scratch（邏輯 bytes，含模型／音訊／build）與 state file，未作全域 prune。[cleanup.json](experiments/p4-20261101/results/cleanup.json) 核對正式容器仍是原 ID、原 StartedAt、running／healthy，既有 runtime image／模型 volume 保留；原模型 hash 未變，實驗標籤及 200ms telemetry process 無殘留。清理後 GPU 使用量回到原有 1,253 MiB。

後續整合順序：先補 runtime EGL dependency 與 GPU discovery 檢查，再加入 CUDA SM61 library 選擇；改善 P4 散熱後重跑交錯矩陣，以較多真實台灣國語／中英混語／數字語料測 client release→final latency，才決定 Q4 是否值得成為預設。這些正式功能變更不包含在此次容器實驗內。
