# Tiro v2 × P4 quick experiment

追蹤 [#17](https://github.com/DF-wu/CapsWriter-Offline-Container/issues/17)，branch `research/tiro-v2-p4-smoke`。總結見[報告](../../tiro-v2-p4-smoke-20261101.md)，公開頁原始檔為 `site/index.html`，由 `make_page.py`生成。

這是一次性快速研究，不是產品部署或通用轉換器。來源固定 `rohansheth/tiro-qwen3-asr-1.7b-v2@5dd5f33e0dbaa46254fc552a658d90152f1eaf75`；CPU量化器固定llama.cpp b7798 `c301172f660a1fe0b42023da990bf7385d69adb4`。來源BF16整檔SHA、模型／library hashes及量化配方保存於results。

## 實際準備

以原runtime image ID建立獨立test container（root僅用於test apt／pip），6 CPUs／18GiB、512 pids、無發布port；原模型volume掛 `/models-original:ro`，scratch掛 `/bench`。只在test安裝libegl1、onnx1.16.2、cmake／ninja／g++。沒有host GPU stack、正式應用或新Docker image變更。

完整原source `238af5b` git archive放 `/bench/src`，原正式container的llama/bin複製為 `/bench/vulkan-bin`。官方模型的所有非.gitattributes檔案由固定revision下載、LFS整檔SHA校驗；沒有保存帶簽名redirect URLs。b7798 source tar SHA `4420071ba535356cb62f8cd65edb924dee321942d45b7e5e402df1f2906e1031`，CPU build：

```sh
cmake -S /bench/llama -B /bench/cpu-build -G Ninja -DGGML_CUDA=OFF -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /bench/cpu-build --target llama-quantize -j6
```

以上是在disposable container內執行的命令；host shell入口一律加 `rtk proxy docker exec <test-container> ...`。無CUDA compiler或kernel修改。

## 轉換與量化

`convert_weights.py`在test內執行，只需numpy／onnx及repo自帶gguf，不需torch／transformers。讀BF16 safetensors，以named bias consumer追查每個ONNX MatMul的learned weight並轉置；所有707來源tensors覆蓋。Source projector映射到既有encoder的proj1／proj2，舊圖sinusoidal position table保持deterministic constant。兩ONNX經topological sort後checker通過；CPU load與finite outputs另驗。

GGUF採既有qwen3vl metadata／vocabulary，但所有語言權重來自Tiro。151,705個已宣告token IDs與BPE merges逐項一致；tied output複製source embedding。輸出F16（norm為F32），再分別：

```sh
/bench/cpu-build/bin/llama-quantize /bench/models/tiro-decoder-f16.gguf /bench/models/tiro-q5/qwen3_asr_llm.gguf Q5_K_M 4
/bench/cpu-build/bin/llama-quantize /bench/models/tiro-decoder-f16.gguf /bench/models/tiro-q4/qwen3_asr_llm.gguf Q4_K_M 4
```

Q5約95.88s、Q4約109.78s；Q5逐tensor type recipe和原模型相同。沒有低位元升位元。Source BF16並非P4專用成品，這只是既有runtime兼容格式，沒有主張P4最優量化或新增DP4A kernel。

## 小型對照

`prepare_corpus.py`恢復同一批公開音訊SHA，增加3句英文各10dB／0dB白噪音與固定sparse echo，共14品質探針、3個巢狀中文延遲裁切。Noise固定seed、whole-clip RMS SNR、peak uniform gain；echo是固定delay copies，不是物理RIR或FFASR。英文只有3句，不能把合成變體當12名獨立說話者。

`run_smoke.py`沿用前次 `run_engine.py`（未修改），實際 `decode_stream()`、temperature0、context2048、threads4、requested batch4096／effective2048、ubatch512、無context／aligner；原b7798 Vulkan decoder、CUDA EP encoder。每組quality14次＋1warmup，latency3clips×3repeats＋3warmup，共92 measured calls。載入／warmups／GPU snapshot不計入wall。每clip n=3，只有median／min／max，不報可靠p95。

`lifecycle.py`為前次已驗證流程，僅state路徑改為本實驗。stop/start原CapsWriter／Immich同IDs，fingerprint Config／HostConfig／network；watchdog在parent退出／超時也恢復。每block前後檢查兩原服務停止且無外部compute process。CapsWriter Docker healthy；Immich Running＋HTTP pong，無Docker Healthcheck。初版runner在check後沒有額外accepted欄位，唯讀review指出失敗路徑可能被只看count的分析誤用，已改善後續runner；本輪另由normal-completion restoration＋完整8blocks＋逐次status推導acceptance.json，分析必須核對accepted run IDs，失敗run不能通過此gate。

已修正準備過程root-owned models目錄建立權限、host上 `/bench` symlink不可解析的驗證位置；實際改在owning container核對，失敗時尚未停止服務。未把pilot共享GPU延遲算進正式結果。獨立read-only review抽查所有output tensors的cast／transpose值映射、707覆蓋、token及shape；無Critical mapping defect。

重要限制：source `n_window_infer=800`是8秒窗口，但既有encoder採全段attention，部分探針>8秒。本輪不是原生HF parity；F16對照也經graph／precision轉換，不能把品質差異全歸因量化或微調。完整Transformers／原生Tiro推論沒有測。此限制在公開頁與conversion evidence均列明。

## 取證、離線重算與清理

`collect_cleanup.py`僅接受正常完成的8blocks，核對92calls、兩原容器ready與設定指紋後，idle test內執行 `capture_runtime.py`：hash原三模型（與前次一致）、來源／F16／Q5／Q4／ONNX與原Vulkan DSOs；ONNX CPU EP smoke finite。hash不在benchmark期間執行。保存curated metadata、results及raw logs，未提交任何模型／音檔／第三方source／build／signedURLs／完整inspect。

只刪指定11,871,814,523 logical bytes scratch、test container與state，沒有新image或全域prune。原images、model volume、兩production container IDs／StartedAt／參數及readiness保留。見results/restoration.json及cleanup.json。

從repo root離線重算：

```sh
rtk proxy python3 docs/reports/experiments/tiro-v2-20261101/analyze_smoke.py docs/reports/experiments/tiro-v2-20261101/results
rtk proxy python3 docs/reports/experiments/tiro-v2-20261101/make_page.py
```

在本目錄可 `rtk proxy sha256sum -c SHA256SUMS`。Raw logs保留原始bytes／trailing spaces；whitespace檢查排除logs。公開頁透過filehost的site preset發布，含測試輸出／reference與sanitized measurements.json，不含container IDs／fingerprints／credentials等內部維運細節。來源／作者評測與本機數據分開，不以作者資料冒充本機收益。
