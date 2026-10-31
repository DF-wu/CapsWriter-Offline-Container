# P4 獨占 GPU 實驗證據

追蹤 [#16](https://github.com/DF-wu/CapsWriter-Offline-Container/issues/16)，完整解讀見 [實驗報告](../../p4-exclusive-container-experiments-20261101.md)。這是研究／測試腳本，沒有改動正式應用。以下是本輪已獲授權的維護流程；matrix 會暫停原 CapsWriter 及 Immich，完成或出錯後以原 ID 啟動，不重建容器。

## 準備與執行

從 repo root 執行；所有 shell 入口使用 RTK。準備時正式服務仍運作。下載器固定 revision／SHA256，模型不提交。準備過程的失敗／修正仍保存在 logs 與 results/preparation-recovery.json。

```sh
rtk proxy python3 docs/reports/experiments/p4-exclusive-20261101/prepare.py
rtk proxy python3 docs/reports/experiments/p4-exclusive-20261101/extra_models.py
rtk proxy python3 docs/reports/experiments/p4-exclusive-20261101/prepare_audio.py
rtk proxy python3 docs/reports/experiments/p4-exclusive-20261101/build_transcribe.py
rtk proxy python3 docs/reports/experiments/p4-exclusive-20261101/snapshot_services.py
rtk proxy python3 docs/reports/experiments/p4-exclusive-20261101/run_all.py --pilot
rtk proxy python3 docs/reports/experiments/p4-exclusive-20261101/run_all.py --matrix
```

`prepare.py --resume` 只處理尚未完成的準備，會以 experiment label 核對並重建 **test container**；不是恢復 matrix。`run_all.py --matrix --resume` 依 schedule 跳過完成的 blocks，不去重任意殘缺 JSONL，故中途失敗須先查明該 block 是否需要排除。不得不檢查就重跑／混算資料。

本輪第一段只停 CapsWriter，Immich 在 FP32 block 啟動 GPU 工作。guard 發現後恢復 CapsWriter。受影響15 calls連同 schedule／telemetry／恢復證據封存到 `results/attempt-1`，主 FP32 JSONL移除後重測；乾淨CPU30／Vulkan15／CUDA Q5 15 calls沿用。使用者追加授權暫停 Immich後，再執行 `--matrix --resume`。

第二段完成全部390 latency後，Immich在FP32品質process期間被外部啟動。該品質process的5 measured＋1 warmup按獨占條件排除，見 `results/attempt-2/exclusion.json`；此前CPU／Vulkan／CUDA Q5共15 quality保留。封存時host無權直接truncate root-owned JSONL，過早續跑一個有效Q4品質block後主動中斷恢復，以host-owned parent目錄atomic replacement修正；見 `attempt-3/note.json`。最後續跑只補缺少品質blocks，不重測或混算390 latency。

`lifecycle.py` 在 stop 前核對 identity、image及 Config／HostConfig／網路設定 SHA256；獨立 watchdog 在 parent 消失或超時時復原。恢復先停止 test GPU 容器，再啟動兩個原 ID；CapsWriter 要 Docker healthy，Immich 要 Running＋本機 `/api/server/ping` JSON pong。Immich 無 Docker Healthcheck，證據中的 health 為 null，沒有假稱 Docker healthy。

## 取證、分析、清理

Matrix 結束即自動恢復原服務；取證時再次核對兩服務。hash 僅在計時結束後運算，避免影響 benchmark。取證會啟動 idle test container 做唯讀 hash，沒有再跑 GPU inference。

```sh
rtk proxy python3 docs/reports/experiments/p4-exclusive-20261101/collect_evidence.py
rtk proxy python3 docs/reports/experiments/p4-exclusive-20261101/cleanup_experiment.py
rtk proxy python3 docs/reports/experiments/p4-exclusive-20261101/make_report.py
```

`cleanup_experiment.py` 是本輪指定 scratch／exact test IDs 的清理紀錄，故刻意拒絕另一個 scratch；不應改成全域 prune。後續重跑需要獨立 run identity／清理 guard。只刪本輪 containers、新拉 devel image、scratch/state，保留原 production images／volume。原始權重／音訊／source/build／signed audio URLs／完整 docker inspect／cache 不保存至 repo。

離線重算（不需要 Docker／GPU／原模型）：

```sh
rtk proxy python3 docs/reports/experiments/p4-exclusive-20261101/analyze_exclusive.py docs/reports/experiments/p4-exclusive-20261101/results
rtk proxy python3 docs/reports/experiments/p4-exclusive-20261101/make_report.py
rtk proxy sha256sum -c SHA256SUMS
```

最後一行在本目錄執行；清單含原樣保存的 raw logs。logs 可能有 trailing spaces，保留 bytes 不為通過 whitespace check 而更動；只對報告／腳本作文字 whitespace 檢查。

## 資料契約

- `results/{config}.jsonl`：metadata、warmups、逐次測量；以 `type=sample` 且 `warmup=false` 計分。每組30 latency＋5 quality，13組共455 measured calls。pilot不計。
- `results/schedule.json`：1 CPU block＋24 GPU latency blocks＋13 quality blocks＝38。`attempt-1` 的污染 FP32不在主 schedule。
- `results/summary.json`：p50／nearest-rank p95、CER／WER、輸出計數、失敗／abort／retry、溫度時脈。每 clip n=10 的p95即最大值，沒有顯著性保證。
- 品質推薦底線依使用者追加要求：同批reference逐筆比原CUDA Q5的CER／英文WER，任一筆更差即排除推薦，不能用別句改善抵銷。通過5筆小樣本不表示普遍品質等價。慢且無品質優勢的Q6／Q8不列效能優先推薦。
- `results/scheme-comparison.json`／`p4-exclusive-comparison.html`：從 summary自動產生的方案比較／獨立互動頁。
- `samples.json`：同第一輪的音訊hash／reference／裁切規格。只有3個巢狀 latency clips與5個完整 quality samples；非台灣實際語料。
- `results/environment.json`、`downloads.json`、`extra-models.json`、`transcribe-build.json`：整檔hash、版本、固定source及build flags。既有decoder與完整handy模型分開記錄，避免同名Q8碰撞。
- `results/gpu-telemetry.csv`：200ms snapshots；含第一次中斷及續測，會有第二個header。主表使用有效calls前後snapshots；不要把整份CSV盲目聚合成無干擾矩陣。
- `results/restoration.json`／`cleanup.json`：exact IDs、image、設定指紋、readiness與限定資源清理。保存摘要，無production secrets。
- `p4-specific-search.json`：追加P4專用格式／kernel搜尋證據，不把泛用GGUF或其他GPU成績當P4 ASR證明。

`run_engine.py` 使用原adapter的 `decode_stream()`，非補齊80秒的獨立 `asr()`。`run_transcribe.py` 使用persistent C API，require CUDA，struct sizes先驗證。兩者都排除load、audio IO、warmup與nvidia-smi查詢；完整engine邊界與文字輸出處理不同，不作純kernel A/B。

原source固定 `d3b334e95c4d958f6d7812cb94c14a33fff427d3`；b7798固定 `c301172f660a1fe0b42023da990bf7385d69adb4`；transcribe固定 `5700f127e38dd53c4735ea23ef743b4b538eacd4`。第三方量化原浮點來源不足以證明全同源；FP32 promoted不恢復half前精度；不以升位元冒充品質恢復。

唯讀審查使用 requesting-code-review skill：未發現本輪取證／清理流程的Critical或Important問題；Minor指出requested batch4096經runtime依context2048夾到effective2048，報告已更正。審查不操作容器或GPU。
