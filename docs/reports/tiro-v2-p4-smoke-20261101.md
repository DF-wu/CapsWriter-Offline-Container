# Tiro v2：P4快速自行量化與既有runtime測試

2026-11-01，[Issue #17](https://github.com/DF-wu/CapsWriter-Offline-Container/issues/17)，branch `research/tiro-v2-p4-smoke`。

**轉換後能用現有runtime，不推薦本輪模型替換。** 四組共92有效calls（56 quality／36 latency），無call失敗。Tiro英文遠場BF16，不能直接替換目前split ONNX＋GGUF；自行填入全部707來源tensors，397encoder initializer／311decoder tensor，token IDs與merges一致後，b7798 F16→Q5_K_M／Q4_K_M量化通過，並用原b7798 Vulkan library＋ORT1.18 CUDA encoder實跑。Q5／Q4量化約96／110秒；test-only補libegl1，正式服務不變。

| 配置 | 0dB英文micro WER | 完整中文CER | 多語CER | 10秒p50（3次） | 品質推薦 |
|---|---:|---:|---:|---:|---|
| 目前模型 Q5 | 10.17% | 3.77% | 10.00% | 0.892s | 基準 |
| Tiro F16 對照 | 13.56% | 3.77% | 8.00% | 1.514s | 排除：有逐筆退步 |
| Tiro Q5_K_M | 15.25% | 9.43% | 8.00% | 0.881s | 排除：有逐筆退步 |
| Tiro Q4_K_M | 20.34% | 3.77% | 8.00% | 0.783s | 排除：有逐筆退步 |

Q4約快12%，仍因品質退步排除。每句對比及accepted run IDs見 results/summary.json、acceptance.json；完整逐次輸出見各JSONL。只3句英文及其9個合成變體、1中文、1多語；0dB三句共59reference words，不是獨立遠場資料集，沒有台灣口音。延遲三個巢狀裁切各3次；不報p95／顯著性。相對同一Vulkan runtime原Q5比較，不與前次CUDA數字混算。

作者[模型卡](https://huggingface.co/rohansheth/tiro-qwen3-asr-1.7b-v2/blob/5dd5f33e0dbaa46254fc552a658d90152f1eaf75/README.md)稱FFASR WER13.41→12.01，約10.4%相對降錯；近場3.76→3.80。屬作者英文評測，未獨立重現，不是P4量化速度或中文收益。來源4,076,193,080 bytes，SHA256 c57399f84b00cadfb3d1d62a3a7b3375f54297e26fdb5dc78138a076a5e8a1b5；revision 5dd5f33e0dbaa46254fc552a658d90152f1eaf75。

關鍵限制：encoder沿用既有圖的整段attention，source n_window_infer800意味8秒窗口，部分測試超過8秒。此為目前runtime的轉換可行性與行為測試，不是原生HF數值等價。F16也經圖轉換與cast，不能把退步全歸因量化或微調；沒有完整Transformers logits parity。定址position table為原圖deterministic sinusoid；其餘learned weights全部替換。Q5／Q4由F16新量化，不是從原Q5升位元。未證明P4最優量化；未build新CUDA kernel或作DP4A profiler。

採用建議：現有中文／多語CapsWriter保留原模型。需要真實英文遠場時，另以原生Tiro runtime和實際遠場資料評估；本輪不繼續擴張runtime或部署。通過格式不等於通過品質。

兩個原CapsWriter／Immich只stop/start同ID，完整Config／HostConfig／網路設定指紋一致；CapsWriter healthy，Immich Running＋ping/pong。原模型SHA未變，volume只讀。限定test container／scratch／state已清理，沒有新image或全域prune。公開頁site/index.html由make_page.py產生，只公開研究樣本與資料，不含private inspect或憑證。
