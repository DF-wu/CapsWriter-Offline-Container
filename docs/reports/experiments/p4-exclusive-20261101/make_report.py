"""Generate comparison tables and the interactive page from measured evidence."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
summary = json.loads((RESULTS / 'summary.json').read_text())
assert len(summary) == 13 and all(v['latency']['n'] == 30 and v['quality']['n'] == 5 for v in summary.values())

# Labels describe actual changes, not hardware marketing names.
SCHEMES = {
    'cpu-fallback': ('CPU decoder fallback', '原 FP16 ONNX encoder 仍用 CUDA；Q5 decoder 強制 CPU，重現缺 EGL 路徑', '保留現況比較基準', 'decoder CPU 瓶頸；不是全 CPU 模型'),
    'vulkan-q5': ('Vulkan＋原 Q5', 'test container 補 libegl1／libglvnd0，原模型與 Vulkan library 不換', '模型不變、依賴修正小', '仍慢於本輪 CUDA；正式映像尚未修補'),
    'cuda-q5': ('CUDA＋原 Q5', '原 encoder／decoder 不換，llama.cpp b7798 編譯 CUDA SM61', '保留既有 adapter 與權重，整合變動較少', '需維護 CUDA SM61 binary；完整服務流程未回歸'),
    'cuda-f32-q5': ('CUDA＋FP32 promoted＋Q5', '既有 FP16 encoder graph／權重提升為 FP32；原 Q5 不換', '驗證 P4 的 FP32 假設', '不恢復原 FP32 精度；較耗記憶體，未見全面速度優勢'),
    'cuda-q4': ('CUDA＋Q4 decoder', '原 encoder 不換，改用上游 Q4_K decoder', '較小 decoder、10 秒片段更快', '模型來源不能證明完全同浮點 checkpoint；短裁切文字改變'),
    'cuda-w4-q4': ('CUDA＋W4 encoder＋Q4', '整套上游 W4 ONNX encoder（MatMulNBits）與 Q4 decoder', '降低 encoder／decoder 常駐記憶體', 'weight-only INT4 並非已證明的 DP4A encoder；encoder 較慢'),
    'cuda-q6': ('CUDA＋Q6 decoder', '原 encoder＋mradermacher decoder-only Q6_K', '可直接接現有 adapter，自動下載且 SHA 驗證', '更大且本輪較慢；未證明原浮點來源，不代表品質必然更好'),
    'cuda-q8': ('CUDA＋Q8 decoder', '原 encoder＋ggml-org decoder-only Q8_0', 'upstream 現成模型、可接原 adapter', '佔較多記憶體，本輪未超越原 Q5'),
    'cuda-iq4-xs': ('CUDA＋IQ4_XS decoder', '原 encoder＋mradermacher decoder-only IQ4_XS', '本輪較低延遲、較小 decoder', '2 秒片段輸出改變；只有少量品質樣本，不能宣稱無損'),
    'transcribe-q4_k_m': ('transcribe.cpp Q4_K_M', '整個 engine 改用 pinned transcribe.cpp CUDA SM61＋handy 完整 Q4 模型', '完整 C API、現成權重，GPU 常駐較省，含 Pascal DP4A 設定', '需改 adapter／前處理；streaming、context、aligner 未整合驗證'),
    'transcribe-q5_k_m': ('transcribe.cpp Q5_K_M', '整個 engine 改用 transcribe.cpp＋handy 完整 Q5 模型', '常駐 C API 可自動載入完整模型', '不是單換 decoder；較新 GGML 不保證每種長度更快'),
    'transcribe-q6_k': ('transcribe.cpp Q6_K', '整個 engine 改用 transcribe.cpp＋handy 完整 Q6 模型', 'Q6 已實際在 P4 CUDA 編譯與轉錄通過', '模型更大；本輪速度不優於低位元候選，整合成本較高'),
    'transcribe-q8_0': ('transcribe.cpp Q8_0', '整個 engine 改用 transcribe.cpp＋handy 完整 Q8 模型', '完整 Q8 候選已實測，不只看檔名推論', '佔較多記憶體；不能從 INT8 TOPS 推出全 engine 最快')}


def latency(cid, seconds, metric='p50'):
    return summary[cid]['latency']['by_sample'][f'mandarin-{seconds}s']['wall_s'][metric]


def quality(cid):
    qs = summary[cid]['quality']['by_sample']
    baseline = summary['cuda-q5']['quality']['by_sample']
    worse = [sid for sid, value in qs.items()
             if any(value[key][0] > baseline[sid][key][0] + 1e-12
                    for key in ['cer', 'wer'] if key in value)]
    libri = [v for k, v in qs.items() if k.startswith('libri-')]
    micro_wer = sum(v['wer'][0] * v['reference_words'] for v in libri) / sum(v['reference_words'] for v in libri)
    return {'english_micro_wer': micro_wer, 'noise2_cer': qs['noise2']['cer'][0],
            'codeswitch_cer': qs['codeswitch']['cer'][0],
            'normalized_matches_cuda_q5': sum(v['normalized_matches_cuda_q5'] for v in qs.values()),
            'worse_samples': worse, 'exclude_recommendation': bool(worse)}


rows = []
for cid, (label, change, benefit, cost) in SCHEMES.items():
    v = summary[cid]
    q = quality(cid)
    if q['exclude_recommendation']:
        label += '〔品質排除〕'
        cost = '依使用者要求排除推薦；' + cost
    rows.append({'id': cid, 'label': label, 'change': change, 'benefit': benefit, 'cost': cost,
                 'latency': {str(s): {m: latency(cid, s, m) for m in ['p50', 'p95']} for s in [2, 5, 10]},
                 'memory_mib': v['latency']['thermal']['total_gpu_mib'],
                 'quality_memory_mib': v['quality']['thermal']['total_gpu_mib'], 'quality': q,
                 'failures': v['latency']['failures'] + v['quality']['failures']})
(RESULTS / 'scheme-comparison.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')

table = '| 方案 | 2s p50／p95 | 5s p50／p95 | 10s p50／p95 | 10s 相對 CPU | 10s 比 CUDA Q5 延遲降低 | GPU MiB 範圍 |\n|---|---:|---:|---:|---:|---:|---:|\n'
changes = '| 方案 | 做了哪些修改 | 優點 | 缺點／限制 |\n|---|---|---|---|\n'
quality_table = '| 方案 | 3 筆英文 micro WER | Mandarin CER | 多語 codeswitch CER | 5 筆正規化與 CUDA Q5 相同 | 品質篩選 |\n|---|---:|---:|---:|---:|---|\n'
thermal_table = '| 方案 | °C | graphics MHz | SW thermal Active／60 | HW thermal Active／60 |\n|---|---:|---:|---:|---:|\n'
for row in rows:
    cid = row['id']; values = [f'{latency(cid, s):.3f}／{latency(cid, s, "p95"):.3f}' for s in [2, 5, 10]]
    speed = latency('cpu-fallback', 10) / latency(cid, 10)
    reduction = 100 * (1 - latency(cid, 10) / latency('cuda-q5', 10))
    table += f'| {row["label"]} | '+ ' | '.join(values) + f' | {speed:.2f}× | {reduction:+.1f}% | {row["memory_mib"][0]}–{row["memory_mib"][1]} |\n'
    changes += f'| {row["label"]} | {row["change"]} | {row["benefit"]} | {row["cost"]} |\n'
    q = row['quality']
    gate = '排除推薦：' + ', '.join(q['worse_samples']) if q['exclude_recommendation'] else '本輪未觀察到更差'
    quality_table += f'| {row["label"]} | {100*q["english_micro_wer"]:.2f}% | {100*q["noise2_cer"]:.2f}% | {100*q["codeswitch_cer"]:.2f}% | {q["normalized_matches_cuda_q5"]}/5 | {gate} |\n'
    t = summary[cid]['latency']['thermal']
    thermal_table += f'| {row["label"]} | {t["temperature_c"][0]}–{t["temperature_c"][1]} | {t["graphics_mhz"][0]}–{t["graphics_mhz"][1]} | {t["sw_thermal_active"]} | {t["hw_thermal_active"]} |\n'

best = min((r for r in rows if not r['quality']['exclude_recommendation']), key=lambda r: r['latency']['10']['p50'])
q5 = latency('cuda-q5', 10); cpu = latency('cpu-fallback', 10); vk = latency('vulkan-q5', 10)
iq = latency('cuda-iq4-xs', 10)
tr4 = latency('transcribe-q4_k_m', 10)
q5mem = summary['cuda-q5']['latency']['thermal']['total_gpu_mib'][1]
tr4mem = summary['transcribe-q4_k_m']['latency']['thermal']['total_gpu_mib'][1]
failures = sum(r['failures'] for r in rows)
excluded = [r['label'] for r in rows if r['quality']['exclude_recommendation']]
quality_gate = ('排除推薦：' + '、'.join(excluded)) if excluded else '這5筆樣本沒有方案比原Q5出現更高的逐筆CER／WER'
iq_status = ('依品質底線已排除推薦' if quality('cuda-iq4-xs')['exclude_recommendation']
             else '可列為第二階段品質回歸候選')
tr4_status = ('但本輪品質比原Q5更差，依使用者要求已排除推薦，不繼續投入整合。'
              if quality('transcribe-q4_k_m')['exclude_recommendation'] else
              '它對與Immich共用P4值得評估，但需完整替換engine；沒有streaming／context／aligner整合與持續共載測試，不能保證部署後同樣空間。')
next_candidates = '、'.join(SCHEMES[cid][0] for cid in ['cuda-q4', 'cuda-w4-q4']
                           if not quality(cid)['exclude_recommendation'])
w4mem = summary['cuda-w4-q4']['latency']['thermal']['total_gpu_mib'][1]
report = f'''# Tesla P4 獨占 GPU：13 組 Qwen3-ASR 量化與 runtime 實測

日期：2026-11-01（Asia/Taipei）。[研究 #14](https://github.com/DF-wu/CapsWriter-Offline-Container/issues/14)、[獨占實驗 #16](https://github.com/DF-wu/CapsWriter-Offline-Container/issues/16)。分支 `research/tesla-p4-asr-model-quantization`。

## 結論

**在已查的一手來源中，沒有找到具可重現 P4 ASR 延遲／品質證據的「P4 專用 Qwen3-ASR 量化權重」。** 找到了真正的 Pascal DP4A runtime kernel 設定，也已在本機編譯 CUDA SM61 並實測。一般 Q6／Q8／IQ4 GGUF 可在 P4 使用，不因此成為 P4 專用模型。

本輪最值得優先整合的是 **原 encoder＋原 Q5 decoder＋CUDA SM61**：保留模型及既有 adapter，10 秒片段 engine p50 **{q5:.3f}s**，相對 CPU decoder fallback **{cpu/q5:.2f}×**、相對修好 EGL 的 Vulkan **{vk/q5:.2f}×**。這主要是讓 decoder 確實使用合適的 GPU backend；正式映像尚未修改。

在本輪未觀察到品質退步的配置中，以 10 秒片段 p50 排序，最快是 **{best['label']}：{best['latency']['10']['p50']:.3f}s**。IQ4_XS 相對原 CUDA Q5 降低約 **{100*(1-iq/q5):.1f}%** 延遲，{iq_status}；2 秒裁切文字已改變，不能宣稱無損或所有情境最快。Q6、Q8、FP32 promoted 與新 runtime 均已實測，沒有因名稱、INT8 峰值或較新版本而自動獲選。

**transcribe.cpp Q4 的主要價值是較省 GPU 記憶體**：本輪 latency endpoints 上限 **{tr4mem} MiB**，相對原 CUDA Q5 **{q5mem} MiB** 少約 **{100*(1-tr4mem/q5mem):.1f}%**；10 秒 p50 **{tr4:.3f}s**。{tr4_status}

兼顧Immich共用空間時，**W4 encoder＋Q4 decoder**保留原adapter，這5筆品質未退步；latency endpoints上限 **{w4mem} MiB**，比原CUDA Q5少約 **{100*(1-w4mem/q5mem):.1f}%**。代價是2／5秒較慢，10秒約 **{latency('cuda-w4-q4',10):.3f}s**，不適合作純速度首選。完整transcribe引擎的Q4與Q5已因品質退步排除；Q6／Q8雖通過本輪小樣本底線，但需新adapter且整體速度未勝原Q5，不優先整合。

## 每個方案改了什麼

{changes}
所有 decoder-only 變體沿用原 FP16 ONNX encoder（除明列 FP32／W4 外）。CUDA b7798 source 固定 `c301172f660a1fe0b42023da990bf7385d69adb4`；transcribe.cpp 固定 `5700f127e38dd53c4735ea23ef743b4b538eacd4`。後者是完整 engine／encoder／前處理替換，**不是只換位元數的 A/B**。

## 完整延遲與記憶體比較

每格為 engine 秒數 p50／p95，各 clip 10 次；正的延遲降低表示比 CUDA Q5 更快，負值表示更慢。GPU MiB 為每次呼叫前後的總使用量範圍，獨占 GPU 下近似該配置常駐量；**不是 process 精確計量，也不是完整瞬間峰值**。

{table}
不能把這輪 cooled 獨占數字直接除以上輪共用、455MHz 熱降頻的數字，當成單一模型優化收益。比較方案請用本表內基準；散熱／其他服務的效應另見[第一輪報告](p4-container-experiments-20261101.md)。2／5／10 秒是同段 Mandarin 音檔的巢狀裁切，不是獨立語料；輸出 token 數不同也會影響延遲。

## P4 專用證據與未採用候選

P4 是 SM61／GP104，具有 INT8 DP4A、沒有 Tensor Cores。GGUF 的低位元儲存格式還需要解包、scale、activation 量化及對應 kernel；不能從 22 INT8 TOPS 推導 Q8 比 Q5 快。ORT 的 FP16 輸入也可能以 FP32 計算，不能宣稱 encoder 必慢 64 倍。逐算子與工具鏈證據見[硬體／kernel 附錄](p4-asr-hardware-kernel-evidence-20261101.md)。

- transcribe 的 [Pascal DP4A 設定](https://github.com/handy-computer/transcribe.cpp/blob/5700f127e38dd53c4735ea23ef743b4b538eacd4/ggml/src/ggml-cuda/mmq-config-pascal-dp4a.cuh) 由 [mmq.cuh](https://github.com/handy-computer/transcribe.cpp/blob/5700f127e38dd53c4735ea23ef743b4b538eacd4/ggml/src/ggml-cuda/mmq.cuh) 引入。本輪 SM61 cubins 與 CUDA backend 已驗證；沒有 kernel profiler trace，因此不把來源檔存在等同每個 ASR 算子皆走該路徑。
- [llama.cpp P4 tile proposal #30235](https://github.com/ggml-org/llama.cpp/issues/30235) 的 Q8 tile 64→128 約 +12% 是 BGE prompt processing；不是 Qwen3-ASR，也不是所有 Q4／Q6 都受益。本輪沒有套用，沒有把 12% 寫成本機 ASR 成績。
- [vllm-pascal](https://github.com/ba2512005/vllm-pascal) 的 ASR 驗證是 P40，沒有足夠 P4 實測及專用 checkpoint 證據；未與正式系統混裝。
- [VinRobotics recipes](https://github.com/VinRobotics/model-quantization-recipes/blob/main/recipes/qwen3-asr/README.md) 目標含 Orin INT8／INT4、RTX5090 FP8／NVFP4，不可直接移植性能結論到 Pascal。
- [bpfio/TeslaP4](https://github.com/bpfio/TeslaP4) 是 P4 LLM 部署指南；[audio.cpp #888](https://github.com/0xShug0/audio.cpp/issues/888) 是 Pascal 相容性討論，皆不提供合格的本題 P4 ASR 量化 A/B。

補充搜尋查詢、request IDs、失敗與取捨保存於 [p4-specific-search.json](experiments/p4-exclusive-20261101/p4-specific-search.json)。Tavily 原 endpoint 曾回 HTTP521、替代 endpoint 一次 quota reserve 500，均有保留；沒有把搜尋失敗當作「世界上不存在」的證明。Q6／Q8 runtime、模型契約與下載證據見[候選附錄](p4-q6-alternative-runtime-evidence-20261101.md)。

## 品質與模型來源限制

依使用者追加要求，**比現有CapsWriter原Q5品質更差的量化方案不列入推薦**。以同批完整reference逐筆比較CER／英文WER，任何一筆較差即排除，不以其他句的改善抵銷。{quality_gate}。這只是小樣本篩選，通過不代表實際使用品質等價；原始結果仍保留供稽核。decoder-only Q6／Q8比原CUDA Q5更慢、更佔記憶體且本輪品質無額外收益，不列效能優先推薦。

每配置 5 筆完整 reference 樣本各一次：3 筆 LibriSpeech 英文、Mandarin noise2（22.833s）、英／法／義／西語 codeswitch（6.230s）。不是中英混語，沒有台灣口音／使用者實際語料。2／5／10 秒裁切無對齊 reference，不計 CER。CER 採 NFKC、lowercase、移除標點／空白、英文 Mr.→mister；不做繁簡或數字等價轉換。WER 保留 apostrophe 詞形。

{quality_table}
有效主矩陣共 **390 latency＋65 quality calls**，狀態失敗共 **{failures}**。非空、未 abort 只表示呼叫成功，不表示文字正確。逐字輸出、CER／WER、重試、時間、各階段計時與 metadata 均保留於 [results](experiments/p4-exclusive-20261101/results/summary.json)。原 adapter 記錄 prefill／generation／n_generate；transcribe 公開 timings 沒有等價 token 計數，不能把跨 runtime wall 差異全歸因 kernel。

decoder-only Q6／Q8／IQ4_XS 均為 `qwen3vl`、311 tensors、2048 embedding，成功接入既有 b7798 adapter；handy 完整模型為 `qwen3_asr`、709 tensors，不能直接當 decoder 替換。所有下載已驗整檔 SHA256，但第三方 Q4／Q6／IQ4 與原 Q5 的原始浮點同源性不足以嚴格證明；不是控制完全相同 checkpoint 的純量化精度實驗。FP32 promoted 由原 half graph 提升，不恢復半精度前資訊。沒有從 Q5 升位元來假裝 Q6 品質恢復。

## 實驗控制與散熱

- Tesla P4 7,680 MiB；CUDA11.8、cuDNN8、ORT1.18.0；完整 package／model／library hashes 見 [environment.json](experiments/p4-exclusive-20261101/results/environment.json)。未更動 host driver、功耗、時脈或風扇。
- 原模型 volume 只讀掛載。test container 6 CPUs／16GiB、compiler 8 CPUs／12GiB，無發布 ports。apt／pip 只在 disposable test container，未修改正式檔案。
- 原 adapter 呼叫實際 `QwenASREngine.decode_stream()`，temperature0、seed1729+repeat、context2048、decoder threads4／requested batch4096（runtime實際2048）／ubatch512、flash attention 開啟、無 context／aligner；ORT thread 選項沿用原設定。
- transcribe 使用常駐 C API，明確指定 CUDA backend、threads4、context2048、greedy default、zh／en／auto、無 timestamps／speculative decoding；公開 struct_size 與 ctypes size 一致。native streaming／hotwords／aligner／服務整合未驗。
- CPU baseline10 repeats；12個 GPU配置各2 blocks×5 repeats×3 clips，每 process3暖機排除。第二 block 輪轉順序，品質每 process1暖機＋5測量。不是全隨機交錯，沒有置信區間或顯著性主張。每 clip n=10，nearest-rank p95就是最大值。
- block間冷卻目標78°C、通常最多60s，開始最多120s；每 block前檢查原服務停止且無其他 compute process。逐次前後 snapshots＋200ms全域 telemetry保留。前次中止與續測共用 telemetry含重啟 header，分析主表採有效 calls 的 endpoints，不把受污染區段混進數據。
- wall排除載入、音檔讀取、暖機與GPU snapshot開銷，含 engine轉錄；**不是 client放開按鍵到螢幕上字**。兩 runtime的呼叫及輸出處理邊界略異，僅作完整 engine替代比較。

{thermal_table}
## 中斷、恢復與 Portainer 設定保護

首次只停 CapsWriter，Immich 隨後啟動約4.2GiB GPU工作。guard中止，CapsWriter自動恢復 healthy且設定指紋相同。該次 FP32 block15 calls已封存於 [attempt-1](experiments/p4-exclusive-20261101/results/attempt-1/exclusion.json)，**排除主分析並重測**；此前無干擾的CPU30／Vulkan15／CUDA Q5 15 calls保留。

使用者進一步授權暫停 Immich。續測只 stop／start兩個既有ID，不 recreate、不compose down/up。獨立watchdog可在parent退出或超過deadline時復原。CapsWriter用Docker healthy；Immich沒有Docker Healthcheck，採Running＋`/api/server/ping` HTTP200且JSON pong驗證，不偽造healthy。

第二段品質階段，原Immich在19:18:57 UTC被外部啟動（RestartCount=0；啟動來源未定位），guard再次中止並恢復兩服務。FP32品質process雖未顯示額外GPU配置量，仍依獨占條件排除5 calls並重測；390 latency與此前15 quality不受影響。證據見 [attempt-2](experiments/p4-exclusive-20261101/results/attempt-2/exclusion.json)。封存時root-owned JSONL的host寫入權限錯誤，曾過早續跑一個有效Q4品質block，隨即主動中斷恢復服務，以atomic replacement修正資料後再續測；保留 [attempt-3](experiments/p4-exclusive-20261101/results/attempt-3/note.json)，沒有隱藏中斷或混算被排除數據。沒有修改其他管理daemon或restart policy。

比對完整 Config、HostConfig及選定網路設定 canonical SHA256，涵蓋環境變數／command／labels／ports／mounts／GPU devices／restart policy；僅保存摘要，不公開inspect裡的secret。恢復的兩個容器仍是原ID與原image。證據見 [restoration.json](experiments/p4-exclusive-20261101/results/restoration.json)，清理見 [cleanup.json](experiments/p4-exclusive-20261101/results/cleanup.json)。

準備期間的root apt權限、FFmpeg4.4.2與host9.0.2的WAV bytes差異、兩來源Q8檔名碰撞都已修正並保留[紀錄](experiments/p4-exclusive-20261101/results/preparation-recovery.json)。Q8碰撞是下載命名錯誤，非模型runtime不相容；現在以來源＋revision命名，handy原檔重新驗hash。

## 重現、清理與下一步

[README](experiments/p4-exclusive-20261101/README.md) 保存準備、下載、SM61編譯、pilot、matrix、恢復、分析與清理步驟。[SHA256SUMS](experiments/p4-exclusive-20261101/SHA256SUMS) 涵蓋公開證據，未提交模型、音訊、第三方source/build、signedURL、完整inspect或cache。只刪本次test／compiler containers、新拉的devel image與指定scratch，不作全域prune；正式images／model volume保留，原模型三個SHA與第一輪一致。

後續整合優先：修EGL依賴與GPU backend檢查，再提供b7798 CUDA SM61選擇；保留原Q5作低變動基準。小樣本品質未退步的後續候選：{next_candidates}，需補台灣口音、數字、人名、混語、context、sampling及長音訊回歸；前者偏速度、後者偏GPU記憶體。完整runtime的Q6／Q8需另補功能整合，這輪不優先。品質較差者排除。正式服務沒有部署本輪優化。
'''
(HERE.parents[1] / 'p4-exclusive-container-experiments-20261101.md').write_text(report)

html = '''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>
body{font-family:var(--font-sans,system-ui);color:var(--foreground,#20242c);margin:0;font-size:14px;line-height:1.55}*{box-sizing:border-box}p{margin:10px 0}small,.muted{color:var(--muted-foreground,#64748b)}.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:16px 0}select,button{font:inherit;color:inherit;background:var(--card,#fafafa);border:1px solid var(--border,#ddd);border-radius:6px;padding:6px 8px}button{cursor:pointer}.row{width:100%;display:grid;grid-template-columns:minmax(0,1fr) 80px 76px;gap:8px;text-align:left;border:0;border-bottom:1px solid var(--border,#ddd);border-radius:0;background:transparent;padding:9px 0;align-items:center}.row[aria-pressed=true]{color:var(--accent-foreground,#2563eb)}.name{font-size:13px}.track{height:6px;background:var(--secondary,#e2e8f0);border-radius:6px;margin-top:5px}.bar{height:6px;background:var(--chart-1,#3b82f6);border-radius:6px}.number{text-align:right;font-variant-numeric:tabular-nums}.panel{padding:16px;border:1px solid var(--border,#ddd);border-radius:var(--radius,10px);margin-top:16px;background:var(--card,#fafafa)}.panel p{margin:6px 0}.caption{font-size:12px}.head{display:grid;grid-template-columns:minmax(0,1fr) 80px 76px;gap:8px;font-size:12px;color:var(--muted-foreground,#64748b)}@media(max-width:440px){.row,.head{grid-template-columns:minmax(0,1fr) 63px 62px;font-size:12px}.name{font-size:12px}}
</style><p><strong>13 組 P4 獨占實測</strong> · 390 次延遲＋65 次品質呼叫</p><p class="muted">點選方案查看修改與取捨。未找到具合格一手實測證據的 P4 專用量化權重。</p>
<div class="controls"><label>音訊 <select id="clip"><option value="2">2 秒</option><option value="5">5 秒</option><option value="10" selected>10 秒</option></select></label><label>指標 <select id="metric"><option>p50</option><option>p95</option></select></label><label>比較 <select id="baseline"><option value="cuda-q5">CUDA 原 Q5</option><option value="cpu-fallback">CPU fallback</option></select></label><label><input type="checkbox" id="all"> 顯示全部實驗</label><label><input type="checkbox" id="cpu"> 顯示 CPU</label></div><p id="base" class="caption"></p><div class="head"><span>方案／延遲（越短越快）</span><span class="number">秒</span><span class="number">延遲降低</span></div><div id="chart"></div><div id="detail" class="panel"></div><p class="caption">每個長度各 10 次；p95 為最大值。3 個巢狀中文裁切，5 筆品質樣本；無台灣口音。GPU MiB 是呼叫前後總量，非瞬間峰值。不同權重與完整 runtime 的比較含來源、前處理及輸出差異；不是普遍加速保證。</p><script>
const rows=__DATA__;let selected='cuda-q5';const $=id=>document.getElementById(id);function draw(){const c=$('clip').value,m=$('metric').value,b=rows.find(r=>r.id===$('baseline').value),v=b.latency[c][m];$('base').textContent=`基準：${b.label} ${v.toFixed(3)} 秒 · 數值為 engine wall，未含 client／網路／上屏。`;const shown=rows.filter(r=>(r.id!=='cpu-fallback'||$('cpu').checked)&&($('all').checked||['cuda-q5','cuda-q4','cuda-w4-q4'].includes(r.id)||(r.id==='cpu-fallback'&&$('cpu').checked))).sort((a,b)=>a.latency[c][m]-b.latency[c][m]);const max=Math.max(...shown.map(r=>r.latency[c][m]));$('chart').replaceChildren(...shown.map(r=>{const x=r.latency[c][m],p=100*(1-x/v);const el=document.createElement('button');el.className='row';el.setAttribute('aria-pressed',r.id===selected);el.innerHTML=`<span class="name">${r.label}<div class="track"><div class="bar" style="width:${100*x/max}%"></div></div></span><span class="number">${x.toFixed(3)}</span><span class="number">${p>=0?'+':''}${p.toFixed(1)}%</span>`;el.onclick=()=>{selected=r.id;draw()};return el}));const r=rows.find(r=>r.id===selected),q=r.quality;$('detail').innerHTML=`<strong>${r.label}</strong><p>${r.change}</p><p>優點：${r.benefit}</p><p>限制：${r.cost}</p><p class="caption">GPU latency ${r.memory_mib.join('–')} MiB／quality ${r.quality_memory_mib.join('–')} MiB · 英文 WER ${(100*q.english_micro_wer).toFixed(2)}% · 中文 CER ${(100*q.noise2_cer).toFixed(2)}% · 多語 CER ${(100*q.codeswitch_cer).toFixed(2)}% · 與 CUDA Q5 正規化相同 ${q.normalized_matches_cuda_q5}/5 筆</p>`}for(const id of ['clip','metric','baseline','all','cpu'])$(id).onchange=draw;draw();console.log('Loaded configurations',rows.length,'sample count',390+65);
</script></html>'''.replace('__DATA__', json.dumps(rows, ensure_ascii=False).replace('</', '<\\/'))
(HERE / 'p4-exclusive-comparison.html').write_text(html)
print('Generated report, scheme-comparison.json and standalone interactive comparison.')
