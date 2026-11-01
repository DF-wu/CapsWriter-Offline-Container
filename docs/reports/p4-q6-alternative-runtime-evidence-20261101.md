# P4 Qwen3-ASR 1.7B：Q6／Q8 與替代 runtime 的補充證據

**後續實測已完成：**見[13組獨占P4實驗](p4-exclusive-container-experiments-20261101.md)。本文保留實測前的候選、來源及build recipe；其中「未測／未編譯」描述指當時研究階段，最新是否通過及品質排除以實驗報告為準。

研究日期：2026-11-01（Asia/Taipei）。本檔補充既有[量化研究](tesla-p4-asr-model-quantization-20261101.md)及[artifact manifest](tesla-p4-asr-model-artifacts-20261101.json)，不重做完整來源研究。

**交付結論：完整 runtime 首選 `handy-computer/transcribe.cpp`，現成 1.7B Q6_K／Q8_0、CUDA SM61 原始碼路徑、常駐 batch CLI 與 benchmark 工具均可核對。另新找到 `mradermacher` 的 decoder-only Q6_K：實際 GGUF header 是 `qwen3vl`，是現有 b7798 adapter 最值得先做載入／品質 smoke test 的直接候選。** ggml-org 現成 decoder Q8_0 同樣為 `qwen3vl`；先前「完整 GGML 路線需要新 adapter」不等於其 decoder-only 一律不能給既有 ONNX encoder 使用。

本研究**未執行 CUDA build、GPU 推論或 P4 benchmark**。沒有操作 GPU、容器或正式服務，也沒有下載完整模型。只讀原始碼、HF／GitHub API，下載約46 MB 的 transcribe.cpp source tar；以 HTTP Range 讀三個 GGUF 各16 MiB及另一個8 MiB header。以下支援判定是來源／契約可行性，不是本機跑通保證。

## 1. Parent 可直接採用的優先順序

1. **維持現有 split FP16 ONNX encoder + b7798 CUDA decoder，先試 mradermacher Q6_K、ggml-org Q8_0。** 不換 llama `.so`；只在隔離 benchmark 中換 decoder path。Q6 檔名、大小、SHA 與實際 tensor 契約見第4節。這是最少改動的試驗，仍須完成 b7798 載入、prompt embedding、中文品質實測。
2. **完整 runtime A/B：transcribe.cpp Q8_0 → Q6_K → Q4_K_M。** Q8 作移植正確性對照，Q6滿足高品質量化要求，Q4_K_M是有作者英語 WER 證據的強量化候選。此路線替換 encoder/preprocessing/decoder，結果不能只歸因於 decoder 位元。
3. **全原生 llama.cpp text + Q8 mmproj** 作後續完整 runtime。可用同一個 decoder Q6（現成或自 BF16量化）配 Q8 encoder；有常駐 HTTP API，但需要重驗 ASR prompt／language token／輸出清理。
4. **CrispASR** 作混合精度備選。已有 Q4 decoder + Q8 audio tower；現成1.7B Q6未在 pinned HF清單發現，但其 quantizer支援從F16產生Q6並保留Q8 audio floor。首輪不優先於上述兩條較直接路線。

服務停止、原ID restart、Portainer Config／HostConfig不變，由 parent持有；本研究沒有碰該流程。

## 2. 完整 runtime 首選：transcribe.cpp

### 固定來源、build target與SM61證據

- Runtime commit：`5700f127e38dd53c4735ea23ef743b4b538eacd4`；[source tar](https://codeload.github.com/handy-computer/transcribe.cpp/tar.gz/5700f127e38dd53c4735ea23ef743b4b538eacd4)。GGML隨來源vendored，`ggml/UPSTREAM`指定`d7cb574130e6f01ad25b3289685489200febcd74`及三個下游patch；tar不是需要另補submodule的空殼。[T1][T2]
- 根CMake必須用 **`TRANSCRIBE_CUDA=ON`**，因為它會以FORCE把`GGML_CUDA`設成該值；只給`GGML_CUDA=ON`可能被根設定覆蓋。`TRANSCRIBE_BUILD_TOOLS=ON`才有quantize與bench。[T1][T3]
- vendored CUDA CMake明列61=Pascal DP4A，CUDA<13預設架構清單含61；可明確給`CMAKE_CUDA_ARCHITECTURES=61`。這是SM61 build path證據；本研究沒有把它升格成實際CUDA11.8完整build已通過。[T4]

以下命令是提供parent在隔離環境執行的recipe，**本研究未執行**。`/work`為範例掛載根目錄，須依parent實際環境代換；不對正式checkout或容器操作。

```sh
rtk proxy curl -fL https://codeload.github.com/handy-computer/transcribe.cpp/tar.gz/5700f127e38dd53c4735ea23ef743b4b538eacd4 -o /work/transcribe-src.tar.gz
rtk proxy tar -xzf /work/transcribe-src.tar.gz -C /work
rtk proxy cmake -S /work/transcribe.cpp-5700f127e38dd53c4735ea23ef743b4b538eacd4 -B /work/transcribe-build \
  -DCMAKE_BUILD_TYPE=Release -DTRANSCRIBE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=61 -DTRANSCRIBE_METAL=OFF \
  -DTRANSCRIBE_VULKAN=OFF -DTRANSCRIBE_BUILD_TESTS=OFF \
  -DTRANSCRIBE_BUILD_TOOLS=ON -DTRANSCRIBE_USE_SYSTEM_BLAS=OFF
rtk proxy cmake --build /work/transcribe-build --target transcribe-cli transcribe-bench transcribe-quantize -j4
```

優先沿用parent已有CUDA11.8／SM61工具鏈；若該固定新版GGML遇到11.8 compiler/API問題，保存原始error，不把既有b7798能build視為此commit必然能build。CUDA13不適合作為Pascal編譯工具鏈替代品。[T4]

### 常駐process與benchmark

**最實用的可調參入口是`transcribe-cli --batch`**，載入一次model及session，再逐檔跑；檔案清單可包含一次warmup及30次同樣音訊，第一列丟棄。`--batch-size 1`固定序列、單請求。JSONL保留每次文字及mel/encode/decode時間；不要把這三項總和誤作client release→final延遲。[T5][T6]

```sh
rtk proxy ffmpeg -i /work/audio/input.wav -ar 16000 -ac 1 -c:a pcm_s16le /work/audio/input-16k.wav
rtk proxy python - <<'PYSAMPLES'
from pathlib import Path
Path('/work/audio/repeat31.list').write_text('/work/audio/input-16k.wav\n' * 31)
PYSAMPLES
rtk proxy /work/transcribe-build/bin/transcribe-cli \
  -m /work/models/Qwen3-ASR-1.7B-Q6_K.gguf \
  --backend cuda --threads 4 --n-ctx 4096 \
  --language zh --timestamps none \
  --batch /work/audio/repeat31.list --batch-size 1 --batch-jsonl
```

`--device N`是`--list-devices`所列的全域device index，**不是保證N=0就是CUDA0**；可只指定`--backend cuda`由runtime選卡，並保存實際backend／device載入log。不能只看flag宣稱全GPU。[T5]

`transcribe-bench`則原生提供**load once、warmup、每次steady_clock wall_ms及stage timings、JSON**：[T7]

```sh
rtk proxy /work/transcribe-build/bin/transcribe-bench \
  --model /work/models/Qwen3-ASR-1.7B-Q6_K.gguf \
  --sample /work/audio/input-16k.wav --backend cuda --threads 4 \
  --warmup 1 --iters 30 --json-out /work/results/transcribe-q6.json
```

**重要限制：此pin的`transcribe-bench`沒有`--language`、`--prompt`、`--n-ctx`選項**，使用預設run/session params。不能把上面bench數字與固定zh／4096的adapter數字當成完全同條件；需要這些控制時，先用batch CLI，或用C API寫極薄計時loop。API順序是`transcribe_model_load_params_init`→`backend=TRANSCRIBE_BACKEND_CUDA`→`transcribe_model_load_file`一次；`transcribe_session_params_init`→`n_ctx=4096`、`n_threads=4`→`transcribe_session_init`一次；`transcribe_run_params_init`→`language="zh"`與相同prompt→每次`transcribe_reset_timings`、steady_clock包住`transcribe_run`，讀`transcribe_get_timings`／`transcribe_full_text`。這是API使用指引，未編譯的新harness不能寫成已驗證工具。[T7][T8]

### 語言、greedy、context與音訊契約

- `--language zh`映射成publisher canonical `Chinese`並加入`language Chinese<asr_text>`前綴；省略則自動辨識。context文字用`--prompt TEXT`，custom vocabulary可用`--vocabulary`，寫入system訊息。空prompt仍要與原adapter的空system/prefix規則比對。[T9][T10]
- Qwen3-ASR graph使用`ggml_argmax`作greedy token選擇；CLI的`--temperature`是**Whisper專用**，不要假裝它能調Qwen greedy。Qwen這裡沒有可設定top-p/temperature的同等sampling介面。[T11][T5]
- `--n-ctx 4096`降低session context/KV cap；不能增加到超過模型65536。它也降低可接受音訊長度，須看`transcribe_session_get_limits()`並記錄truncated/repetition錯誤。短語音比較先用4096；不能直接套官方87分鐘上限到此較小context。[T9][T10]
- 輸入為**16kHz、mono WAV**；C API為mono float PCM。header具128 mel、FFT/window400、hop160、Slaney、periodic Hann等前處理參數。只有取樣率一致不足以保證與ONNX pipeline完全相同。[T9][T12]
- 此runtime的Qwen不提供timestamps、VAD、diarization或即時streaming；保留CapsWriter外層分段與原aligner需另做整合，不是可直接換正式服務的結論。[T9]
- 原始碼明記K-quant token embedding `get_rows`可能造成CPU/CUDA graph split；即使主backend是CUDA也不能宣稱沒有CPU或host copy。這是profiling需核對的具體風險。[T11]

### Q6的來源與強量化品質證據

官方runtime文件的reproduction由`Qwen/Qwen3-ASR-1.7B@7278e1e70fe206f11671096ffdd38061171dd6e5`轉為BF16 GGUF，再使用`transcribe-quantize`。bulk quantizer對每個preset都傳入同一個reference input，而非先Q8再逐級轉。**文件支持的正規pipeline是原BF16→Q6／Q8／Q4，未找到其聲明Q6由Q8 requantize。** 但HF已發佈每個檔案的精確build log沒有完整附上，不能單靠SHA證明其歷史輸入；若要求嚴格可稽核來源，下載BF16並自行重做兩份。[T9][T13][T14]

```sh
rtk proxy /work/transcribe-build/bin/transcribe-quantize /work/models/Qwen3-ASR-1.7B-BF16.gguf /work/models/self-Q6_K.gguf --quant Q6_K
rtk proxy /work/transcribe-build/bin/transcribe-quantize /work/models/Qwen3-ASR-1.7B-BF16.gguf /work/models/self-Q8_0.gguf --quant Q8_0
```

工具本身可以對已量化tensor反量化再量化；因此**「支援Q6」不代表任意輸入都有原浮點Q6品質**。Q5→Q6／Q8不恢復丟失資訊；Q8→Q6要標成Q8 requantize。policy的Q6是主要linear Q6_K、形狀不合K-block時fallback Q8、norm／conv保浮點；Q4_K_M把token embedding提升Q6及attention output保Q8，與llama.cpp同名preset的混合策略不完全相同。[T13][T14]

作者LibriSpeech test-clean（2620段、batch1、Apple M4）WER：BF16/F16/Q8=1.62%、Q6/Q5=1.65%、Q4_K_M=1.81%。可支持Q4_K_M值得試及Q6品質優先，**不能宣稱P4速度、繁中非劣化或所有語言無損**。FLEURS zh的7.14% CER屬作者Q8資料，沒有對應Q6／Q4的繁中量化比較。[T12]

## 3. 完整runtime模型URL／hash（發佈方API值）

handy模型revision固定為`0e0a596e3959c3f3f2ab7d443a5b6f0f4a4bba6f`。以下皆完整encoder+decoder GGUF，**不能直接交給既有llama adapter**。[T12][T15]

- [Qwen3-ASR-1.7B-BF16.gguf](https://huggingface.co/handy-computer/Qwen3-ASR-1.7B-gguf/resolve/0e0a596e3959c3f3f2ab7d443a5b6f0f4a4bba6f/Qwen3-ASR-1.7B-BF16.gguf)：4,083,087,904 bytes；SHA-256 `57db745f8ec3ad2ea391b0205661e7d74f76158877a33f151e4d4624ed2b7cc9`。
- [Qwen3-ASR-1.7B-Q4_K_M.gguf](https://huggingface.co/handy-computer/Qwen3-ASR-1.7B-gguf/resolve/0e0a596e3959c3f3f2ab7d443a5b6f0f4a4bba6f/Qwen3-ASR-1.7B-Q4_K_M.gguf)：1,319,830,496 bytes；SHA-256 `b7afe3674f653fa84f712ed2440353c6e7cf7f93697fef76b05a26538b24844e`。
- [Qwen3-ASR-1.7B-Q6_K.gguf](https://huggingface.co/handy-computer/Qwen3-ASR-1.7B-gguf/resolve/0e0a596e3959c3f3f2ab7d443a5b6f0f4a4bba6f/Qwen3-ASR-1.7B-Q6_K.gguf)：1,692,554,208 bytes；SHA-256 `c75a961b7134a6c952d89797865cb0d0376876185aee04ef6d12c31c2952e4e1`。
- [Qwen3-ASR-1.7B-Q8_0.gguf](https://huggingface.co/handy-computer/Qwen3-ASR-1.7B-gguf/resolve/0e0a596e3959c3f3f2ab7d443a5b6f0f4a4bba6f/Qwen3-ASR-1.7B-Q8_0.gguf)：2,185,030,624 bytes；SHA-256 `9a0d81792dfea2d5f278b8a63deb3ea6e02139ce42c2301f32ea19c4f77526b7`。

完整下載後需本機`rtk proxy sha256sum FILE`比對；此研究只讀header，**沒有重算整檔SHA**。

## 4. 新發現：現成decoder-only Q6／Q8與既有adapter契約

### 4.1 mradermacher Q6_K：具有直接測試價值

HF revision：`664b960e358348943aefcdc26b3cf09eff1fb0ac`。[D1]

- [Qwen3-ASR-1.7B.Q6_K.gguf](https://huggingface.co/mradermacher/Qwen3-ASR-1.7B-GGUF/resolve/664b960e358348943aefcdc26b3cf09eff1fb0ac/Qwen3-ASR-1.7B.Q6_K.gguf)：1,673,003,488 bytes；SHA-256 `534b653ee0074264c80533df9637e9eb556c04457f30cd1ba08fe48dcb608263`。
- [Qwen3-ASR-1.7B.Q8_0.gguf](https://huggingface.co/mradermacher/Qwen3-ASR-1.7B-GGUF/resolve/664b960e358348943aefcdc26b3cf09eff1fb0ac/Qwen3-ASR-1.7B.Q8_0.gguf)：2,165,035,488 bytes；SHA-256 `49eefdfad5b5b272220d150ca0cd2ca1734fe2e3651c412ea686e35299f4d7d7`。

**Q6實際HTTP206 header讀取結果**（讀8MiB，含全部metadata及311個tensor descriptor，未讀權重本體）：

```text
general.architecture = qwen3vl
general.file_type = 18 (MOSTLY_Q6_K)
qwen3vl.block_count = 28
qwen3vl.embedding_length = 2048
qwen3vl.feed_forward_length = 6144
qwen3vl.attention.head_count / head_count_kv = 16 / 8
qwen3vl.attention.key_length / value_length = 128 / 128
qwen3vl.rope.dimension_sections = [24,20,20,0]
qwen3vl.rope.freq_base = 1000000
qwen3vl.n_deepstack_layers = 0
tokenizer.ggml.model / pre = gpt2 / qwen2
BOS/EOS = 151645; token vocabulary array = 151936 entries
output.weight       [2048,151936] GGML_TYPE_Q6_K (14)
token_embd.weight   [2048,151936] GGML_TYPE_Q6_K (14)
output_norm.weight  [2048]        F32
blk.0.attn_k.weight [2048,1024]   Q6_K
```

這與既有adapter要求的`token_embd.weight`、2048 audio embeddings與4組position layout相吻合。adapter的`get_token_embeddings_gguf`會依tensor qtype使用`gguf.quants.dequantize`，不是把所有embedding硬當Q5；本repo附帶Q6_K／Q8_0解量化實作。故判定是：**可作直接更換decoder檔案的候選，尚未證明b7798實際載入及逐token數值正確。** 不需要把它的mmproj硬接進目前兩個ONNX encoder。[L1][L2][L3]

模型卡及header聲稱source為原始Qwen模型、`convert_type=hf`、static quants，量化時間2026-04-23；**未公開足以將此Q6 SHA綁到精確原checkpoint revision及quantizer SHA的build log，也沒有明載其輸入究竟F16或Q8**。因此其來源判定是「發佈方宣稱原模型static quant；不是已證實Q8 requantize；也不能嚴格證實從原浮點一次量化」。需要品質歸因時，自己從固定原浮點decoder轉Q6才最乾淨。[D2]

強量化同repo另有Q4_K_M、Q4_K_S、Q3、Q2及IQ4_XS；其模型卡用語不是中文ASR品質實測。本輪不把「very good quality」或通用LLM量化圖當ASR證據；首輪以有資料的handy Q4_K_M及現有Q5為對照。

### 4.2 ggml-org現成Q8也是decoder-only候選

[Qwen3-ASR-1.7B-Q8_0.gguf](https://huggingface.co/ggml-org/Qwen3-ASR-1.7B-GGUF/resolve/36a678687ba7d07a74ca70ccb0e36902e005fb80/Qwen3-ASR-1.7B-Q8_0.gguf)：2,165,034,944 bytes；SHA-256 `58e22d0532d4eacaf034cfac17a6fed159f37c41390c710186783be439d1fc57`。[D3]

本輪讀16MiB header：同樣`qwen3vl`、311 tensors、2048／28層／16Q8KV／head_dim128／65536context；`token_embd.weight`及`output.weight`皆[2048,151936] Q8_0(type8)。這比「檔名含Qwen3-ASR」強得多，足以排入現有b7798 adapter的直接smoke test。其原生llama.cpp整套才需要Q8 mmproj；對既有ONNX encoder的decoder-only測試不必下載mmproj。

**不應宣稱直接替換已通過**：本研究沒有拿目前正式Q5的完整metadata/tokenizer逐項diff，也沒有跑b7798。必驗token ID映射、mrope sections、output/tied-embedding權重處理、原prompt語義、空白與重複輸出；只有尺寸相同不是品質保證。

### 4.3 三個模型格式已實讀，確定不通用

- **handy Q6**：`general.architecture=qwen3_asr`；709 tensors；`enc.*`、`dec.token_embd.weight`，有`stt.qwen3_asr.*`與frontend buffers。encoder／decoder在同檔，沒有現有adapter要求的頂層`token_embd.weight`。
- **cstr Q8**：`general.architecture=qwen3asr`；710 tensors；`audio.*`、`llm.*`，`qwen3asr.*` metadata。
- **ggml-org Q8／mrad Q6**：`general.architecture=qwen3vl`；311 tensors；`token_embd.weight`、`output.weight`、`blk.*`，encoder在另一個mmproj。

前兩者不是換檔名或override `general.architecture`就能給llama載入；tensor名稱、metadata、frontend與圖契約均不同。後兩者有直接adapter候選證據，但仍保留上述驗證界線。

## 5. ggml-org原生llama.cpp audio mmproj路線

Runtime pin：`0ec4027a2a0183de87c007069adddc06e665f7de`。`conversion/qwen3vl.py`把Qwen3ASR text映射成QWEN3VL，audio映射到QWEN3A；runtime有`tools/mtmd/models/qwen3a.cpp`。[G1][G2]

```sh
rtk proxy curl -fL https://codeload.github.com/ggml-org/llama.cpp/tar.gz/0ec4027a2a0183de87c007069adddc06e665f7de -o /work/llama-native.tar.gz
rtk proxy tar -xzf /work/llama-native.tar.gz -C /work
rtk proxy cmake -S /work/llama.cpp-0ec4027a2a0183de87c007069adddc06e665f7de -B /work/llama-native-build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_BUILD_TYPE=Release
rtk proxy cmake --build /work/llama-native-build --target llama-server llama-mtmd-cli llama-quantize -j4
rtk proxy /work/llama-native-build/bin/llama-server \
  -m /work/models/Qwen3-ASR-1.7B.Q6_K.gguf \
  --mmproj /work/models/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf \
  -ngl 99 --mmproj-offload -c 4096 --temp 0 \
  --host 127.0.0.1 --port 18081
```

這是完整新runtime的隔離server範例，不是叫parent替換b7798的動態庫。`-ngl`控制decoder offload，`--mmproj-offload`另控制encoder/projector；啟動log需確認兩者都實際使用CUDA。context固定4096、單請求、temperature0。[G3]

常駐API用`POST /v1/chat/completions`，user content包含`{"type":"input_audio","input_audio":{"data":"BASE64_WAV","format":"wav"}}`；server docs明載input_audio接受miniaudio支援的WAV/MP3/FLAC，format由內容判斷。為同條件測試統一16k mono WAV，request固定temperature0及max_tokens512。**此pin的server README沒有提供`/v1/audio/transcriptions` endpoint契約，不可自行假定multipart Whisper API可用**。[G3]

原adapter固定`language Chinese<asr_text>`；原生server ASR template／assistant prefix能否精確達到相同條件須先驗，不能僅在user加「請用中文」便當同一language control。輸出可能含`language ...<asr_text>`需保留raw及清理後文字，避免計分誤差。[G4]

現成Q8 mmproj：[下載](https://huggingface.co/ggml-org/Qwen3-ASR-1.7B-GGUF/resolve/36a678687ba7d07a74ca70ccb0e36902e005fb80/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf)，355,709,344 bytes，SHA-256 `46c1d533af3f354ceb37ce855dbceff7da7fa7cf1e6a523df3b13440bd164c0d`。同revision另有BF16 decoder及BF16 mmproj，既有manifest有hash。自己Q6應以BF16 decoder輸入`llama-quantize INPUT-BF16.gguf OUTPUT-Q6.gguf Q6_K`；不要用現有Q5升位元。ggml-org該revision無現成Q6檔。[D3]

## 6. cstr／CrispASR：有價值的Q8 encoder floor備案

Runtime本輪API pin：`CrispStrobe/CrispASR@1dd0bf6fb9e7a8bf5849cb7a14605b2234e6ee93`；模型pin沿用`cstr/qwen3-asr-1.7b-GGUF@674df5d44b50a63e7102a18895ed20e3f91de301`。[C1][C2]

模型卡記錄舊Q4 audio tower造成空白／重複，重烤後audio保Q8。它的`crispasr-quantize`支援q6_k，對Qwen3-ASR `audio.*`在低於Q8時強制保Q8（除非`CRISPASR_QWEN3ASR_QUANT_AUDIO=1`明確取消保護）；因此可由F16做出**Q6 decoder + Q8 encoder**。quantizer也接受Q8輸入再requant，必須在紀錄中明示。[C3]

現成HF清單只有F16、Q8、Q5、Q4_0、Q4_K，沒有Q6；其Q4_0/Q4_K相同bytes但hash不同，不能從名字或大小猜tensor recipe。模型卡的單段JFK「完美」與CPU速度不作繁中品質或P4證據。[C2]

- [Q8](https://huggingface.co/cstr/qwen3-asr-1.7b-GGUF/resolve/674df5d44b50a63e7102a18895ed20e3f91de301/qwen3-asr-1.7b-q8_0.gguf)：2,506,723,200 bytes；SHA `9851ab996591a2d0cb0efb216002764b509c86bd40c95e613d7b65b8e69c8a6e`。
- [Q4_K](https://huggingface.co/cstr/qwen3-asr-1.7b-GGUF/resolve/674df5d44b50a63e7102a18895ed20e3f91de301/qwen3-asr-1.7b-q4_k.gguf)：1,490,915,200 bytes；SHA `ec197cef7ccc589fdcae1becc3f4a3de119d0a41e790b898b519b1a048dad8d4`。
- [F16](https://huggingface.co/cstr/qwen3-asr-1.7b-GGUF/resolve/674df5d44b50a63e7102a18895ed20e3f91de301/qwen3-asr-1.7b-f16.gguf)：4,704,800,576 bytes；SHA `ac6c58f9cb0f2bb22290319cde01b0e25a2c1f1ee69aa2f1a9265f29a8bfc9d7`。

CrispASR根CMake有GGML_CUDA路徑，但本輪沒有完成此pin之SM61完整build/runtime契約核對；不把它標為比transcribe更低風險的首測選擇。

## 7. 驗證邊界與搜尋紀錄

本輪Tavily advanced查詢聚焦transcribe.cpp CUDA/language/benchmark、Qwen3-ASR Q6_K/Q8 decoder、CrispASR CUDA/Q6與llama.cpp Q6，然後追到作者HF卡、HF API及pinned source。記錄到的request IDs：`85fce5ac-99d7-4a1f-82bb-7150ace6845f`、`74f4b4b4-b14f-4e48-be0a-3b9882291302`、`04b7b81e-bb0f-4ff1-8bef-80929fae1064`。HF discovery一次limit100，新增decoder Q6由直接HF API核對，並非窮盡全Hub。

header檢查使用HTTP Range，伺服器回206，檔案總長與API相符；header完整解析到全部tensor descriptors。這驗證metadata/shape/型別，**不驗證tensor payload、原浮點來源、量化品質或GPU execution**。對已發佈模型的量化來源缺口已逐項保留，沒有把Q5 upquant寫成Q6品質恢復。

parent實測最少保存：完整模型SHA、runtime SHA/build flags、實際CUDA backend、語言/prompt/context／max output、warmup與逐次wall/stage timing、raw文字、CER/WER及空白/重複/截斷狀態。其他硬體的作者速度沒有被寫成P4成績。

## 一手來源索引

[T1]: https://github.com/handy-computer/transcribe.cpp/blob/5700f127e38dd53c4735ea23ef743b4b538eacd4/CMakeLists.txt
[T2]: https://github.com/handy-computer/transcribe.cpp/blob/5700f127e38dd53c4735ea23ef743b4b538eacd4/ggml/UPSTREAM
[T3]: https://github.com/handy-computer/transcribe.cpp/blob/5700f127e38dd53c4735ea23ef743b4b538eacd4/tools/CMakeLists.txt
[T4]: https://github.com/handy-computer/transcribe.cpp/blob/5700f127e38dd53c4735ea23ef743b4b538eacd4/ggml/src/ggml-cuda/CMakeLists.txt
[T5]: https://github.com/handy-computer/transcribe.cpp/blob/5700f127e38dd53c4735ea23ef743b4b538eacd4/examples/cli/main.cpp
[T6]: https://github.com/handy-computer/transcribe.cpp/blob/5700f127e38dd53c4735ea23ef743b4b538eacd4/examples/cli/asr.cpp
[T7]: https://github.com/handy-computer/transcribe.cpp/blob/5700f127e38dd53c4735ea23ef743b4b538eacd4/tools/transcribe-bench/main.cpp
[T8]: https://github.com/handy-computer/transcribe.cpp/blob/5700f127e38dd53c4735ea23ef743b4b538eacd4/include/transcribe.h
[T9]: https://github.com/handy-computer/transcribe.cpp/blob/5700f127e38dd53c4735ea23ef743b4b538eacd4/docs/models/qwen3-asr-1.7b.md
[T10]: https://github.com/handy-computer/transcribe.cpp/blob/5700f127e38dd53c4735ea23ef743b4b538eacd4/src/arch/qwen3_asr/model.cpp
[T11]: https://github.com/handy-computer/transcribe.cpp/tree/5700f127e38dd53c4735ea23ef743b4b538eacd4/src/arch/qwen3_asr
[T12]: https://huggingface.co/handy-computer/Qwen3-ASR-1.7B-gguf/blob/0e0a596e3959c3f3f2ab7d443a5b6f0f4a4bba6f/README.md
[T13]: https://github.com/handy-computer/transcribe.cpp/blob/5700f127e38dd53c4735ea23ef743b4b538eacd4/docs/tools/quantization.md
[T14]: https://github.com/handy-computer/transcribe.cpp/blob/5700f127e38dd53c4735ea23ef743b4b538eacd4/scripts/quantize-all.py
[T15]: https://huggingface.co/api/models/handy-computer/Qwen3-ASR-1.7B-gguf/revision/0e0a596e3959c3f3f2ab7d443a5b6f0f4a4bba6f?blobs=true
[D1]: https://huggingface.co/api/models/mradermacher/Qwen3-ASR-1.7B-GGUF/revision/664b960e358348943aefcdc26b3cf09eff1fb0ac?blobs=true
[D2]: https://huggingface.co/mradermacher/Qwen3-ASR-1.7B-GGUF/blob/664b960e358348943aefcdc26b3cf09eff1fb0ac/README.md
[D3]: https://huggingface.co/api/models/ggml-org/Qwen3-ASR-1.7B-GGUF/revision/36a678687ba7d07a74ca70ccb0e36902e005fb80?blobs=true
[L1]: ../../core/server/engines/qwen_asr_gguf/inference/asr.py
[L2]: ../../core/server/engines/qwen_asr_gguf/inference/llama.py
[L3]: ../../core/server/engines/qwen_asr_gguf/export/gguf/quants.py
[G1]: https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/conversion/qwen3vl.py
[G2]: https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/tools/mtmd/models/qwen3a.cpp
[G3]: https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/tools/server/README.md
[G4]: https://github.com/ggml-org/llama.cpp/issues/26749
[C1]: https://github.com/CrispStrobe/CrispASR/tree/1dd0bf6fb9e7a8bf5849cb7a14605b2234e6ee93
[C2]: https://huggingface.co/cstr/qwen3-asr-1.7b-GGUF/blob/674df5d44b50a63e7102a18895ed20e3f91de301/README.md
[C3]: https://github.com/CrispStrobe/CrispASR/blob/1dd0bf6fb9e7a8bf5849cb7a14605b2234e6ee93/examples/crispasr-quantize/main.cpp
