# Tesla P4：ASR 硬體、量化 kernel 與版本邊界證據

研究日期：2026-11-01（Asia/Taipei）。本機閱讀基準：`95912cf66f384aa752f3dda291e02c893070beff`。
本報告只做背景研究，未編譯／執行 GPU 程式、安裝 GPU stack、下載大型模型、讀取正式容器或量測使用者的 P4。
本篇為 Issue #14 主研究的硬體／kernel 附錄；相關研究文件與追蹤由主研究統一整合。

先閱讀既有 [P4 runtime 研究](tesla-p4-qwen3-asr-runtime-20261031.md) 與
[P4／RTX2080／RTX2060 比較](p4-rtx2080-rtx2060-qwen3-asr-20261101.md)。
本篇補足它們未展開的 kernel 選擇、FP16 計算型別及套件版本邊界，不重複服務層功能盤點。
關鍵主張均重新查官方文件／固定 commit 原始碼，不能把舊報告本身當硬體證據。

本檔是 [P4 ASR 模型與量化主報告](tesla-p4-asr-model-quantization-20261101.md) 的硬體／kernel 附錄。
模型 artifact 與完整 ASR 整合由主報告處理；主研究已核對 current 的
`tools/mtmd/models/qwen3a.cpp` 及官方 Q8 text／Q8 mmproj，本文的「整合待驗」不表示上游沒有 Qwen3-ASR 實作。

## 1. 可直接交給主研究的結論

1. **P4 是 GP104、SM61，確有高速 INT8 DP4A，沒有 Tensor Cores。**
   5.5 TFLOPS FP32、22 INT8 TOPS 是規格峰值；GP104 的原生 half2 算術吞吐約為 FP32 的 1/64。
   這是算術指令的吞吐差異，**不是 FP16 encoder 必慢 64 倍**。[H1] [H2] [H3] [H4] [H5]
2. **GGUF Q4_K／Q5_K 不是「直接把 P4 當 INT4／INT5 Tensor Core 使用」。**
   llama.cpp CUDA 會 unpack／重建低位元權重、量化 activation 成 Q8_1、做 DP4A，
   再做 scale／min 修正；另有 MMVQ、MMQ、浮點反量化＋cuBLAS 等路徑。
   因此 INT8 TOPS 不等於 GGUF 模型的實際吞吐，也不能推出 Q8_0 一定比 Q5_K 快。[L8] [L9] [L10] [L11] [L12] [L13] [L14] [L15] [L16]
3. **b7798 與查詢當下 current 都保留 SM61 CUDA 原始碼支援。**
   在一般 dense、F32 activation/output、沒有不安全 padding/view 的量化矩陣乘法，
   SM61 上這四種格式，小 batch（欄數 `ne11 <= 8`）優先 MMVQ，較大時可走 MMQ。
   SM61 沒有 FP16 MMA，預設 MMQ predicate 本來就回 true；
   **`GGML_CUDA_FORCE_MMQ` 可能完全不改路徑，不能宣稱必然加速**。[L1] [L2] [L3] [L4] [L5] [L6] [L7] [L8] [L9] [L17] [L18] [L19] [L20] [L21] [L22]
4. **Vulkan 也有整數 dot-product 路線，不能描述成只能 FP16 或 CPU。**
   但必須是該 driver 回報 accelerated packed signed int-dot，且 binary/shader 編譯包含該功能。
   Q4_K／Q5_K／Q8_0 可見 scalar 整數 pipeline；IQ4_NL 在所查版本沒有同等的 Pascal scalar
   Q8_1 pipeline，應保留浮點反量化路線的判斷，不把其他架構的 cooperative-matrix 註冊套到 P4。
   本機未驗證正式 P4 的 Vulkan feature bits。[V1] [V2] [V3] [V4] [V5] [V6] [V7] [V8] [V9] [V10]
5. **ORT 1.18.0 不是「half 檔就全程 half 算術」。**
   cuDNN Conv 對 half input 設 FP32 compute type；一般 half GEMM 預設亦採 FP32 compute，
   但 buffers/output 可仍為 FP16。其他 elementwise／fusion 路徑需各自查證。
   FP32 vs FP16 encoder 應做同模型實驗，不能單靠 1/64 規格宣布勝負。[O1] [O2] [O3] [O4] [O5]
6. **工具鏈、wheel 所含架構、kernel 門檻、模型格式支援是不同邊界。**
   CUDA 12.9 還能編譯 Pascal，不代表 PyTorch cu128 或 bitsandbytes cu128 wheel 含 Pascal。
   CUDA 13.0 移除 Pascal 離線編譯及函式庫支援；舊 CUDA 應用的可執行性則另受 driver 與其攜帶庫控制。[C1] [C2] [P1] [P2] [P3] [B1]
7. **沒有找到可重現且符合「P4＋量化 ASR」的一手延遲／品質 A/B。**
   找到的一手 P4 llama.cpp tile 實驗是 BGE encoder／文字 decoder，不是 ASR；
   上游 faster-whisper GPU benchmark 是 RTX3070Ti。不能補造 P4 ASR 毫秒數或加速倍率。[E1] [E2]

### 本報告的支援等級

- **已驗證支援（文件／原始碼）**：官方矩陣或固定 commit 明列該架構／kernel。這不表示本研究在 P4 成功編譯或試跑。
- **僅原理可行**：硬體／計算方式允許，但缺少該版本完整 runtime、模型轉換、所有算子與部署相容性的證據。
- **未測效能**：上述兩類均可能未測；本文所有本專案的優化建議皆沒有 P4 ASR 實測收益。
- **已確認排除／證據不足**：官方門檻高於 SM61，或來源互相矛盾／不足。不能寫成已驗證可用。

## 2. 硬體能力：FP32、FP16、DP4A 分別代表什麼

| 項目 | 可核實事實 | 對本任務的限制 |
| --- | --- | --- |
| 晶片／CC | NVIDIA 文章明列 Tesla P4 為 Pascal GP104；legacy 表列 CC6.1 | P100 是 SM60、不同 FP16 設計，T4 是 Turing；都不能直接類推 |
| FP32 | datasheet 5.5 TFLOPS；發表表格在1063MHz boost下列5.442TFLOPS | 峰值、boost、散熱／功率與持續工作負載不同 |
| FP16 | 有 half 儲存及原生 half/half2 指令；GP104 half2 峰值為 FP32 的1/64 | 可省權重／activation帶寬及容量；庫可能FP32計算，不能全程套1/64 |
| INT8 | DP4A 一次處理4組8-bit乘積並累加到32-bit；21.8（四捨五入22）TOPS | 是SIMT整數點積，不是Tensor Core；要滿足資料布局、packing及kernel利用率 |
| 記憶體 | 8GB GDDR5、192GB/s | 低batch decode可能受帶寬影響；整體還包含encoder、KV、workspace與資料搬移 |
| Tensor／BF16／TF32 | P4沒有FP16/INT8 Tensor Cores，TRT8.6.1矩陣也列TF32 No | 不可用新GPU的BF16、TF32、FP8或Marlin tensor-core成績預測 |

來源：[H1] [H2] [H3] [H4] [H5]。用 5.5/64 得約 **0.086 TFLOPS 原生 half2 峰值**僅是按規格推算，
不是 cuDNN/cuBLAS encoder benchmark。DP4A 每條指令有4個乘加，
在「乘與加各算一 operation」口徑下，可解釋 INT8 TOPS 約為 FP32 FLOPS 四倍，
而不是 INT8 網路端到端自動快四倍。[H1] [H2] [H4]

## 3. INT8 TOPS 為何不能直接換成 GGUF Q4／Q5 效能

GGUF 是容器；量化 tensor type 才決定 block layout。一個檔案也可混用量化與F16/F32 tensor。
`Q4_K_M`／`Q5_K_M` 這類模型配方名稱，不表示每個 tensor 都是4或5-bit純整數，
更不表示 encoder ONNX、attention、normalization 都會變成 INT8。[L13] [L14] [L15]

| tensor type | 儲存與CUDA整數計算工作 | 能證實及不能證實的事 |
| --- | --- | --- |
| Q4_K | 256值super-block；6-bit scale/min metadata；有效4.5bpw；拆nibble後與Q8_1做DP4A並修正min項 | 已有MMQ/MMVQ；不是原生INT4矩陣指令 |
| Q5_K | 在Q4_K布局外加高bit；有效5.5bpw；unpack與scale/min修正 | 已有MMQ/MMVQ；多一bit不等於延遲按5/4比例增加 |
| Q8_0 | 32個int8加half scale；有效8.5bpw；與Q8_1點積 | unpack較簡單，但讀取權重更多；P4上未證實比Q5_K快 |
| IQ4_NL | 32個4-bit code加half scale，4.5bpw；查非線性16值表重建int8，再DP4A | CUDA確有DP4A；「非線性」不代表只能浮點GEMM，但查表有成本 |

來源：[L10] [L11] [L12] [L13] [L14] [L15]。CUDA MMVQ的 activation 為Q8_1，MMQ有其專用Q8_1布局；
block scale等中介資料／後處理仍會用浮點。低bit儲存並不是全圖W4A4或W5A5。
cuBLAS fallback甚至可能先反量化到F32。可獲得多少收益，取決於帶寬、unpack、查表、
activation量化、tile/shared-memory利用率、launch overhead與矩陣大小，而非只看TOPS。[L5] [L16]

**推論、未測效能：**短句單人ASR通常同時有encoder、decoder prefill及逐token decode；
即使某一段kernel變快，固定前處理／搬移／排程／最後輸出時間仍存在。
因此應分開量測encoder、prefill、token generation與放開後到最終文字的p50/p95。
量化格式改變還需中文／混語CER/WER、熱詞／數字、漏字與幻覺回歸，不能聲稱天然無損。

## 4. llama.cpp b7798 與 current：SM61 CUDA 選路

### 4.1 固定比較基準

- 本 repo downloader pin：b7798 CPU與Vulkan資產，未列CUDA資產。[R1]
- b7798解析為 **`c301172f660a1fe0b42023da990bf7385d69adb4`**（2026-01-21）。
- 本研究的current指 **`0ec4027a2a0183de87c007069adddc06e665f7de`**（2026-10-31），不是未固定的「永遠最新」。
- 兩者CMake都把61註解為DP4A架構；非native預設只在CUDA toolkit<13時加入61-virtual。
  `61-virtual`是PTX，由driver首次JIT；`61`可產生該架構device code＋PTX。
  建置環境的native偵測也影響輸出，不能把來源支援當成任意預編譯包已含SM61。[L1] [L2]

### 4.2 一般 dense MUL_MAT 的判斷（不是全模型所有 op）

前提：權重在CUDA、可用的SM61 device code／PTX、正常buffer layout，
activation及output為F32、非多GPU split特例、沒有迫使fallback的不安全view/padding。

| 路徑 | b7798 | current | 四種格式對P4的結論 |
| --- | --- | --- | --- |
| MMVQ，量化matrix-vector／小batch | `ne11<=8`，在MMQ前分派 | 仍預設<=8；SM61不在Ada／Blackwell／Volta等調整門檻的特例 | Q4_K/Q5_K/Q8_0/IQ4_NL均有vecdot；單token通常到此 |
| MMQ，量化matrix-matrix | type支援＋compiled arch>=610；NVIDIA條件為「沒有FP16 MMA或batch<64」 | 相同主邏輯，另檢查每block可用shared memory>=48KiB，使用分開的Pascal DP4A config | SM61沒有FP16 MMA，因此一般較大batch預設就符合，並非必須FORCE_MMQ |
| cuBLAS fallback | 不符前述dtype/layout/type等條件；P4非fast FP16時，量化權重轉F32後SGEMM | dispatch重構；自動compute type把P4量化/F16轉F32，並提供新的compute-type env override | 不等於「GGUF是FP16 GEMM」；fallback也不等於整模型CPU |
| 非量化attention／其他op | 有MMVF/MMF及batched cuBLAS等路徑 | 又有FWHT／fusion與更多heuristics | 不可把量化linear的選路推廣到全部attention及elementwise |

來源：[L3] [L4] [L5] [L6] [L7] [L8] [L9] [L16] [L17] [L18] [L19] [L20] [L21] [L22] [L23]。此處batch是單次GGML算子的矩陣欄數，
不是API同時使用者數或音訊秒數；某次prefill也可能被切成不同micro-batch。

**重要修正：**「Pascal要強制MMQ才能啟用DP4A」不成立。MMVQ本身也呼叫DP4A；
SM61的預設MMQ條件又本來成立。強制MMQ是**編譯選項**，只改其predicate，
沒有把先行MMVQ分派移到後面，也不會為不支援型別或不安全layout創造kernel。
`FORCE_CUBLAS`在MMQ predicate前置return false，也不等同關掉全部MMVQ。
因此兩個force build若要比較，先核對實際分派；沒有路徑差異便不是有效的kernel A/B。[L4] [L5] [L6] [L7] [L8] [L18] [L19]

current的Pascal config對Q8_0、Q4_K、Q5_K、IQ4_NL都有明確tile entries；
Q8_0的I仍為64。這證明current還有針對SM61的路徑，
**不證明current在本repo ABI、Qwen split模型、取消／串流介面及全部算子上直接可換**。[L20] [L21] [L22]

### 4.3 P4 的一手 MMQ 實驗，為何不能擴大為ASR結論

上游issue #30235的實驗作者記錄一張P4、Windows／driver551.78／CUDA12.6，
llama.cpp build10793 `d230ddd76`，SM61＋FORCE_MMQ，交錯A/B、每模型8對，
只把Q8_0的I從64改128：

- BGE reranker與BGE embedding的 `pp512` 約+12.1%／+11.6%。
- 對所有quant type一律改tile，Q4_K_M與Q6_K反而大幅退步。
- 作者明列未測Q5_K、IQ系列、其他GPU、prompt/batch sweep、Q8 decoder；
  更新時只確認patch仍可套在b11476 `9881906`。[E1]

這是**實驗作者一手、範圍有限、未由本研究重現**的P4量化矩陣乘法證據，
不是上游已合併的通用修正，不是Qwen3-ASR或Whisper測試。
current原始碼仍可看到I=64，[L21] 與其觀察一致。
只能列為後續獨立microbenchmark候選，不應移植patch後就承諾ASR加速12%；
其thermal drift觀察也提醒A/B需交錯與充分散熱，並非要求本次操作GPU。

## 5. Vulkan：可用量化不代表選同一個CUDA kernel

b7798先檢查extension／feature與
`integerDotProduct4x8BitPackedSignedAccelerated`，最後還核對
`shaderIntegerDotProduct`。如果shader編譯器建置時沒有
`GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT`，原始碼有分支也不代表binary含該shader。
啟動偵測列可列出fp16、int dot、matrix cores；**本研究未讀取正式P4的該列**。[V1] [V2] [V3]

| 格式／情境 | b7798可見路徑 | current可見路徑 | 支援等級 |
| --- | --- | --- | --- |
| Q4_K／Q5_K／Q8_0矩陣 | 有Q8_1 activation的scalar int-dot MMQ；不可用時dequant＋floating shader | scalar/subgroup int-dot MMQ仍在，依可用shader選activation型別 | 原始碼已驗證；P4實際driver/binary及效能未測 |
| Q4_K／Q5_K／Q8_0小batch | int-dot／floating vector pipeline按條件選 | MMVQ再按batch與k選；NVIDIA k<=4096的一般情況會避免量化，Q8_0大k有pre-Turing特例 | 不能只因硬體有DP4A就斷言每token走int-dot |
| IQ4_NL | 有一般量化floating pipeline；scalar Q8_1 MMQ列表未列此型別 | 一般floating保留；IQ4_NL Q8_1註冊可見於coopmat1，scalar MMQ及Q8_1 MMVQ列表未列 | 不把coopmat1當成P4 Tensor能力；P4應預期floating fallback候選，仍需實際trace |
| F16／F32 accumulation | 看device fp16、precision及coopmat能力選f16acc/f32acc | capability與精度選擇持續演進 | fp16儲存、fp16算術與f16acc是不同條件 |

來源：[V3] [V4] [V5] [V6] [V7] [V8] [V9] [V10]。P4沒有Tensor Cores不能推出Vulkan不可用；
同樣地，Vulkan支援int-dot也不能推定driver一定把指定shader降成理想DP4A SASS。
Vulkan與CUDA的比較要同GGUF、同offload、同batch／KV／context、同暖機條件，
並確認實際裝置與fallback。**「CUDA一定比Vulkan快」及「Vulkan必定拖慢P4」均無本研究證據。**

## 6. ORT 1.18／cuDNN：FP32 vs FP16 encoder 的正確問題

本repo CUDA基底為11.8/cuDNN8，ORT pin為1.18.0；encoder建立兩個session，
從frontend ONNX input type選numpy float16或float32，沒有依P4自動把模型改成FP32。
只換numpy dtype也不能把ONNX graph的權重／內部節點精度完整轉換。[R2] [R3] [R4] [R5]

**已驗證的原始碼行為：**

- ORT1.18.0的cuDNN Conv descriptor：half input對應
  `CUDNN_DATA_HALF` tensor，但compute type改成 `CUDNN_DATA_FLOAT`。[O1]
- Conv先設math mode，再進行EXHAUSTIVE／HEURISTIC／DEFAULT的algorithm search，
  受workspace與cache影響。設 `CUDNN_TENSOR_OP_MATH` 是容許庫選擇，
  不會讓沒有Tensor Cores的P4憑空執行Tensor指令。[O1] [O2] [H5]
- 一般MatMul走cuBLAS helper；`HalfGemmOptions`預設
  `compute_16f_=false`，half buffers也能選FP32 compute。[O3] [O4] [O5]
- cuDNN8.9.2官方矩陣列CC6.1/Pascal與CUDA11/12；這是版本族支援證據，
  未確認實際容器中cuDNN8的patch版與ONNX graph的所有算子。[O7]
- ORT1.18.x的CUDA11.8版配cuDNN8。CUDA12的1.18.0也配cuDNN8，
  **1.18.1的CUDA12版改要求cuDNN9**；不能只寫「ORT1.18配某一個cuDNN」。
  cuDNN8與9不可互換major ABI。[O6]

**僅原理可行、未測效能：**FP16 artifact可減少帶寬／容量，
FP32 artifact可能避免部分slow native half與cast成本；但庫已採mixed precision的op、
圖fusion、反覆session.run、CPU↔GPU搬移、短shape的launch overhead與workspace都會改變勝負。
不能因1/64直接要求全部FP32，也不能因「半精度省一半」保證FP16較快。
P4也沒有TF32／BF16 Tensor Core路線，套用Ampere的FP16/TF32設定成績無效。[H4] [H5] [O1] [O2] [O3] [O4] [O5] [O6] [O7]

建議A/B（本次未執行）：先保留同一1.7B checkpoint與decoder Q5_K，只比較完整且有效的
FP32／FP16 encoder exports；記錄兩個session輸入／內部精度、CUDA節點placement、
CPU fallback、每階段warm p50/p95、首次shape搜尋、VRAM及文字品質。
再獨立比較I/O Binding等搬移改善，避免同時更換精度／runtime／量化導致無法歸因。

**INT8 算子範圍補註（主研究已核對，詳見主報告）：**ORT v1.18.0 的 CUDA
OperatorKernels 表有 S8×S8 `MatMulInteger`，不能泛稱所有 `MatMulInteger` 都沒有 CUDA kernel。
`DynamicQuantizeLinear` 的 U8 路線不在該 CUDA 表，圖是否能完整留在 GPU 需逐節點判斷。
另不能把新版 `MatMulNBits(bits=8)` 的能力套回 1.18 的 4-bit 實作；本篇的 half GEMM／Conv 證據
不構成 INT8 encoder 全圖支援或加速證據。此補註不新增本次已做硬體實測的範圍。

## 7. Pascal 軟體版本邊界

下列是**官方文件／來源宣告的支援**或**排除**；沒有在本專案P4完成安裝驗證。
「支援量化方法」不等於有Qwen3-ASR1.7B完整轉換與服務adapter。

| 元件／版本 | Pascal／P4邊界 | 本任務判斷 |
| --- | --- | --- |
| CUDA 11.8、12.x至12.9 | 仍有Pascal編譯路線；llama的61建置可選 | 已驗證來源支援，評測映像候選；不推論所有12.x wheels含SM61 |
| CUDA13.0起 | 移除Maxwell/Pascal/Volta離線編譯及庫支援 | 新工具鏈不能因加 `-arch=sm_61` 就恢復；保留相容舊工具鏈／runtime |
| NVIDIA driver | 工具鏈移除不等於舊binary立即失效；官方指向R580作舊架構維護分支 | 應核對driver分支、舊CUDA庫與binary實際架構；不按 `nvidia-smi` 顯示的CUDA上限選wheel |
| PyTorch2.8起 cu128/cu129 builds | 發行政策刪除Maxwell/Pascal的預編譯架構；cu126仍保留 | 不可寫「PyTorch2.8整體不支援Pascal」；需指定CUDA build |
| PyTorch2.14／2.15邊界 | 2026-09-04官方公告：2.14最後發布cu126 wheels；2.15從CD移除cu126，僅保留一個source-build CI配置 | 不能再把2025「cu126仍持續供應」當成無期限承諾；P4可評估舊wheel或CUDA12來源建置，效能與擴充未測 |
| bitsandbytes0.48.0 | 移除Maxwell；加入CUDA13但該build只Turing+；CUDA11.8–12.6仍有SM60路線 | Pascal未被全數移除，需匹配wheel與PyTorch |
| bitsandbytes current／0.50.2文件 | NF4/FP4、8-bit optimizers表列CC6.0+；Linux x86-64 11.8–12.6列sm60，cu128從sm70起，cu13從sm75起 | NF4/FP4屬文件已列支援；SM60 cubin對SM61屬同major較高minor相容，不代表DP4A kernel已驗證；P4效能未測 |
| bitsandbytes LLM.int8() | 功能文件表列7.5+；**同commit README又把CUDA SM60+與LLM.int8都打勾**，來源不一致 | 不列為已驗證P4 INT8加速；current程式有cuBLASLt及FP32 fallback，但不能據此證明P4全模型可用或用到高速DP4A |
| AutoAWQ／AutoAWQ_kernels | NVIDIA要求CC7.5+與CUDA>=11.8；AutoAWQ已deprecated，最後測試Torch2.6.0／Transformers4.51.3 | 標準kernel排除P4；AWQ作為演算法不等於所有其他實作也排除 |
| AutoGPTQ舊CUDA kernels | archived/unmaintained；setup明稱requires Pascal or higher，unsupported清單排除Maxwell以下 | 非Marlin的舊CUDA路線有原始碼支援依據；完整模型、相容Torch／extension與效能仍未驗證，不等於GGUF即插即用 |
| AutoGPTQ0.7.0的Marlin整合 | README特別要求SM80或86 | 排除P4；不可把GPTQ的一般Pascal門檻套到Marlin |
| 原始Marlin | FP16×INT4、tensor-core／async載入設計，CUDA>=11.8、CC>=8.0 | 排除P4；沒有Tensor Core版fallback供此路線直接使用 |
| vLLM0.14.0 | 官方GPU需求CC>=7.0，預編譯CUDA12.8 | P4排除；不能只換PyTorch wheel繞過vLLM kernel要求 |
| vLLM current | 官方GPU需求CC>=7.5，主要預編譯CUDA12.9 | P4排除；社群Pascal fork為另案移植，非官方支援 |
| TensorRT8.6.1 | 矩陣明列P4 FP32/FP16/INT8，但Pascal已deprecated；無Tensor Cores | 舊encoder研究候選；支援資料型別不證明整張Qwen ONNX圖可建engine或較快 |
| TensorRT9.x | 本次9.0/9.1/9.3舊文件多404或轉址，未核實各9.x版精確Pascal清單 | **保留空白**；不宣稱8.6.1是「所有產品線最後版」，也不採issue路人建議9.1作支援證明 |
| TensorRT10.0.1 | NVIDIA現行矩陣的歷史資料 `supportData['10.0.1'].computeCapabilities`，Linux／Windows由SM70起，無SM61 | 已核實排除P4；不要把現行頁標題11.4.1誤當歷史矩陣 |
| TensorRT11.4.1 current | 官方明定SM>=7.5 | 已核實排除P4；TRT-LLM等其他產品仍需其自己的支援矩陣 |
| ORT TensorRT EP | 官方配對：ORT1.17/1.16配TRT8.6；ORT1.18配TRT10.0 | 現有ORT1.18 CUDA EP可用，不代表TRT8.6可直接塞進同一ORT1.18映像 |

來源：CUDA／driver [C1] [C2]；PyTorch [P1] [P2] [P3]；bitsandbytes [B1] [B2] [B3] [B4]；
AutoAWQ [A1] [A2]；AutoGPTQ [G1] [G2] [G3] [G4]；Marlin [M1] [M2]；vLLM [W1] [W2]；TRT／ORT [H5] [T1] [T2]。

### 解讀邊界時應保留的細節

**CUDA12來源建置≠CUDA13輪子可用。**套件可以在CUDA尚支援Pascal時，為縮小binary先刪架構；
使用者的driver支援新CUDA上限也不會替wheel補回SM61。
相反地，工具鏈升級刪架構，不代表已建好的舊CUDA應用無條件當天停止執行。[C1] [C2] [P1]

**bitsandbytes矛盾需保留。**本次先看到功能文件的LLM.int8門檻7.5，
再查到current README的更寬矩陣。本文採「不列已驗證P4快路徑」，
不是宣稱所有新版本的LLM.int8在P4絕對無法執行。
其current INT8函式對未對齊shape與部分無algorithm錯誤會做FP32 fallback，
即使API成功，也不能證明達到P4 22TOPS或有延遲收益。[B1] [B2] [B3]

**AWQ／GPTQ是量化演算法／格式族，Marlin是kernel。**同一權重格式可由多種kernel執行。
AutoGPTQ舊kernel接受Pascal與其Marlin要求Ampere可同時為真。
兩者也都沒有自動提供本專案GGUF/ONNX split模型的轉換與ASR支援。[A2] [G1] [G2] [G3] [G4] [M1] [M2]

## 8. P4 量化 ASR 一手實測的搜尋結果

**截至本次搜尋，未找到符合以下條件的公開一手A/B：**
明確Tesla P4（不是P40/P100/T4）、指定ASR模型與量化、確切runtime/build、
可重現音訊／解碼條件、GPU實際執行與延遲／品質數據。
這是「未找到」，不是「證明沒有人測過」。

| 候選 | 採用或淘汰理由 |
| --- | --- |
| llama.cpp #30235 | **保留為P4量化kernel旁證**；作者確有P4 A/B與環境，但測BGE pp512／文字decoder，未測ASR、Q5_K與IQ4_NL。[E1] |
| faster-whisper README | **保留為量化ASR的一手方法參考，淘汰為P4數據**；上游清楚寫RTX3070Ti、CUDA12.4。[E2] |
| CT2 quantization表 | **保留為SM61 INT8_float32支援依據，淘汰為效能證據**；6.1表列INT8可用、FP16轉F32，但沒有P4 ASR時間；不等於Qwen3-ASR已支援。[E3] |
| whisper.cpp discussion #3752 | **淘汰為P4證據**；原作者真有Q4/Q5/Q8 ASR表，但硬體是Intel i5-460M CPU。[E4] |
| openai/whisper discussion #716 | **淘汰**；讀到的實際案例是P100／T4，不是P4，也非合格P4量化A/B |
| whisper.cpp issue #1127 | **淘汰**；CPU、Raspberry Pi或RTX4080等比較，非P4量化ASR |
| Edge-ASR 2507.07877 | **保留為ASR量化品質研究，淘汰為P4效能**；論文研究Whisper/Moonshine的PTQ與memory I/O，未提供P4延遲結果。[E5] |
| Qwen3-ASR官方repo／2601.21337 | **保留模型能力來源，淘汰P4效能推論**；未取得P4量化ASR A/B |
| FitMyLLM的P4頁／Qwen3文字量化論文 | **淘汰**；LLM fitting／文字生成或估算，非本任務ASR測試 |
| P4 datasheet的15倍／60倍推論宣傳 | **淘汰為ASR效能**；圖像網路／不同CPU或M4基準，不能套Qwen3-ASR。[H2] |

## 9. 後續建議與證據等級

| 建議 | 已驗證支援部分 | 僅原理可行／尚缺部分 | 效能狀態 |
| --- | --- | --- | --- |
| 先做同b7798、同1.7B Q5_K的CUDA SM61對現行Vulkan | 原始碼有SM61 MMVQ/MMQ；repo pin清楚 | 另建相容CUDA binary、runtime封裝、ABI／完整op與功能回歸；正式GPU feature/placement未讀 | **未測**，不保證CUDA贏 |
| 保留預設heuristic，再確認FORCE_MMQ有無實際差異 | SM61預設MMQ已true、小batch先MMVQ | 用trace／profiler核對kernel與batch；不能只憑build option命名 | **未測**；可能完全沒差 |
| FP32對FP16 encoder exports | ORT/cuDNN提供兩種dtype及mixed-precision compute | 確認同checkpoint、graph casts／fusions、CPU fallback、shape與品質 | **未測**，不預先判FP16必慢 |
| Q4_K/Q5_K/Q8_0/IQ4_NL獨立比較 | CUDA四種kernel均存在；Vulkan路線有差別 | 需取得同checkpoint且正確的ASR decoder artifact，品質與VRAM回歸 | **未測**；Q8／IQ4均無預設優勝 |
| b7798升current | current保留SM61且有分離Pascal config | ABI、GGUF模型架構、全部op與服務adapter驗證；先與後端切換分開 | **未測**，不把新版本當速度證據 |
| Q8_0 tile patch | 有作者P4 BGE測試旁證 | 本Q5_K不受該patch直接改善，ASR與其他formats未驗證 | **本任務未測**，低於同模型後端A/B優先級 |
| CT2 INT8_float32研究 | 官方表確有CC6.1 | Qwen3-ASR整體轉換／runtime模型支援未證實；Whisper替換是換模型 | **未測**，不列直接替換 |
| bitsandbytes／舊AutoGPTQ路線 | NF4及AutoGPTQ非Marlin有Pascal文件／來源依據 | 舊stack維護、完整ASR算子／格式轉換；LLM.int8還有文件矛盾 | **未測**，不優先引入新stack |
| TensorRT8.6 encoder | 歷史矩陣列P4 | ORT EP版本配對、ONNX dynamic shapes／算子、engine建置／校準／品質 | **未測**、工程成本較高 |
| 官方vLLM／AutoAWQ／Marlin／新TRT | 官方門檻已高於SM61 | 須另案移植／非標準fork | **排除現成標準安裝路線** |

本次沒有實作或執行上述實驗。主研究應先掌握同音訊／模型的warm各階段耗時，再決定工程優先序。
不以減模型、關功能、換音訊或把長檔throughput當成短句放開後延遲的改善。

## 10. 一手來源索引（精確 commit／行範圍）

程式碼URL固定到source commit，不以main branch連結取代。
無公開git來源的廠商網頁／PDF，以文件版本、章節／表格與2026-11-01查閱日期定位；
它們沒有可誠實填寫的source commit。動態TRT頁另指定版本資料鍵。
正文的 [L8] [L9] [L10] [L11] [L12] [L13] [L14] [L15] [L16] 等範圍表示本表中的來源ID集合，不是未提供的網頁引用。

| ID | 精確來源 | 定位及用途 |
| --- | --- | --- |
| H1 | [來源](https://developer.nvidia.com/blog/new-pascal-gpus-accelerate-inference-in-the-data-center/) | NVIDIA P4 發表文章，2016-09-13，P4 章節及比較表：GP104、20 SM、2560 cores、21.8 TOPS |
| H2 | [來源](https://images.nvidia.com/content/pdf/tesla/184457-Tesla-P4-Datasheet-NV-Final-Letter-Web.pdf) | NVIDIA Tesla P4 datasheet，規格表：5.5 TFLOPS、22 INT8 TOPS、8 GB、192 GB/s、75 W；PDF 無 git commit |
| H3 | [來源](https://developer.nvidia.com/cuda-legacy-gpus) | NVIDIA Legacy CUDA GPU 表，CC 6.1 列 Tesla P4；網頁無 source commit |
| H4 | [來源](https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html#fp16-arithmetic-support) | Pascal Tuning Guide §1.4.2.1–1.4.2.2：GP104 half2 吞吐及 INT8 dot product |
| H5 | [來源](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/support-matrix/index.html#hardware-precision-matrix) | TensorRT 8.6.1 表 3 Deprecated Hardware：P4 支援 FP32/FP16/INT8，沒有 FP16/INT8 Tensor Cores；表 5 平台 CC |
| R1 | [來源](https://github.com/DF-wu/CapsWriter-Offline-Container/blob/95912cf66f384aa752f3dda291e02c893070beff/docker/server/download_models.py#L414-L432) | 本 repo 固定 b7798 CPU/Vulkan artifacts；commit `95912cf66f384aa752f3dda291e02c893070beff`；docker/server/download_models.py L414–L432 |
| R2 | [來源](https://github.com/DF-wu/CapsWriter-Offline-Container/blob/95912cf66f384aa752f3dda291e02c893070beff/docker/server/Dockerfile#L1-L14) | CUDA 11.8、cuDNN 8 runtime 基底；commit `95912cf66f384aa752f3dda291e02c893070beff`；docker/server/Dockerfile L1–L14 |
| R3 | [來源](https://github.com/DF-wu/CapsWriter-Offline-Container/blob/95912cf66f384aa752f3dda291e02c893070beff/requirements/server-docker.txt#L1-L5) | ORT GPU 1.18.0 pin；commit `95912cf66f384aa752f3dda291e02c893070beff`；requirements/server-docker.txt L1–L5 |
| R4 | [來源](https://github.com/DF-wu/CapsWriter-Offline-Container/blob/95912cf66f384aa752f3dda291e02c893070beff/core/server/engines/qwen_asr_gguf/inference/encoder.py#L130-L168) | CUDA EP／兩個 ONNX session 與 input dtype 判斷；commit `95912cf66f384aa752f3dda291e02c893070beff`；core/server/engines/qwen_asr_gguf/inference/encoder.py L130–L168 |
| R5 | [來源](https://github.com/DF-wu/CapsWriter-Offline-Container/blob/95912cf66f384aa752f3dda291e02c893070beff/core/server/engines/qwen_asr_gguf/inference/encoder.py#L232-L263) | session.run 回傳及 encoder 計時路徑；commit `95912cf66f384aa752f3dda291e02c893070beff`；core/server/engines/qwen_asr_gguf/inference/encoder.py L232–L263 |
| L1 | [來源](https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-cuda/CMakeLists.txt#L8-L35) | b7798 SM61、PTX/real 與 CUDA <13 的預設架構；commit `c301172f660a1fe0b42023da990bf7385d69adb4`；ggml/src/ggml-cuda/CMakeLists.txt L8–L35 |
| L2 | [來源](https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/ggml/src/ggml-cuda/CMakeLists.txt#L21-L50) | current 保留 SM61、CUDA <13 條件；commit `0ec4027a2a0183de87c007069adddc06e665f7de`；ggml/src/ggml-cuda/CMakeLists.txt L21–L50 |
| L3 | [來源](https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-cuda/common.cuh#L224-L289) | FP16_AVAILABLE、排除 610 的 FAST_FP16、hardware predicates；commit `c301172f660a1fe0b42023da990bf7385d69adb4`；ggml/src/ggml-cuda/common.cuh L224–L289 |
| L4 | [來源](https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-cuda/common.cuh#L665-L708) | DP4A wrapper，CUDA 架構可用時 __dp4a；commit `c301172f660a1fe0b42023da990bf7385d69adb4`；ggml/src/ggml-cuda/common.cuh L665–L708 |
| L5 | [來源](https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-cuda/ggml-cuda.cu#L2182-L2264) | b7798 dense MUL_MAT dispatch，MMVQ 在 MMQ 前；commit `c301172f660a1fe0b42023da990bf7385d69adb4`；ggml/src/ggml-cuda/ggml-cuda.cu L2182–L2264 |
| L6 | [來源](https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-cuda/mmq.cu#L262-L314) | b7798 should_use_mmq 支援型別、FORCE_CUBLAS/FORCE_MMQ、SM61 分支；commit `c301172f660a1fe0b42023da990bf7385d69adb4`；ggml/src/ggml-cuda/mmq.cu L262–L314 |
| L7 | [來源](https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-cuda/mmvq.cuh#L1-L12) | b7798 MMVQ_MAX_BATCH_SIZE=8；commit `c301172f660a1fe0b42023da990bf7385d69adb4`；ggml/src/ggml-cuda/mmvq.cuh L1–L12 |
| L8 | [來源](https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-cuda/mmvq.cu#L10-L31) | b7798 四種格式的 MMVQ vecdot dispatch；commit `c301172f660a1fe0b42023da990bf7385d69adb4`；ggml/src/ggml-cuda/mmvq.cu L10–L31 |
| L9 | [來源](https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-cuda/mmq.cu#L6-L65) | b7798 四種格式的 MMQ case；commit `c301172f660a1fe0b42023da990bf7385d69adb4`；ggml/src/ggml-cuda/mmq.cu L6–L65 |
| L10 | [來源](https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-cuda/vecdotq.cuh#L461-L580) | Q4_K/Q5_K unpack、scale/min 修正及 DP4A；commit `c301172f660a1fe0b42023da990bf7385d69adb4`；ggml/src/ggml-cuda/vecdotq.cuh L461–L580 |
| L11 | [來源](https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-cuda/vecdotq.cuh#L231-L269) | Q8_0/Q8_1 dot product 與尺度還原；commit `c301172f660a1fe0b42023da990bf7385d69adb4`；ggml/src/ggml-cuda/vecdotq.cuh L231–L269 |
| L12 | [來源](https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-cuda/vecdotq.cuh#L1173-L1195) | IQ4_NL 查表到 int8 後 DP4A，浮點尺度；commit `c301172f660a1fe0b42023da990bf7385d69adb4`；ggml/src/ggml-cuda/vecdotq.cuh L1173–L1195 |
| L13 | [來源](https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-common.h#L291-L324) | Q4_K/Q5_K block 定義、4.5/5.5 bpw；commit `c301172f660a1fe0b42023da990bf7385d69adb4`；ggml/src/ggml-common.h L291–L324 |
| L14 | [來源](https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-common.h#L219-L237) | Q8_0/Q8_1 block 定義；commit `c301172f660a1fe0b42023da990bf7385d69adb4`；ggml/src/ggml-common.h L219–L237 |
| L15 | [來源](https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-common.h#L415-L420) | IQ4_NL block 定義；commit `c301172f660a1fe0b42023da990bf7385d69adb4`；ggml/src/ggml-common.h L415–L420 |
| L16 | [來源](https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-cuda/ggml-cuda.cu#L1285-L1367) | b7798 cuBLAS fallback：慢 FP16 裝置轉 FP32＋SGEMM；commit `c301172f660a1fe0b42023da990bf7385d69adb4`；ggml/src/ggml-cuda/ggml-cuda.cu L1285–L1367 |
| L17 | [來源](https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/ggml/src/ggml-cuda/ggml-cuda.cu#L2112-L2164) | current dense MUL_MAT dispatch 與 fallback；commit `0ec4027a2a0183de87c007069adddc06e665f7de`；ggml/src/ggml-cuda/ggml-cuda.cu L2112–L2164 |
| L18 | [來源](https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/ggml/src/ggml-cuda/mmq.cu#L382-L450) | current 型別、48 KiB shared-memory 門檻、SM61 MMQ predicate；commit `0ec4027a2a0183de87c007069adddc06e665f7de`；ggml/src/ggml-cuda/mmq.cu L382–L450 |
| L19 | [來源](https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/ggml/src/ggml-cuda/mmvq.cu#L318-L427) | current MMVQ heuristics：SM61 落到 <=8，非其他 GPU 的特例；commit `0ec4027a2a0183de87c007069adddc06e665f7de`；ggml/src/ggml-cuda/mmvq.cu L318–L427 |
| L20 | [來源](https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/ggml/src/ggml-cuda/mmq.cuh#L218-L299) | current MMQ config 按 compiled arch 選 Pascal DP4A；commit `0ec4027a2a0183de87c007069adddc06e665f7de`；ggml/src/ggml-cuda/mmq.cuh L218–L299 |
| L21 | [來源](https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/ggml/src/ggml-cuda/mmq-config-pascal-dp4a.cuh#L74-L134) | current Q8_0、Q4_K、Q5_K Pascal tiles；commit `0ec4027a2a0183de87c007069adddc06e665f7de`；ggml/src/ggml-cuda/mmq-config-pascal-dp4a.cuh L74–L134 |
| L22 | [來源](https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/ggml/src/ggml-cuda/mmq-config-pascal-dp4a.cuh#L234-L244) | current IQ4_NL Pascal tiles；commit `0ec4027a2a0183de87c007069adddc06e665f7de`；ggml/src/ggml-cuda/mmq-config-pascal-dp4a.cuh L234–L244 |
| L23 | [來源](https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/ggml/src/ggml-cuda/ggml-cuda.cu#L1764-L1805) | current cuBLAS compute type 自動選擇及新 env override；commit `0ec4027a2a0183de87c007069adddc06e665f7de`；ggml/src/ggml-cuda/ggml-cuda.cu L1764–L1805 |
| V1 | [來源](https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L4710-L4723) | b7798 Vulkan fp16／accelerated signed packed int-dot capability gate；commit `c301172f660a1fe0b42023da990bf7385d69adb4`；ggml/src/ggml-vulkan/ggml-vulkan.cpp L4710–L4723 |
| V2 | [來源](https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L5355-L5388) | b7798 最終 feature 判定及啟動列印 int dot/fp16/matrix cores；commit `c301172f660a1fe0b42023da990bf7385d69adb4`；ggml/src/ggml-vulkan/ggml-vulkan.cpp L5355–L5388 |
| V3 | [來源](https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L3464-L3505) | b7798 scalar F32 與整數 MMQ pipeline 註冊；無 scalar IQ4_NL Q8_1 case；commit `c301172f660a1fe0b42023da990bf7385d69adb4`；ggml/src/ggml-vulkan/ggml-vulkan.cpp L3464–L3505 |
| V4 | [來源](https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L5704-L5779) | b7798 matmul pipeline 選擇與 FP16/FP32 accumulation；commit `c301172f660a1fe0b42023da990bf7385d69adb4`；ggml/src/ggml-vulkan/ggml-vulkan.cpp L5704–L5779 |
| V5 | [來源](https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L7014-L7048) | b7798 quantize_y、先嘗試整數 MMQ，否則 dequant/floating fallback；commit `c301172f660a1fe0b42023da990bf7385d69adb4`；ggml/src/ggml-vulkan/ggml-vulkan.cpp L7014–L7048 |
| V6 | [來源](https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L2738-L2761) | current scalar/subgroup 整數 MMQ 註冊，含 Q4_K/Q5_K/Q8_0；commit `0ec4027a2a0183de87c007069adddc06e665f7de`；ggml/src/ggml-vulkan/ggml-vulkan.cpp L2738–L2761 |
| V7 | [來源](https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L2989-L3013) | current Q8_1 MMVQ 註冊；未列 IQ4_NL；commit `0ec4027a2a0183de87c007069adddc06e665f7de`；ggml/src/ggml-vulkan/ggml-vulkan.cpp L2989–L3013 |
| V8 | [來源](https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L6368-L6379) | current 依可用 shader 選 activation dtype；commit `0ec4027a2a0183de87c007069adddc06e665f7de`；ggml/src/ggml-vulkan/ggml-vulkan.cpp L6368–L6379 |
| V9 | [來源](https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L6656-L6692) | current Vulkan MMVQ batch／k heuristics 與 pre-Turing Q8_0 特例；commit `0ec4027a2a0183de87c007069adddc06e665f7de`；ggml/src/ggml-vulkan/ggml-vulkan.cpp L6656–L6692 |
| V10 | [來源](https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L2475-L2489) | current IQ4_NL 整數 cooperative-matrix-1 註冊，不能套成 P4 scalar 能力；commit `0ec4027a2a0183de87c007069adddc06e665f7de`；ggml/src/ggml-vulkan/ggml-vulkan.cpp L2475–L2489 |
| O1 | [來源](https://github.com/microsoft/onnxruntime/blob/45737400a2f3015c11f005ed7603611eaed306a6/onnxruntime/core/providers/cuda/nn/conv.cc#L489-L535) | ORT 1.18.0 half convolution 的 FP32 compute type 及 math mode；commit `45737400a2f3015c11f005ed7603611eaed306a6`；onnxruntime/core/providers/cuda/nn/conv.cc L489–L535 |
| O2 | [來源](https://github.com/microsoft/onnxruntime/blob/45737400a2f3015c11f005ed7603611eaed306a6/onnxruntime/core/providers/cuda/nn/conv.cc#L361-L425) | ORT cuDNN algo search/workspace/perf cache；commit `45737400a2f3015c11f005ed7603611eaed306a6`；onnxruntime/core/providers/cuda/nn/conv.cc L361–L425 |
| O3 | [來源](https://github.com/microsoft/onnxruntime/blob/45737400a2f3015c11f005ed7603611eaed306a6/onnxruntime/core/providers/cuda/cuda_common.h#L143-L197) | HalfGemmOptions 預設 compute_16f=false，預設 FP32；commit `45737400a2f3015c11f005ed7603611eaed306a6`；onnxruntime/core/providers/cuda/cuda_common.h L143–L197 |
| O4 | [來源](https://github.com/microsoft/onnxruntime/blob/45737400a2f3015c11f005ed7603611eaed306a6/onnxruntime/core/providers/cuda/shared_inc/fpgeneric.h#L76-L116) | Half GEMM 傳入 FP16 buffers，但按 options 選 compute type；commit `45737400a2f3015c11f005ed7603611eaed306a6`；onnxruntime/core/providers/cuda/shared_inc/fpgeneric.h L76–L116 |
| O5 | [來源](https://github.com/microsoft/onnxruntime/blob/45737400a2f3015c11f005ed7603611eaed306a6/onnxruntime/core/providers/cuda/math/matmul.cc#L146-L177) | MatMul 呼叫 cublasGemmHelper；commit `45737400a2f3015c11f005ed7603611eaed306a6`；onnxruntime/core/providers/cuda/math/matmul.cc L146–L177 |
| O6 | [來源](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements) | ORT CUDA EP Requirements：1.18.x cu118/cuDNN8；CUDA12 1.18.0 cuDNN8 vs 1.18.1 cuDNN9；Performance tuning |
| O7 | [來源](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-892/support-matrix/index.html#cudnn-cuda-hardware-versions) | cuDNN 8.9.2 表1：CC6.1/Pascal，CUDA11/12；不是已查明正式映像的確切 cuDNN patch |
| C1 | [來源](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html#deprecated-architectures) | CUDA 13.0 §2.6.2：移除 Maxwell/Pascal/Volta 離線編譯及函式庫支援 |
| C2 | [來源](https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers/) | NVIDIA CUDA 架構生命週期說明：保留12.9建置、driver580分支；無 git commit |
| P1 | [來源](https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128) | PyTorch 團隊公告，2025-07-14：2.8 起 cu128 去除 Maxwell/Pascal；搭配 issue157517 釐清 cu126 |
| P2 | [來源](https://github.com/pytorch/pytorch/issues/157517) | PyTorch 發行追蹤：cu126 保留 Maxwell/Pascal/Volta，不能泛化成 PyTorch2.8 全不支援 |
| P3 | [來源](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432) | PyTorch 團隊公告，2026-09-04：2.14 最後 cu126 prebuilt，2.15 移除 CD但保留來源CI配置 |
| B1 | [來源](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/a616886cacf86701514f0cb1d6be760f967481c0/docs/source/installation.mdx#L25-L64) | current bitsandbytes 功能門檻與各 CUDA wheel 架構表；commit `a616886cacf86701514f0cb1d6be760f967481c0`；docs/source/installation.mdx L25–L64 |
| B2 | [來源](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/a616886cacf86701514f0cb1d6be760f967481c0/README.md#L40-L68) | current README 對SM60+及LLM.int8的廣義矩陣，與B1功能表矛盾；commit `a616886cacf86701514f0cb1d6be760f967481c0`；README.md L40–L68 |
| B3 | [來源](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/a616886cacf86701514f0cb1d6be760f967481c0/bitsandbytes/backends/cuda/ops.py#L118-L165) | current INT8 cuBLASLt／shape及錯誤時 FP32 fallback；未證實P4快速DP4A；commit `a616886cacf86701514f0cb1d6be760f967481c0`；bitsandbytes/backends/cuda/ops.py L118–L165 |
| B4 | [來源](https://github.com/bitsandbytes-foundation/bitsandbytes/releases/tag/0.48.0) | 0.48.0 release 2025-09-30：移除Maxwell、加入CUDA13但僅Turing+；不等於移除Pascal所有舊CUDA builds |
| A1 | [來源](https://github.com/casper-hansen/AutoAWQ/blob/88e4c76b20755db275574e6a03c83c84ba3bece5/README.md#L1-L8) | AutoAWQ deprecated，最後測試Torch2.6.0/Transformers4.51.3；commit `88e4c76b20755db275574e6a03c83c84ba3bece5`；README.md L1–L8 |
| A2 | [來源](https://github.com/casper-hansen/AutoAWQ/blob/88e4c76b20755db275574e6a03c83c84ba3bece5/README.md#L59-L66) | AutoAWQ NVIDIA CC7.5、CUDA11.8；commit `88e4c76b20755db275574e6a03c83c84ba3bece5`；README.md L59–L66 |
| G1 | [來源](https://github.com/AutoGPTQ/AutoGPTQ/blob/9f7d37072917ab3a7545835f23e808294a542153/setup.py#L44-L48) | AutoGPTQ unsupported arch 清單未排除Pascal；commit `9f7d37072917ab3a7545835f23e808294a542153`；setup.py L44–L48 |
| G2 | [來源](https://github.com/AutoGPTQ/AutoGPTQ/blob/9f7d37072917ab3a7545835f23e808294a542153/setup.py#L100-L119) | AutoGPTQ 建置要求Pascal以上；commit `9f7d37072917ab3a7545835f23e808294a542153`；setup.py L100–L119 |
| G3 | [來源](https://github.com/AutoGPTQ/AutoGPTQ/blob/9f7d37072917ab3a7545835f23e808294a542153/README.md#L1-L20) | AutoGPTQ停止維護，0.7.0加入Marlin的歷史；commit `9f7d37072917ab3a7545835f23e808294a542153`；README.md L1–L20 |
| G4 | [來源](https://github.com/AutoGPTQ/AutoGPTQ/blob/9f7d37072917ab3a7545835f23e808294a542153/README.md#L334-L336) | AutoGPTQ特定Marlin整合僅SM80/86；commit `9f7d37072917ab3a7545835f23e808294a542153`；README.md L334–L336 |
| M1 | [來源](https://github.com/IST-DASLab/marlin/blob/1f25790bdd49fba53106164a24666dade68d7c90/README.md#L7-L28) | Marlin FP16×INT4 tensor-core kernel與async載入設計；commit `1f25790bdd49fba53106164a24666dade68d7c90`；README.md L7–L28 |
| M2 | [來源](https://github.com/IST-DASLab/marlin/blob/1f25790bdd49fba53106164a24666dade68d7c90/README.md#L60-L66) | 原始Marlin CUDA11.8、CC>=8.0；commit `1f25790bdd49fba53106164a24666dade68d7c90`；README.md L60–L66 |
| W1 | [來源](https://github.com/vllm-project/vllm/blob/b17039bccc17b94a2ea107272ad7bc93508708df/docs/getting_started/installation/gpu.cuda.inc.md#L1-L18) | vLLM0.14.0 要CC7.0、預編譯CUDA12.8；commit `b17039bccc17b94a2ea107272ad7bc93508708df`；docs/getting_started/installation/gpu.cuda.inc.md L1–L18 |
| W2 | [來源](https://github.com/vllm-project/vllm/blob/0c63ce9f0e0b5b2782f0f30cb491069dac158bb2/docs/getting_started/installation/gpu.cuda.inc.md#L1-L53) | vLLM current 要CC7.5、CUDA12.9預編譯；commit `0c63ce9f0e0b5b2782f0f30cb491069dac158bb2`；docs/getting_started/installation/gpu.cuda.inc.md L1–L53 |
| T1 | [來源](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html) | NVIDIA現行版本化互動矩陣：supportData['10.0.1'].computeCapabilities Linux/Windows從7.0起；11.4.1明定SM>=7.5 |
| T2 | [來源](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#requirements) | ORT TensorRT EP相容表：1.17/1.16配TRT8.6，1.18配TRT10.0；與CUDA EP分開 |
| E1 | [來源](https://github.com/ggml-org/llama.cpp/issues/30235) | P4實驗作者一手issue，2026-10-09／更新10-26：Q8_0 BGE pp512約+12%；不是ASR，源d230ddd76，重查9881906 |
| E2 | [來源](https://github.com/SYSTRAN/faster-whisper#benchmark) | 上游ASR benchmark實際為RTX3070Ti 8GB／CUDA12.4，不能當P4實測 |
| E3 | [來源](https://opennmt.net/CTranslate2/quantization.html#implicit-type-conversion-on-load) | CT2的CC6.1支援INT8/FP32與隱式轉型表，屬支援表而非P4效能 |
| E4 | [來源](https://github.com/ggml-org/whisper.cpp/discussions/3752) | 原作者量化Whisper實測，但CPU為Intel i5-460M，淘汰為P4證據 |
| E5 | [來源](https://arxiv.org/html/2507.07877v1) | Edge-ASR論文，量化Whisper/Moonshine品質與memory I/O研究，未提供P4量化延遲證據 |

## 11. Tavily 查詢與候選取捨紀錄

依使用者授權使用不限額的Tavily深查；本次沒有設定費用／查詢次數上限。下列為實際送出的查詢，並非聲稱搜尋能窮盡所有未索引資料。使用 `df-meta-mcp` 技能；其本機catalog記錄 `tavily-hikari` family，當前可直接呼叫的live工具為同dfmcp的 `mcp__dfmcp__searchix_tavily__search_proxy_tavily_search`／`search_proxy_tavily_extract`；所有搜尋均 `search_depth=advanced`，每查詢4–7筆。沒有採Tavily生成答案作主證，先由URL定位，再讀官方文件／raw source。

共36次Tavily搜尋、168個去重候選URL，另3批Tavily extract（6／7／2個URL）。結果ID保留所有回傳URL；「採用或定位」仍以正文及一手索引列出的已讀證據為準，未讀的泛首頁不作事實依據。

| 查詢 | 實際query（照送） | 找到的候選 | Tavily request ID |
| --- | --- | --- | --- |
| Q01 | NVIDIA Pascal GP104 FP16 1/64 DP4A P4 22 TOPS | S001、S002、S003、S004、S005、S006、S007 | `8a2117ae-c8ad-4757-a531-821fc3463c9c` |
| Q02 | Tesla P4 whisper int8 benchmark CTranslate2 | S008、S009、S010、S011、S012、S013、S014 | `3047ec25-fb06-43d9-81e9-2309f49386e7` |
| Q03 | Tesla P4 Qwen3 ASR benchmark quantization | S015、S016、S017、S018、S019、S020、S021 | `becef6a6-9a29-4901-8196-5431776d2f9f` |
| Q04 | NVIDIA CUDA 13 Pascal CUDA 12.9 offline compilation PyTorch 2.8 Pascal | S022、S023、S024、S025、S026、S027、S028 | `3eeda4ff-f0dc-4c9f-be0f-981aff0615b9` |
| Q05 | bitsandbytes Pascal NF4 FP4 LLM.int8 hardware compatibility | S029、S030、S031、S032、S033、S034、S035 | `1afc0406-1dbf-4d20-8335-51fbc68a2b23` |
| Q06 | AutoAWQ minimum compute capability Pascal AutoGPTQ Marlin compute capability | S036、S037、S038、S039、S040、S041、S042 | `bb41c8a1-c3aa-461e-8773-3d6af5bbe2ff` |
| Q07 | ONNX Runtime 1.18 CUDA 11.8 cuDNN 8 float16 Pascal | S043、S044、S045、S046、S047、S048、S049 | `a78ad360-e567-4856-8e0f-dae4a1d7890a` |
| Q08 | "Tesla P4" "whisper" "int8" benchmark | S050、S051、S014、S052、S053、S054 | `10b62f53-0ea3-44c9-bc2b-93523b1bec89` |
| Q09 | "Tesla P4" "ASR" quantization benchmark | S055、S056、S057、S058、S059、S060 | `3a2c7d1c-d1a0-48d3-9984-3c6a0496bbe4` |
| Q10 | "P4" "Qwen3-ASR" benchmark | S018、S061、S062、S063、S064、S065 | `87492816-1bb5-43fc-a911-31dc18b174e6` |
| Q11 | site.pytorch.org PyTorch 2.8 Maxwell Pascal Volta CUDA 12.6 12.8 12.9 | S022、S026、S066、S067、S068、S069 | `7ac1d7e0-fdbc-4972-b989-5a1fb384a9e1` |
| Q12 | site.docs.nvidia.com cudnn 8.9.7 Pascal half float convolution support | S070、S071、S072、S073、S074、S075 | `08a44b4f-fae3-472d-be0d-44ef79832081` |
| Q13 | site.docs.nvidia.com tensorrt 8.6.1 9.0 Pascal removed | S076、S077、S078、S079、S080、S081 | `23c201d8-8bec-4f31-8337-df9ce3756613` |
| Q14 | site.github.com ggml llama.cpp Pascal FORCE_MMQ IQ4_NL | S082、S083、S084、S085、S086、S087 | `a2dd5d34-b02e-4820-b4ff-f8f998a96c07` |
| Q15 | site.docs.nvidia.com CUDA 13.0 release notes removed offline Maxwell Pascal Volta | S088、S089、S090、S091、S092、S093 | `e77797aa-b9fe-4ea7-8b56-fe71c36e966f` |
| Q16 | site.docs.nvidia.com tensorrt 9.0 "Pascal" "removed" | S077、S079、S081、S078、S094、S095 | `6d1341a0-1ca7-4df1-a189-af195cc65762` |
| Q17 | site.docs.nvidia.com "cuDNN 8.9.7" "6.1" Pascal | S096、S097、S098、S099、S100、S072 | `bd0c7bf5-2837-4d7f-91dc-2e1bd535c573` |
| Q18 | site.github.com "P4" "whisper" "benchmark" quant | S101、S102、S103、S104、S105、S106 | `b8b7ff59-6bf4-448a-878c-4bd5bd1c7a13` |
| Q19 | site.github.com ggml-org llama.cpp issues 30235 Pascal Q8_0 | S087、S107、S108、S109、S110、S111 | `64033d07-ae93-4bc4-9bfe-ca7fffb9d68b` |
| Q20 | "TensorRT 9.0" "Pascal" | S112、S079、S113、S081、S114 | `be4e0eb0-3847-4dd6-a1f1-939b31daf582` |
| Q21 | "TensorRT 10.0" "Pascal" removed support | S076、S115、S116、S081、S117 | `f2ec109b-195c-4c42-8dbc-9ca7752100aa` |
| Q22 | "Tesla P4" "faster-whisper" benchmark | S118、S119、S120、S105、S121 | `d26f039c-b920-43af-9460-45a92f8eaca5` |
| Q23 | "Tesla P4" "whisper.cpp" benchmark | S120、S122、S123、S124、S125 | `218845f7-bdf5-49d8-9592-81b672e7e110` |
| Q24 | site.github.com ggerganov whisper.cpp "P4" CUDA | S126、S127、S128、S129、S130 | `a95d3cd5-eb35-4369-9012-c8d7bc83a97f` |
| Q25 | site.github.com pytorch builder "2.8" "cu126" "6.0" | S131、S132、S133、S134、S135 | `eed8b825-a650-4c2e-8c15-50b5bf1d2165` |
| Q26 | site.docs.nvidia.com "Pascal" "TensorRT 9.3" | S136、S137、S077、S078、S138 | `564b2f1a-e547-49ad-9f65-c0a00f1693e8` |
| Q27 | site.docs.nvidia.com "Pascal" "TensorRT 9.2" | S136、S139、S078、S138、S079 | `de997735-15ce-4045-9cea-2dbf41b4509f` |
| Q28 | site.github.com/NVIDIA/TensorRT "Pascal" "9.0" | S081、S116、S140、S114、S138 | `3080d991-2d7f-47ce-a250-a688cb3e7d8b` |
| Q29 | site.developer.nvidia.com "P4" "GP104" | S007、S141、S142、S143、S144 | `f4f5bba4-ced0-414b-a864-8b227abd6edd` |
| Q30 | "TensorRT 9.0.1" "6.1" | S145、S081、S138、S146、S147 | `a9caf297-1161-401e-a62c-0a854613d14c` |
| Q31 | "TensorRT 9.3.0" "7.0" | S114、S148、S149、S150、S151 | `6e27412b-a58b-498b-9019-5b78041505e9` |
| Q32 | "TensorRT 10.0.1" "SM 7.0" | S076、S115、S152、S112、S114 | `5f44ff98-93e4-4e32-b6fb-787c1db685e7` |
| Q33 | "TensorRT" "Support for CUDA architecture SMs version 6.x" | S078、S153、S154、S155、S156 | `27b30c65-3e70-4ef2-90d6-29c15bad21e9` |
| Q34 | site.github.com/microsoft/onnxruntime "cublasGemmHelper" "HalfGemmOptions" | S157、S158、S159、S160 | `41eea191-169c-44c6-8f13-1a20a971b01c` |
| Q35 | site.github.com/bitsandbytes-foundation/bitsandbytes/releases "0.48.0" "Pascal" | S161、S162、S163、S164 | `62005a2d-f245-4e11-8d86-93cb798dae5e` |
| Q36 | site.docs.nvidia.com/cuda/archive/12.9.1 "sm_61" | S165、S166、S167、S168 | `d4e64a0e-af50-4db3-b9ef-103a762c61b4` |

### 全部去重搜尋候選：保留與淘汰並列

| 候選 | 找到的URL／標題 | 取捨 | 理由 |
| --- | --- | --- | --- |
| S001 | [NVIDIA Tesla P4 GPU Datasheet](https://images.nvidia.com/content/pdf/tesla/184457-Tesla-P4-Datasheet-NV-Final-Letter-Web.pdf) | 採用或沿用定位 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S002 | [Data Sheet: Tesla P4 - NVIDIA](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/184457-Tesla-P4-Datasheet-NV-Final-Letter-Web.pdf) | 採用或沿用定位 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S003 | [1. Pascal Tuning Guide — Pascal Tuning Guide 13.4 documentation](https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html) | 採用或沿用定位 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S004 | [Nvidia Pascal TITAN Xp, TITAN X, GeForce GTX 1080 Ti, GTX 1080, GTX 1070, GTX 1060, GTX 1050 & GT 1030 - Page 4 - CUDA Programming and Performance - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/nvidia-pascal-titan-xp-titan-x-geforce-gtx-1080-ti-gtx-1080-gtx-1070-gtx-1060-gtx-1050-gt-1030/42660?page=4) | 保留線索 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S005 | [Pascal Tuning Guide - NVIDIA Documentation Hub](https://docs.nvidia.com/cuda/pdf/Pascal_Tuning_Guide.pdf) | 採用或沿用定位 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S006 | [Tuning CUDA Applications for Pascal - NVIDIA Documentation Hub](https://docs.nvidia.com/cuda/archive/11.4.4/pdf/Pascal_Tuning_Guide.pdf) | 採用或沿用定位 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S007 | [NVIDIA GP104 GPU ／ VideoCardz.net](https://videocardz.net/gpu/nvidia-gp104) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S008 | [Performance Benchmarks ／ OpenNMT/CTranslate2 ／ Zread](https://zread.ai/OpenNMT/CTranslate2/4-performance-benchmarks) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S009 | [Faster whisper transcription with CTranslate2](https://www.facebook.com/niyitech/posts/whisper-transcription-but-up-to-4-times-faster-faster-whisper-rebuilds-openais-w/122157662463121169) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S010 | [Whisper vs Faster-Whisper: Speed Comparison by GPU GIGAGPU](https://gigagpu.com/whisper-vs-faster-whisper-speed) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S011 | [Converting Your Fine-Tuned Whisper Model to Faster-Whisper Using CTranslate2 ／ by Balaragavesh ／ Medium](https://medium.com/@balaragavesh/converting-your-fine-tuned-whisper-model-to-faster-whisper-using-ctranslate2-b272063d3204) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S012 | [GitHub - SYSTRAN/faster-whisper: Faster Whisper transcription with CTranslate2 · GitHub](https://github.com/SYSTRAN/faster-whisper) | 採用，限方法 | ASR上游一手benchmark，但GPU是RTX3070Ti。 |
| S013 | [faster-whisper - Fast Whisper Transcription in Python ／ AI/TLDR](https://ai-tldr.dev/tools/faster-whisper) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S014 | [Nvidia's Tesla P4 And P40 GPUs Boost Deep Learning Inference Performance With INT8, TensorRT Support ／ Tom's Hardware](https://www.tomshardware.com/news/nvidia-tesla-p40-p4-inference,32680.html) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S015 | [An Empirical Study of Qwen3 Quantization](https://arxiv.org/html/2505.02214v1) | 保留論文，淘汰P4速度 | 原論文來源；未取得P4量化ASR可重現延遲A/B；不按搜尋片段當實測。 |
| S016 | [Speed Benchmark - Qwen](https://qwen.readthedocs.io/en/v3.0/getting_started/speed_benchmark.html) | 保留模型一手資料 | 非P4量化ASR硬體A/B，不用來估算P4速度。 |
| S017 | [[2511.08093v1] Quantizing Whisper-small: How design choices ...Running LLMs on NVIDIA Tesla P4: What You Can Run[2507.07877] Edge-ASR: Towards Low-Bit Quantization of ...GitHub - alazarchuk/quantization-examples: Bitsandbytes ...](https://arxiv.org/abs/2511.08093v1) | 保留論文，淘汰P4速度 | 原論文來源；未取得P4量化ASR可重現延遲A/B；不按搜尋片段當實測。 |
| S018 | [GitHub - QwenLM/Qwen3-ASR: Qwen3-ASR is an open-source series of ASR models developed by the Qwen team at Alibaba Cloud, supporting stable multilingual speech/music/song recognition, language detection and timestamp prediction. · GitHub](https://github.com/QwenLM/Qwen3-ASR) | 保留模型一手資料 | 非P4量化ASR硬體A/B，不用來估算P4速度。 |
| S019 | [Pushing the Limits of On-Device Streaming ASR:A Compact, High-Accuracy English Modelfor Low-Latency Inference](https://arxiv.org/html/2604.14493v2) | 保留論文，淘汰P4速度 | 原論文來源；未取得P4量化ASR可重現延遲A/B；不按搜尋片段當實測。 |
| S020 | [Qwen3-ASR Just ENDED Whisper in 2026 😱 52 Languages + 2000x Faster – Beats It On ALL Tests 💀](https://www.youtube.com/watch?v=KEo4c2REwXA) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S021 | [Qwen3.8-4B Distilled: Q4 vs Q6 vs Q8 Quantization Compared ／ MindStudio](https://www.mindstudio.ai/blog/qwen3-8-4b-distilled-quant-comparison) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S022 | [[release 2.8-2.9] Delete support for Maxwell, Pascal, and Volta architectures for CUDA 12.8 and 12.9 builds · Issue #157517 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/157517?timeline_page=1) | 採用 | 官方發行政策／原專案追蹤，已直接讀取重驗。 |
| S023 | [[release 2.8-2.9] Delete support for Maxwell, Pascal, and Volta architectures for CUDA 12.8 and 12.9 builds · Issue #157517 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/157517) | 採用 | 官方發行政策／原專案追蹤，已直接讀取重驗。 |
| S024 | [nVidia Pascal GPU not supporting Cuda 13. Can I stick ...](https://www.reddit.com/r/archlinux/comments/1nxipcu/nvidia_pascal_gpu_not_supporting_cuda_13_can_i) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S025 | [PyTorch Version Compatibility With CUDA](https://www.facebook.com/fb-answers/pytorch-version-compatibility-with-cuda) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S026 | [CUDA toolkit version and architecture support update: Maxwell and Pascal architecture support removed in CUDA 12.8 and 12.9 builds - release/packaging - PyTorch Developer Mailing List](https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128) | 採用 | 官方發行政策／原專案追蹤，已直接讀取重驗。 |
| S027 | [Navigating GPU Architecture Support: A Guide for NVIDIA CUDA Developers ／ NVIDIA Technical Blog](https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers) | 採用或沿用定位 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S028 | [Pascal's Demise - notes on waning support for Nvidia Pascal architecture · GitHub](https://gist.github.com/apnea/ef56dbe682c3b3e0c82d6a2459f82554) | 淘汰或保留工具入口 | 搜尋命中多為其他GPU、一般benchmark工具，或P4網路語言同名；未驗得P4 ASR數據。 |
| S029 | [Bitsandbytes · Hugging Face](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes) | 保留交叉比對 | HF整合文件功能表，仍回bnb固定commit及實作，並保留README矛盾。 |
| S030 | [Bitsandbytes · Hugging Face](https://huggingface.co/docs/transformers/quantization/bitsandbytes) | 保留交叉比對 | HF整合文件功能表，仍回bnb固定commit及實作，並保留README矛盾。 |
| S031 | [BitsAndBytes W4 - vLLM-Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/quantization/bitsandbytes) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S032 | [Using bitsandbytes for 8-bit and 4-bit Quantization](https://apxml.com/courses/practical-llm-quantization/chapter-5-quantization-formats-tooling/using-bitsandbytes) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S033 | [Bitsandbytes - Learn AI](https://ai.miraheze.org/wiki/Bitsandbytes) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S034 | [Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA :: 밤에 쓰는 편지](https://letter-night.tistory.com/379) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S035 | [Quantizing models with bitsandbytes · bitsandbytes-foundation/bitsandbytes · Discussion #1163 · GitHub](https://github.com/bitsandbytes-foundation/bitsandbytes/discussions/1163) | 採用定位 | 原專案文件／release；正文以固定commit或tag重新核實，泛repo入口不作單獨主證。 |
| S036 | [AutoGPTQ/AutoGPTQ ／ DeepWiki](https://deepwiki.com/AutoGPTQ/AutoGPTQ) | 採用定位 | 原專案文件／release；正文以固定commit或tag重新核實，泛repo入口不作單獨主證。 |
| S037 | [AutoAWQ_kernels - casper-hansen](https://github.com/casper-hansen/AutoAWQ_kernels) | 採用定位 | 原專案文件／release；正文以固定commit或tag重新核實，泛repo入口不作單獨主證。 |
| S038 | [autoawq-kernels](https://pypi.org/project/autoawq-kernels) | 保留發行metadata | 套件索引不能證明特定GPU kernel；以官方source功能表判斷。 |
| S039 | [AutoGPTQ/setup.py at main · AutoGPTQ/AutoGPTQ · GitHub](https://github.com/AutoGPTQ/AutoGPTQ/blob/main/setup.py) | 採用定位 | 原專案文件／release；正文以固定commit或tag重新核實，泛repo入口不作單獨主證。 |
| S040 | [Release v0.7.0: Marlin int4*fp16 kernel, AWQ checkpoints loading · AutoGPTQ/AutoGPTQ · GitHub](https://github.com/AutoGPTQ/AutoGPTQ/releases/tag/v0.7.0) | 採用定位 | 原專案文件／release；正文以固定commit或tag重新核實，泛repo入口不作單獨主證。 |
| S041 | [Awq Quantization (Grade A) - Claude Skill ／ Skills Directory](https://www.skillsdirectory.com/skills/kalarislabs-awq-quantization) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S042 | [GPTQ vs AWQ vs GGUF: Which 4-Bit to Pick in 2026](https://theaiengineer.substack.com/p/quantization-in-practice-gptq-vs) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S043 | [[onnxrumtime]onnxruntime和cuda对应关系表-腾讯云开发者社区-腾讯云](https://cloud.tencent.com/developer/article/2543758) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S044 | [https://github.com/microsoft/onnxruntime/raw/refs/heads/gh-pages/docs/execution-providers/CUDA-ExecutionProvider.md](https://github.com/microsoft/onnxruntime/raw/refs/heads/gh-pages/docs/execution-providers/CUDA-ExecutionProvider.md) | 採用官方文件線索 | 後續回ORT文件與v1.18.0固定commit，避免跟隨gh-pages內容推論舊版kernel。 |
| S045 | [NVIDIA - CUDA ／ onnxruntime](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) | 採用 | 官方EP需求表；以固定ORT原始碼補足計算型別。 |
| S046 | [The Hidden Pitfalls of ONNXRuntime GPU Setup - DEV Community](https://dev.to/deskpai/the-hidden-pitfalls-of-onnxruntime-gpu-setup-4kb7) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S047 | [NVIDIA - CUDA ／ onnxruntime](https://faxu.github.io/onnxruntime/docs/execution-providers/CUDA-ExecutionProvider.html) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S048 | [onnxruntime-gpu](https://pypi.org/project/onnxruntime-gpu) | 保留發行metadata | 套件索引不能證明特定GPU kernel；以官方source功能表判斷。 |
| S049 | [Install ONNX Runtime ／ onnxruntime](https://onnxruntime.ai/docs/install) | 採用 | 官方EP需求表；以固定ORT原始碼補足計算型別。 |
| S050 | [NVIDIA Tesla P4 AI Performance and Hardware Specs ／ WareDB](https://www.waredb.com/processor/nvidia-tesla-p4) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S051 | [PassMark Video Card (GPU) Benchmarks - PassMark Software](https://www.videocardbenchmark.net/high_end_gpus.html/gpu.php?gpu=Tesla+P4) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S052 | [OpenAI Whisper Model Benchmarks: CPU vs GPU Latency — Voxxwire](https://voxxwire.com/articles/whisper-model-performance-benchmarks.html) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S053 | [Best GPU for Whisper: What the benchmarks show in 2026](https://www.runpod.io/articles/guides/best-gpu-for-whisper) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S054 | [NVIDIA Tesla P4 Inference Accelerator Data Sheet - Manuals+](https://manuals.plus/m/d18be6c561b9c8ba2732f3c630be08c9831b1f64ba509823bbb5cdc5f5598a33) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S055 | [Tesla P4 for AI — 8GB VRAM, 192 GB/s ／ FitMyLLM](https://www.fitmyllm.com/gpu/tesla-p4) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S056 | [https://arxiv.org/pdf/2507.07877](https://arxiv.org/pdf/2507.07877) | 保留論文，淘汰P4速度 | 原論文來源；未取得P4量化ASR可重現延遲A/B；不按搜尋片段當實測。 |
| S057 | [Tesla T4 vs Tesla P4 [1-Benchmark Showdown]](https://technical.city/en/gpu/Tesla-P4-vs-Tesla-T4) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S058 | [Paper page - Edge-ASR: Towards Low-Bit Quantization of Automatic Speech Recognition   Models](https://huggingface.co/papers/2507.07877) | 淘汰重複 | 論文聚合頁；回原arXiv，非P4速度證據。 |
| S059 | [Running LLMs on NVIDIA Tesla P4: What You Can Run — FitMyLLM](https://www.fitmyllm.com/blog/gpu/tesla-p4) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S060 | [Edge-ASR: Towards Low-Bit Quantization of Automatic Speech Recognition Models](https://arxiv.org/html/2507.07877v1) | 保留論文，淘汰P4速度 | 原論文來源；未取得P4量化ASR可重現延遲A/B；不按搜尋片段當實測。 |
| S061 | [Qwen3-ASR Technical Report](https://arxiv.org/html/2601.21337v1) | 保留論文，淘汰P4速度 | 原論文來源；未取得P4量化ASR可重現延遲A/B；不按搜尋片段當實測。 |
| S062 | [Paper page - Qwen3-ASR Technical Report](https://huggingface.co/papers/2601.21337) | 淘汰重複 | 論文聚合頁；回原arXiv，非P4速度證據。 |
| S063 | [Best Japanese ASR Models 2026: Qwen3 ASR vs Whisper vs Reazonspeech Benchmark ／ Neosophie Blog](https://neosophie.com/en/blog/20260226-japanese-asr-benchmark) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S064 | [Qwen3-ASR 1.7B Context, Specs & Sources (October 2026)](https://benchlm.ai/models/qwen3-asr-1-7b) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S065 | [Whisper vs Parakeet vs Canary vs Qwen3-ASR for broadcast](https://amiralabs.com/blog/whisper-parakeet-canary-qwen3-asr-broadcast-model-selection) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S066 | [Notice: CUDA 12.6 wheels will no longer be published from PyTorch 2.15 (drops Maxwell / Pascal / Volta) - release/packaging - PyTorch Developer Mailing List](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432) | 採用 | 官方發行政策／原專案追蹤，已直接讀取重驗。 |
| S067 | [referencesource/gpu-cuda-pytorch-compatibility · Datasets at Hugging Face](https://huggingface.co/datasets/referencesource/gpu-cuda-pytorch-compatibility) | 淘汰 | 彙整資料集，不取代PyTorch官方發行政策。 |
| S068 | [release/packaging - PyTorch Developer Mailing List](https://dev-discuss.pytorch.org/c/release-packaging/25) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S069 | [CUDA & PyTorch Compatibility ／ Guides ／ Clore.ai](https://docs.clore.ai/guides/getting-started/cuda-pytorch-compatibility) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S070 | [cuDNN Library](https://hpc.pku.edu.cn/docs/20170830182053891231.pdf) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S071 | [Release Notes — NVIDIA cuDNN Backend](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.11.0/release-notes.html) | 採用或沿用定位 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S072 | [Release Notes — NVIDIA cuDNN Backend](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/release-notes.html) | 採用或沿用定位 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S073 | [Release Notes — NVIDIA cuDNN v9.1.0 documentation](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.1.0/release-notes.html) | 採用或沿用定位 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S074 | [Mixed-Precision Programming with CUDA 8 ／ NVIDIA Technical Blog](https://developer.nvidia.com/blog/mixed-precision-programming-cuda-8) | 採用或沿用定位 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S075 | [cuDNN Archive ／ NVIDIA Developer](https://developer.nvidia.com/rdp/cudnn-archive) | 採用或沿用定位 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S076 | [TensorRT 10.0.1 Release Notes — NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.0.1.html) | 採用或沿用定位 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S077 | [NVIDIA TensorRT Documentation — NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html) | 採用或沿用定位 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S078 | [Architecture Overview — NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/architecture-overview.html) | 採用或沿用定位 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S079 | [TensorRT Release Notes - NVIDIA Docs](https://docs.nvidia.com/deeplearning/frameworks/container-release-notes/index.html) | 採用或沿用定位 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S080 | [sample failure of TensorRT 8.6 when running ...](https://github.com/NVIDIA/TensorRT/issues/3589) | 保留版本線索 | 回官方矩陣重驗；通用repo／issue不足以證明Pascal版本邊界。 |
| S081 | [GitHub - NVIDIA/TensorRT: NVIDIA® TensorRT™ is an SDK for high-performance deep learning inference on NVIDIA GPUs. This repository contains the open source components of TensorRT. · GitHub](https://github.com/NVIDIA/TensorRT) | 保留版本線索 | 回官方矩陣重驗；通用repo／issue不足以證明Pascal版本邊界。 |
| S082 | [Eval bug: vulkan performance drop pp512 560 -> 180 · Issue #23681 · ggml-org/llama.cpp · GitHub](https://github.com/ggml-org/llama.cpp/issues/23681) | 保留原專案線索 | 僅正文引用的固定commit／P4實驗採用；其他GPU/FA/opencl issue不外推。 |
| S083 | [Running older Pascal GPU with Llama.cpp · ggml-org/llama.cpp · Discussion #19248 · GitHub](https://github.com/ggml-org/llama.cpp/discussions/19248) | 保留原專案線索 | 僅正文引用的固定commit／P4實驗採用；其他GPU/FA/opencl issue不外推。 |
| S084 | [opencl: add iq4_nl support · ggml-org/llama.cpp@abfeb2e · GitHub](https://github.com/ggml-org/llama.cpp/actions/runs/24818962999) | 保留原專案線索 | 僅正文引用的固定commit／P4實驗採用；其他GPU/FA/opencl issue不外推。 |
| S085 | [IQ4_NL: 4-bit non-linear quants with blocks of 32 by ikawrakow · Pull Request #5590 · ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/pull/5590/files) | 保留原專案線索 | 僅正文引用的固定commit／P4實驗採用；其他GPU/FA/opencl issue不外推。 |
| S086 | [llama-cpp-pascal-patches/patches/40-mmq-iq4-nl-threads.patch ...](https://github.com/sbendhiba/llama-cpp-pascal-patches/blob/main/patches/40-mmq-iq4-nl-threads.patch) | 保留patch候選，未採實作 | 非上游Pascal patch，未在本P4 ASR驗證；以官方kernel為主。 |
| S087 | [CUDA: Pascal DP4A MMQ config uses I=64 for Q8_0; I=128 gives +12% pp512 on sm_61 · Issue #30235 · ggml-org/llama.cpp · GitHub](https://github.com/ggml-org/llama.cpp/issues/30235) | 採用，限旁證 | 作者P4量化MMQ實測；非ASR，未獨立重現。 |
| S088 | [NVIDIA CUDA Toolkit 13.0 Is Out: What Changed?](https://rottenwifi.com/nvidia-cuda-toolkit-13-0-is-out-major-changes-compatibility-and-upgrade-advice) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S089 | [CUDA Toolkit 13.0 - Release Notes — Release Notes 13.0 documentation](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html) | 採用或沿用定位 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S090 | [CUDA Toolkit 13.0 Update 3 - Release Notes — Release Notes 13.0 documentation](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-toolkit-release-notes/index.html) | 採用或沿用定位 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S091 | [CUDA Toolkit 13.2 - Release Notes — Release Notes 13.2 documentation](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-toolkit-release-notes/index.html) | 採用或沿用定位 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S092 | [NVIDIA CUDA Toolkit 13.0 Is Out: Changes and Compatibility](https://blog.rottenwifi.com/nvidia-cuda-toolkit-13-0-is-out-what-changed-and-who-should-upgrade) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S093 | [Release Notes](https://docs.nvidia.com/cuda/pdf/CUDA_Toolkit_Release_Notes.pdf) | 採用或沿用定位 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S094 | [TensorFlow Release 24.03 - NVIDIA Docs](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-24-03.html) | 採用或沿用定位 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S095 | [nVidia release versions compatibility - TensorRT - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/nvidia-release-versions-compatibility/265643) | 保留線索 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S096 | [Support Matrix :: NVIDIA cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-892/support-matrix/index.html) | 採用或沿用定位 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S097 | [NVIDIA cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-891/index.html) | 採用或沿用定位 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S098 | [Support Matrix — NVIDIA cuDNN Backend](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.10.2/reference/support-matrix.html) | 採用或沿用定位 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S099 | [Release Notes — NVIDIA cuDNN v9.2.1 documentation](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.2.1/release-notes.html) | 採用或沿用定位 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S100 | [cudnn - conda-forge ／ Anaconda.org](https://anaconda.org/conda-forge/cudnn) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S101 | [P4 Benchmark - usi-systems.github.io](https://usi-systems.github.io/p4benchmark) | 淘汰或保留工具入口 | 搜尋命中多為其他GPU、一般benchmark工具，或P4網路語言同名；未驗得P4 ASR數據。 |
| S102 | [Benchmarking Results · openai/whisper · Discussion #530 · GitHub](https://github.com/openai/whisper/discussions/530) | 淘汰或保留工具入口 | 搜尋命中多為其他GPU、一般benchmark工具，或P4網路語言同名；未驗得P4 ASR數據。 |
| S103 | [GitHub - siddhantbajaj1/Whippersnapper_p4benchmark_with_p4 ...](https://github.com/siddhantbajaj1/Whippersnapper_p4benchmark_with_p4-version16_Compatibility) | 淘汰或保留工具入口 | 搜尋命中多為其他GPU、一般benchmark工具，或P4網路語言同名；未驗得P4 ASR數據。 |
| S104 | [GitHub - steveneppler/whisperbench: Whisper benchmark · GitHub](https://github.com/steveneppler/whisperbench) | 淘汰或保留工具入口 | 搜尋命中多為其他GPU、一般benchmark工具，或P4網路語言同名；未驗得P4 ASR數據。 |
| S105 | [Performance benchmark of different GPUs · openai/whisper · Discussion #918 · GitHub](https://github.com/openai/whisper/discussions/918) | 淘汰P4效能 | 非合格P4量化ASR A/B；其中P40 FP16泛化也不能採信。 |
| S106 | [GitHub - cloudmercato/whisper-benchmark: A simple tool to ...](https://github.com/cloudmercato/whisper-benchmark) | 淘汰或保留工具入口 | 搜尋命中多為其他GPU、一般benchmark工具，或P4網路語言同名；未驗得P4 ASR數據。 |
| S107 | [Misc. bug: (clip.cpp) q8_0 mmproj is broken on gemma 3 · Issue #13025 · ggml-org/llama.cpp · GitHub](https://github.com/ggml-org/llama.cpp/issues/13025) | 保留原專案線索 | 僅正文引用的固定commit／P4實驗採用；其他GPU/FA/opencl issue不外推。 |
| S108 | [Another new llama.cpp / GGML breaking change, affecting ...](https://www.reddit.com/r/LocalLLaMA/comments/13md90j/another_new_llamacpp_ggml_breaking_change) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S109 | [ggml-org/llama.cpp issues and pull requests ／ Ecosyste.ms: Issues](https://issues.ecosyste.ms/hosts/GitHub/repositories/ggml-org%2Fllama.cpp/issues) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S110 | [Issues · ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/labels/help%20wanted) | 保留原專案線索 | 僅正文引用的固定commit／P4實驗採用；其他GPU/FA/opencl issue不外推。 |
| S111 | [api.github.com](https://api.github.com/repos/ggml-org/llama.cpp/releases/latest) | 保留原專案metadata | release索引不能取代固定source commit的kernel選擇。 |
| S112 | [TensorRT - Wikipedia](https://en.wikipedia.org/wiki/TensorRT) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S113 | [What is NVIDIA TensorRT? Optimize YOLO26 ／ Ultralytics](https://www.ultralytics.com/glossary/tensorrt) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S114 | [TensorRT download ／ SourceForge.net](https://sourceforge.net/projects/tensorrt.mirror) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S115 | [Unsupported SM: 0x601 failure of TensorRT 10.0.1 on 1080ti · Issue #3826 · NVIDIA/TensorRT · GitHub](https://github.com/NVIDIA/TensorRT/issues/3826) | 淘汰支援主證 | SM61錯誤案例；建議降9.1的留言者非NVIDIA維護者，不能證明9.x官方支援。 |
| S116 | [TensorRT/CHANGELOG.md at main · NVIDIA/TensorRT · GitHub](https://github.com/NVIDIA/TensorRT/blob/main/CHANGELOG.md) | 保留版本線索 | 回官方矩陣重驗；通用repo／issue不足以證明Pascal版本邊界。 |
| S117 | [Release Notes — TensorRT LLM](https://nvidia.github.io/TensorRT-LLM/release-notes.html) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S118 | [Making OpenAI Whisper faster - Nikolas' Blog](https://nikolas.blog/making-openai-whisper-faster) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S119 | [speed up whisper? · openai/whisper · Discussion #716 · GitHub](https://github.com/openai/whisper/discussions/716) | 淘汰P4效能 | 讀取案例為P100／T4，不是P4量化A/B。 |
| S120 | [Whisper.cpp vs faster-whisper 2026 : Benchmarks STT](https://www.promptquorum.com/power-local-llm/local-whisper-stt-comparison) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S121 | [Tesla P4 [in 1 benchmark]](https://technical.city/en/gpu/Tesla-P4) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S122 | [Whisper.cpp Benchmark Report: Complete Performance Analysis on Legacy Hardware (Intel Core i5-460M) · ggml-org/whisper.cpp · Discussion #3752 · GitHub](https://github.com/ggml-org/whisper.cpp/discussions/3752) | 淘汰P4效能 | 原作者實測硬體是Intel i5-460M CPU。 |
| S123 | [Testing and Benchmarking](https://deepwiki.com/ggml-org/whisper.cpp/6.2-testing-and-benchmarking) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S124 | [Whisper.cpp Benchmark](https://openbenchmarking.org/test/pts/whisper-cpp) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S125 | [Comparision with faster-whisper · Issue #1127 · ggml-org/whisper.cpp · GitHub](https://github.com/ggml-org/whisper.cpp/issues/1127) | 淘汰P4效能 | CPU／RTX4080等比較，非P4。 |
| S126 | [v1.9.1 · ggml-org/whisper.cpp · Discussion #3895 · GitHub](https://github.com/ggml-org/whisper.cpp/discussions/3895) | 保留候選，非P4實測 | CUDA支援／其他硬體議題，未取得合格P4量化ASR A/B。 |
| S127 | [Releases · ggml-org/whisper.cpp · GitHub](https://github.com/ggml-org/whisper.cpp/releases) | 保留候選，非P4實測 | CUDA支援／其他硬體議題，未取得合格P4量化ASR A/B。 |
| S128 | [CUDA error · Issue #2258 · ggml-org/whisper.cpp · GitHub](https://github.com/ggml-org/whisper.cpp/issues/2258) | 保留候選，非P4實測 | CUDA支援／其他硬體議題，未取得合格P4量化ASR A/B。 |
| S129 | [GitHub - ggml-org/whisper.cpp: Port of OpenAI's Whisper model in C/C++ · GitHub](https://github.com/ggml-org/whisper.cpp) | 保留候選，非P4實測 | CUDA支援／其他硬體議題，未取得合格P4量化ASR A/B。 |
| S130 | [Dockerfile for Whisper.cpp with PyTorch, CUDA, and GPU Support (whisper-cli / whisper-stream) · Issue #2752 · ggml-org/whisper.cpp · GitHub](https://github.com/ggml-org/whisper.cpp/issues/2752) | 保留候選，非P4實測 | CUDA支援／其他硬體議題，未取得合格P4量化ASR A/B。 |
| S131 | [Windows 2.8.0+cu126 doesn't have Maxwell kernel support? · Issue #160575 · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/issues/160575) | 保留版本線索 | 回團隊公告／實際CUDA build架構判斷，泛release或builder頁不單獨作主證。 |
| S132 | [Building Pytorch3d with CUDA12.8+pytorch2.8 on RTX 5090 GPU · Issue #1962 · facebookresearch/pytorch3d · GitHub](https://github.com/facebookresearch/pytorch3d/issues/1962) | 淘汰或保留工具入口 | 搜尋命中多為其他GPU、一般benchmark工具，或P4網路語言同名；未驗得P4 ASR數據。 |
| S133 | [GitHub - pytorch/builder: Continuous builder and binary build scripts for pytorch · GitHub](https://github.com/pytorch/builder) | 保留版本線索 | 回團隊公告／實際CUDA build架構判斷，泛release或builder頁不單獨作主證。 |
| S134 | [Releases · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/releases) | 保留版本線索 | 回團隊公告／實際CUDA build架構判斷，泛release或builder頁不單獨作主證。 |
| S135 | [builder/CUDA_UPGRADE_GUIDE.MD at main · pytorch/builder · GitHub](https://github.com/pytorch/builder/blob/main/CUDA_UPGRADE_GUIDE.MD) | 保留版本線索 | 回團隊公告／實際CUDA build架構判斷，泛release或builder頁不單獨作主證。 |
| S136 | [NVIDIA TensorRT Documentation — NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest) | 採用或沿用定位 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S137 | [NVIDIA TensorRT Product Family - NVIDIA Docs](https://docs.nvidia.com/tensorrt/index.html) | 採用或沿用定位 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S138 | [TensorRT SDK ／ NVIDIA Developer](https://developer.nvidia.com/tensorrt) | 採用或沿用定位 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S139 | [tensorrt 9.2 for Tensorrt llm 0.6? · Issue #3514 · NVIDIA/TensorRT · GitHub](https://github.com/NVIDIA/TensorRT/issues/3514) | 保留版本線索 | 回官方矩陣重驗；通用repo／issue不足以證明Pascal版本邊界。 |
| S140 | [Issues · NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT/issues) | 保留版本線索 | 回官方矩陣重驗；通用repo／issue不足以證明Pascal版本邊界。 |
| S141 | [New Pascal GPUs Accelerate Inference in the Data Center ／ NVIDIA Technical Blog](https://developer.nvidia.com/blog/new-pascal-gpus-accelerate-inference-in-the-data-center) | 採用或沿用定位 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S142 | [Nvidia Pascal TITAN Xp, TITAN X, GeForce GTX 1080 Ti, GTX 1080, GTX 1070, GTX 1060, GTX 1050 & GT 1030 - Page 2 - CUDA Programming and Performance - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/nvidia-pascal-titan-xp-titan-x-geforce-gtx-1080-ti-gtx-1080-gtx-1070-gtx-1060-gtx-1050-gt-1030/42660?page=2) | 保留線索 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S143 | [NVIDIA GP104 GPU Specs - TechPowerUp](https://www.techpowerup.com/gpu-specs/nvidia-gp104.g793) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S144 | [Hope to add Tesla P4 to enable WDDM #357 - GitHub](https://github.com/dartraiden/NVIDIA-patcher/issues/357) | 淘汰或保留工具入口 | 搜尋命中多為其他GPU、一般benchmark工具，或P4網路語言同名；未驗得P4 ASR數據。 |
| S145 | [tensorrt · PyPI](https://pypi.org/project/tensorrt) | 保留發行metadata | 套件索引不能證明特定GPU kernel；以官方source功能表判斷。 |
| S146 | [NVIDIA TensorRT ／ NVIDIA 开发者](https://developer.nvidia.cn/tensorrt) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S147 | [Installing TensorRT — NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html) | 採用或沿用定位 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S148 | [High performance inference with TensorRT Integration — The TensorFlow Blog](https://blog.tensorflow.org/2019/06/high-performance-inference-with-TensorRT.html) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S149 | [Tensorrt 9.3.0 - TensorRT - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/tensorrt-9-3-0/344516) | 保留線索 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S150 | [Introduction to TensorRT](https://python.plainenglish.io/introduction-to-tensorrt-b3edf88c746c) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S151 | [Speeding Up Deep Learning Inference Using TensorRT - Edge AI and Vision Alliance](https://www.edge-ai-vision.com/2020/04/speeding-up-deep-learning-inference-using-tensorrt) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S152 | [NVIDIA TensorRT 10.0 Upgrades Usability, Performance, and AI Model Support ／ Diffusion Singularity](https://dsin.ai/news/article/dcMxlWm/nvidia_tensorrt_10_0_upgrades_usability_performance_and_ai_model_support) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S153 | [Unsupported SM: 0x601 - TensorRT](https://forums.developer.nvidia.com/t/unsupported-sm-0x601/289377) | 保留線索 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S154 | [Support Matrix — NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html) | 採用或沿用定位 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S155 | [TensorRT installation question - General Discussion - Google AI Developers Forum](https://discuss.ai.google.dev/t/tensorrt-installation-question/29969) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S156 | [NVIDIA - TensorRT ／ onnxruntime](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html) | 採用 | 官方EP需求表；以固定ORT原始碼補足計算型別。 |
| S157 | [GitHub - microsoft/onnxruntime: ONNX Runtime: cross-platform, high performance ML inferencing and training accelerator · GitHub](https://github.com/microsoft/onnxruntime) | 淘汰或保留工具入口 | 搜尋命中多為其他GPU、一般benchmark工具，或P4網路語言同名；未驗得P4 ASR數據。 |
| S158 | [NuGet Gallery ／ Microsoft.ML.OnnxRuntime 1.31.0](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime) | 保留發行metadata | 套件索引不能證明特定GPU kernel；以官方source功能表判斷。 |
| S159 | [Releases · microsoft/onnxruntime · GitHub](https://github.com/microsoft/onnxruntime/releases) | 淘汰或保留工具入口 | 搜尋命中多為其他GPU、一般benchmark工具，或P4網路語言同名；未驗得P4 ASR數據。 |
| S160 | [onnxruntime Changelog](https://data.safetycli.com/packages/pypi/onnxruntime/changelog) | 淘汰主證 | 二手解說／聚合／估算／其他硬體／不相關；未回到合格P4量化ASR一手證據。 |
| S161 | [Releases · bitsandbytes-foundation/bitsandbytes · GitHub](https://github.com/bitsandbytes-foundation/bitsandbytes/releases) | 採用定位 | 原專案文件／release；正文以固定commit或tag重新核實，泛repo入口不作單獨主證。 |
| S162 | [bitsandbytes-foundation/bitsandbytes on GitHub ／ Release Alert](https://releasealert.dev/github/bitsandbytes-foundation/bitsandbytes) | 淘汰主證／保留發行線索 | 第三方發行彙整，非原專案；正文回到官方固定commit或release核實。 |
| S163 | [bitsandbytes](https://pypi.org/project/bitsandbytes) | 保留發行metadata | 套件索引不能證明特定GPU kernel；以官方source功能表判斷。 |
| S164 | [GitHub - bitsandbytes-foundation/bitsandbytes: Accessible large language models via k-bit quantization for PyTorch. · GitHub](https://github.com/bitsandbytes-foundation/bitsandbytes) | 採用定位 | 原專案文件／release；正文以固定commit或tag重新核實，泛repo入口不作單獨主證。 |
| S165 | [CUDA Features Archive — CUDA Features Archive 12.9 documentation](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-features-archive/index.html) | 採用或沿用定位 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S166 | [CUDA Features Archive Release 12.9 - NVIDIA Documentation Hub](https://docs.nvidia.com/cuda/archive/12.9.1/pdf/CUDA_Features_Archive.pdf) | 採用或沿用定位 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S167 | [CUDA Toolkit Documentation 12.9 Update 1](https://docs.nvidia.com/cuda/archive/12.9.1) | 採用或沿用定位 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |
| S168 | [1. CUDA 12.9 Release Notes — Release Notes 12.9 documentation](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html) | 採用或沿用定位 | 廠商一手來源；只有正文明確核實章節用作結論，通用首頁與404不採。 |

### 深讀／HTTP補查與無效來源

- Tavily extract第一批：CUDA13.0 release notes、Pascal Tuning Guide、cuDNN8.9.2 matrix、ORT CUDA EP、legacy GPU表、CUDA架構支援文章。均成功回傳，正文採用相關章節。
- 第二批：whisper.cpp #3752、#1127、whisper #716、Edge-ASR論文、faster-whisper README、TRT10.0.1 release notes、TRT9.3舊矩陣。後者雖被extract列為成功，內容實為Page Not Found，**淘汰**；不能把HTTP/extractor成功當成證據成立。
- 第三批：NVIDIA P4 GP104發表文章、TRT10.0.1舊矩陣URL。後者轉到新TRT首頁，**淘汰其歷史矩陣身分**；改直接讀現行support-matrix HTML內的 `supportData['10.0.1']`，Linux／Windows明列 `['7.0','7.5','8.0','8.6','8.9','9.0']`，不含6.1。
- 直接HTTP取得GitHub commit API、raw.githubusercontent.com固定SHA的CMake／CUDA／Vulkan／ORT／量化專案source。僅小型文字檔，不下載repo tarball、GPU stack或模型。llama pin與current、ORT、bnb、AutoAWQ、AutoGPTQ、Marlin及vLLM均核對commit時間。
- current vLLM的舊路徑 `vllm/model_executor/layers/quantization/{gptq,awq,gptq_marlin}.py` 回404，tree已顯示改名。**沒有拿舊路徑推斷current kernel**；本報告只用其官方GPU硬體floor判定P4已在入口排除。
- NVIDIA `archives/tensorrt-{900,901,930}/...` 多個舊release／matrix URL直接HTTP回404；9.x沒有得到可採用版本矩陣。TRT issue #3826建議降9.1的留言者 `author_association=NONE`，維護者僅回Checking internally，故不採為9.x官方支援證據。
- bitsandbytes固定current `a616886…` 的installation功能表與README互相矛盾；另查0.48.0、0.50.2 tag文件及release，矛盾沒有被簡化成已驗證P4 LLM.int8快路徑。原始碼的fallback也有記錄，見B1–B4。
- NVIDIA新矩陣／官網舊頁會重新導向，引用時區分正文版本。搜尋結果對GP104核心數、FP16能力及P4/P40/P100/T4常混淆，只有NVIDIA的GPU表、P4發表文章及Pascal tuning guide作硬體主證。

### 本次命令／重現方式（只讀示意）

所有shell呼叫都以 `rtk` 為前綴。工具呼叫本身不經shell。常用命令形式：

```sh
rtk read /home/df/.codex/RTK.md
rtk read /home/df/.agents/skills/df-meta-mcp/SKILL.md
rtk rg --files docs/reports
rtk git rev-parse HEAD
rtk proxy python3 - <<'PY'
# 用urllib.request讀GitHub API或官方raw source，固定commit後以splitlines逐行核對。
# 不安裝依賴、不執行GPU程式、不接觸正式容器。
PY
```

研究暫存為 `/tmp/p4-kernel-evidence-20261101/` 的小型文字source與API回應，僅供本次逐行核對；不把其存在當成可攜式證據。所有必要結論、精確URL、SHA／行範圍、查詢與候選取捨均留在本檔。

<!-- 來源連結定義：便於正文就地點擊固定commit證據。 -->
[H1]: https://developer.nvidia.com/blog/new-pascal-gpus-accelerate-inference-in-the-data-center/
[H2]: https://images.nvidia.com/content/pdf/tesla/184457-Tesla-P4-Datasheet-NV-Final-Letter-Web.pdf
[H3]: https://developer.nvidia.com/cuda-legacy-gpus
[H4]: https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html#fp16-arithmetic-support
[H5]: https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/support-matrix/index.html#hardware-precision-matrix
[R1]: https://github.com/DF-wu/CapsWriter-Offline-Container/blob/95912cf66f384aa752f3dda291e02c893070beff/docker/server/download_models.py#L414-L432
[R2]: https://github.com/DF-wu/CapsWriter-Offline-Container/blob/95912cf66f384aa752f3dda291e02c893070beff/docker/server/Dockerfile#L1-L14
[R3]: https://github.com/DF-wu/CapsWriter-Offline-Container/blob/95912cf66f384aa752f3dda291e02c893070beff/requirements/server-docker.txt#L1-L5
[R4]: https://github.com/DF-wu/CapsWriter-Offline-Container/blob/95912cf66f384aa752f3dda291e02c893070beff/core/server/engines/qwen_asr_gguf/inference/encoder.py#L130-L168
[R5]: https://github.com/DF-wu/CapsWriter-Offline-Container/blob/95912cf66f384aa752f3dda291e02c893070beff/core/server/engines/qwen_asr_gguf/inference/encoder.py#L232-L263
[L1]: https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-cuda/CMakeLists.txt#L8-L35
[L2]: https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/ggml/src/ggml-cuda/CMakeLists.txt#L21-L50
[L3]: https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-cuda/common.cuh#L224-L289
[L4]: https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-cuda/common.cuh#L665-L708
[L5]: https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-cuda/ggml-cuda.cu#L2182-L2264
[L6]: https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-cuda/mmq.cu#L262-L314
[L7]: https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-cuda/mmvq.cuh#L1-L12
[L8]: https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-cuda/mmvq.cu#L10-L31
[L9]: https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-cuda/mmq.cu#L6-L65
[L10]: https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-cuda/vecdotq.cuh#L461-L580
[L11]: https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-cuda/vecdotq.cuh#L231-L269
[L12]: https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-cuda/vecdotq.cuh#L1173-L1195
[L13]: https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-common.h#L291-L324
[L14]: https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-common.h#L219-L237
[L15]: https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-common.h#L415-L420
[L16]: https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-cuda/ggml-cuda.cu#L1285-L1367
[L17]: https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/ggml/src/ggml-cuda/ggml-cuda.cu#L2112-L2164
[L18]: https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/ggml/src/ggml-cuda/mmq.cu#L382-L450
[L19]: https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/ggml/src/ggml-cuda/mmvq.cu#L318-L427
[L20]: https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/ggml/src/ggml-cuda/mmq.cuh#L218-L299
[L21]: https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/ggml/src/ggml-cuda/mmq-config-pascal-dp4a.cuh#L74-L134
[L22]: https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/ggml/src/ggml-cuda/mmq-config-pascal-dp4a.cuh#L234-L244
[L23]: https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/ggml/src/ggml-cuda/ggml-cuda.cu#L1764-L1805
[V1]: https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L4710-L4723
[V2]: https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L5355-L5388
[V3]: https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L3464-L3505
[V4]: https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L5704-L5779
[V5]: https://github.com/ggml-org/llama.cpp/blob/c301172f660a1fe0b42023da990bf7385d69adb4/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L7014-L7048
[V6]: https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L2738-L2761
[V7]: https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L2989-L3013
[V8]: https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L6368-L6379
[V9]: https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L6656-L6692
[V10]: https://github.com/ggml-org/llama.cpp/blob/0ec4027a2a0183de87c007069adddc06e665f7de/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L2475-L2489
[O1]: https://github.com/microsoft/onnxruntime/blob/45737400a2f3015c11f005ed7603611eaed306a6/onnxruntime/core/providers/cuda/nn/conv.cc#L489-L535
[O2]: https://github.com/microsoft/onnxruntime/blob/45737400a2f3015c11f005ed7603611eaed306a6/onnxruntime/core/providers/cuda/nn/conv.cc#L361-L425
[O3]: https://github.com/microsoft/onnxruntime/blob/45737400a2f3015c11f005ed7603611eaed306a6/onnxruntime/core/providers/cuda/cuda_common.h#L143-L197
[O4]: https://github.com/microsoft/onnxruntime/blob/45737400a2f3015c11f005ed7603611eaed306a6/onnxruntime/core/providers/cuda/shared_inc/fpgeneric.h#L76-L116
[O5]: https://github.com/microsoft/onnxruntime/blob/45737400a2f3015c11f005ed7603611eaed306a6/onnxruntime/core/providers/cuda/math/matmul.cc#L146-L177
[O6]: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements
[O7]: https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-892/support-matrix/index.html#cudnn-cuda-hardware-versions
[C1]: https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html#deprecated-architectures
[C2]: https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers/
[P1]: https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128
[P2]: https://github.com/pytorch/pytorch/issues/157517
[P3]: https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432
[B1]: https://github.com/bitsandbytes-foundation/bitsandbytes/blob/a616886cacf86701514f0cb1d6be760f967481c0/docs/source/installation.mdx#L25-L64
[B2]: https://github.com/bitsandbytes-foundation/bitsandbytes/blob/a616886cacf86701514f0cb1d6be760f967481c0/README.md#L40-L68
[B3]: https://github.com/bitsandbytes-foundation/bitsandbytes/blob/a616886cacf86701514f0cb1d6be760f967481c0/bitsandbytes/backends/cuda/ops.py#L118-L165
[B4]: https://github.com/bitsandbytes-foundation/bitsandbytes/releases/tag/0.48.0
[A1]: https://github.com/casper-hansen/AutoAWQ/blob/88e4c76b20755db275574e6a03c83c84ba3bece5/README.md#L1-L8
[A2]: https://github.com/casper-hansen/AutoAWQ/blob/88e4c76b20755db275574e6a03c83c84ba3bece5/README.md#L59-L66
[G1]: https://github.com/AutoGPTQ/AutoGPTQ/blob/9f7d37072917ab3a7545835f23e808294a542153/setup.py#L44-L48
[G2]: https://github.com/AutoGPTQ/AutoGPTQ/blob/9f7d37072917ab3a7545835f23e808294a542153/setup.py#L100-L119
[G3]: https://github.com/AutoGPTQ/AutoGPTQ/blob/9f7d37072917ab3a7545835f23e808294a542153/README.md#L1-L20
[G4]: https://github.com/AutoGPTQ/AutoGPTQ/blob/9f7d37072917ab3a7545835f23e808294a542153/README.md#L334-L336
[M1]: https://github.com/IST-DASLab/marlin/blob/1f25790bdd49fba53106164a24666dade68d7c90/README.md#L7-L28
[M2]: https://github.com/IST-DASLab/marlin/blob/1f25790bdd49fba53106164a24666dade68d7c90/README.md#L60-L66
[W1]: https://github.com/vllm-project/vllm/blob/b17039bccc17b94a2ea107272ad7bc93508708df/docs/getting_started/installation/gpu.cuda.inc.md#L1-L18
[W2]: https://github.com/vllm-project/vllm/blob/0c63ce9f0e0b5b2782f0f30cb491069dac158bb2/docs/getting_started/installation/gpu.cuda.inc.md#L1-L53
[T1]: https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html
[T2]: https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#requirements
[E1]: https://github.com/ggml-org/llama.cpp/issues/30235
[E2]: https://github.com/SYSTRAN/faster-whisper#benchmark
[E3]: https://opennmt.net/CTranslate2/quantization.html#implicit-type-conversion-on-load
[E4]: https://github.com/ggml-org/whisper.cpp/discussions/3752
[E5]: https://arxiv.org/html/2507.07877v1
