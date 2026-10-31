# P4 容器實驗證據與重現

主報告：[p4-container-experiments-20261101.md](../../p4-container-experiments-20261101.md)。本目錄不含模型、音訊、編譯 libraries 或第三方原始碼。執行日期為 2026-11-01，正式服務未修改。

## 先重算已保存結果（不需要 GPU）

在 repository root：

```sh
rtk proxy python3 docs/reports/experiments/p4-20261101/analyze.py docs/reports/experiments/p4-20261101/results
rtk proxy python3 docs/reports/experiments/p4-20261101/audit_results.py docs/reports/experiments/p4-20261101/results
```

`analyze.py` 排除暖機與 pilot，p95 用 nearest-rank；每 clip 的 n=10，故 p95=max。`audit_results.py` 核對 6 配置×30 latency、6×5 quality、每 clip 次數、有效輸出、重試、15 個 GPU blocks，以及 9 次 temperature=0.4 smoke。它另計算 CER／英文 WER、相對 CPU 的輸出差異與 thermal snapshots。這是保存證據的驗證程式，不是應用程式的單元測試。

## 重新做 GPU 實驗需要的資源

- 保留的 runtime image ID：`sha256:b413de423317afd354ddeaa17898fabbd4ae03bcb8040f261dc029ec451f5ea5`；用 ID 固定，不用會漂移的 tag。
- 當次 compiler image：`nvidia/cuda:11.8.0-devel-ubuntu22.04`，RepoDigest `sha256:94fd755736cb58979173d491504f0b573247b1745250249415b07fefc738e41f`。當次測完已移除，重做須重新取得。
- 正式模型 volume `capswriter_capswriter-server-models`，**唯讀** 掛到 `/models-original`。原模型目錄 `/models-original/Qwen3-ASR/Qwen3-ASR-1.7B`。先核對 environment 中原模型 SHA256；不同 artifact 不應標成同一實驗。
- 獨立 scratch bind mount `/bench`，以及 test／compiler container；不要使用正式容器做 apt/pip 或改 library symlink。
- 測試容器：`--gpus all --cpus 6 --memory 16g --pids-limit 512`，沒有 published ports，`NVIDIA_DRIVER_CAPABILITIES=all`、`OMP_NUM_THREADS=4`、`OPENBLAS_NUM_THREADS=4`、`PYTHONPATH=/bench/src`、workdir `/bench/src`、entrypoint 覆寫為 `tail -f /dev/null`。
- compiler 不需要 GPU，只掛同一 `/bench`。兩容器標籤 `capswriter.experiment=p4-20261101`；再次實驗請使用新的命名／標籤，並同步清理檢查。
- scratch 的 `/bench/src` 是 code commit `d3b334e95c4d958f6d7812cb94c14a33fff427d3` 的乾淨 git archive。`/bench/vulkan-bin` 為當時正式 `/app/core/server/engines/llama/bin` 的唯讀複製；核對 environment 中 library hash。

當次 host state 在 `/tmp/capswriter-p4-experiment-state.json`，runner 讀取格式：

```json
{
  "scratch": "/tmp/capswriter-p4-exp-UNIQUE",
  "container": "YOUR-TEST-CONTAINER",
  "compiler_container": "YOUR-COMPILER-CONTAINER",
  "base_image": "sha256:b413de423317afd354ddeaa17898fabbd4ae03bcb8040f261dc029ec451f5ea5",
  "model_volume": "capswriter_capswriter-server-models",
  "created_images": ["nvidia/cuda:11.8.0-devel-ubuntu22.04"],
  "created_volumes": []
}
```

`created_images` 必須只列當次新增者；不要把原有 image 當垃圾。這個 state file 是一次性執行識別，不是正式設定。不要同時跑兩份矩陣，因 runner 會切換 `/bench/src/core/server/engines/llama/bin` symlink。

將本目錄 `.py` 與 `samples.json` 複製到 scratch root，建立 `audio`、`logs`、`results`、`models` 等子目錄。容器內 symlink 的 target 要以 `/bench/...` 或 `/models-original/...` 表示，而非 host scratch 絕對路徑。

## 模型與公開音訊

原 encoder 與 Q5 僅讀 volume。下載 [Q4 ZIP](https://github.com/HaujetZhao/CapsWriter-Offline/releases/download/models/Qwen3-ASR-1.7B-q4_k.zip)，先驗 ZIP SHA256 `9b3d2a66a4a26a0404c32085ec838b7c482495a7827919a5aa674de617c2757b`。三個檔案解到 `/bench/models/q4-full`，核對 environment 的個別 SHA256。`/bench/models/q4` 則連原 encoder 兩檔與這份 Q4 decoder。

測試容器只加 `onnx==1.16.2`（其餘 runtime packages 沿用映像）；在 `/bench/models/fp32-promoted` 執行：

```sh
# 以下兩條在 test container 執行。
python /bench/promote_encoder.py /models-original/Qwen3-ASR/Qwen3-ASR-1.7B/qwen3_asr_encoder_frontend.onnx /bench/models/fp32-promoted/qwen3_asr_encoder_frontend.onnx
python /bench/promote_encoder.py /models-original/Qwen3-ASR/Qwen3-ASR-1.7B/qwen3_asr_encoder_backend.onnx /bench/models/fp32-promoted/qwen3_asr_encoder_backend.onnx
```

該目錄 decoder symlink 指向原 Q5。promotion 做相同 half→float 轉換與分層 topological sort；重現腳本已對當次兩個 promoted artifact 驗得 **byte-identical SHA256**。它不恢復原始浮點精度。

`samples.json` 列音檔來源、固定 revision、reference、hash 與 transform。當次取得方法：

1. LibriSpeech dummy clean/validation 前 3 rows，API 端點在 manifest；下載 rows 回傳的音檔，儲存為列出的 FLAC 名稱。簽名 URL 有期限，不應提交。核對 dataset revision 與音檔 SHA256，避免把最新 API 結果當固定 artifact。
2. `noise2.wav`、`codeswitch.wav` 由 manifest 的固定 HF revision `resolve` URL 下載；原 reference 取同 revision 的 `test_wavs/transcript.txt`。
3. codeswitch 經 ffmpeg `-ar 16000 -ac 1` 轉成 `codeswitch-16k.wav`；noise2 原為 mono/16k。2／5／10 秒片段用 soundfile 讀 `float32`，切 `audio[:16000*seconds]`，`sf.write(..., 16000)`（預設 WAV PCM_16）。最後核對 manifest 的每個 audio SHA256。

不用任何私人音訊。這份 manifest 提供的是小型可重現 smoke 語料，不是完整品質驗證集。

## CUDA build

下載固定 [GitHub API tarball](https://api.github.com/repos/ggml-org/llama.cpp/tarball/c301172f660a1fe0b42023da990bf7385d69adb4)，當次 SHA256 `4420071ba535356cb62f8cd65edb924dee321942d45b7e5e402df1f2906e1031`。解壓到 scratch 並建立 `/bench/llama-source` symlink。

compiler 安裝 `cmake ninja-build g++` 等 build 工具。以下在 compiler container 執行，完整 compiler／cmake logs 已保存：

```sh
cmake -S /bench/llama-source -B /bench/cuda-build -G Ninja \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON
cmake --build /bench/cuda-build --target llama llama-quantize -j8
```

當次無 GPU compiler 在最後 link 失敗，補 compiler-only stub 後才成功：

```sh
ln -sf libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs cmake --build /bench/cuda-build --target llama llama-quantize -j8
cuobjdump --list-elf /bench/cuda-build/bin/libggml-cuda.so.0.9.5
```

runtime **不可**帶入這個 stub `LD_LIBRARY_PATH`。實際 GPU 確認用 ggml devices、offload log、SM61 cubin 與 runtime GPU snapshots 一起看。source tarball 沒有 `.git`，build log 的 commit 是 unknown；來源由下載 commit、tar SHA256 與 library hashes 固定。

## 執行順序

1. 先驗原映像 EGL／Vulkan discovery，記錄 CPU devices。原 CPU baseline 使用原 F16／Q5、temperature=0、10 repeats、warmup=2，`phase=latency`，輸出 `results/shipped-cpu-fallback.jsonl`。原 runner 當時尚無 run_id；audit 依單一初始 process 處理，不竄改 raw rows。
2. 只在 test container apt 安裝 `libegl1`、`strace`。保存 discovery before/after；Vulkan 測試設定 `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`。當次未用 graphics DSO 副本或自訂 LD_LIBRARY_PATH。
3. 每個候選先 pilot，核對輸出與 GPU backend。runner 枚舉不到 CUDA／Vulkan 時會報錯，除非明確 `--allow-cpu-decoder`。
4. host `python3 "$TASK_ROOT/matrix.py"`：15 個 GPU latency blocks（4/3/3 repeats，配置順序輪轉），再每配置 5 品質樣本。CPU quality 使用不合法 ICD 路徑明確重現 CPU fallback。`TASK_ROOT` 只是 scratch 位置，不能用系統的 HOME 等變數代替。
5. host `python3 temperature_smoke.py`：CPU fallback／Vulkan／CUDA 的原 F16/Q5、3 clips、temperature=0.4、每 clip 一次且固定 NumPy seed。它寫獨立 `temperature04.jsonl`，不合併到 latency 分布。
6. test container `python /bench/check_encoders.py`：獨立 profiling、checker、embedding 比對。它包含 encoder 自帶 2 秒暖機，profile timings 不當 latency。
7. 重跑 `analyze.py` 與 `audit_results.py`。`capture_environment.py` 用來補模型 hash／版本，且會對 `promoted-reproduced-{frontend,backend}.onnx` 與實際 promoted models 核對，需先用 promotion 腳本在這兩個檔名重建。

範例命令前的 `python` 是 container 內命令；從 host 執行時用 `rtk proxy docker exec ... python ...` 或 `rtk proxy python3 ...`。script 中 orchestration 使用 subprocess argv，沒有 shell 字串拼接。

## 檔案與清理範圍

- `results/*.jsonl`：原始逐次資料含 warmups／pilot，分析時明確排除；保留當時版本差異。
- `results/summary.json`、`audit.json`：可重算統計；`schedule.json` 15 個 GPU blocks。temperature smoke 不在 schedule，另有原始檔與 logs。
- `encoder-diagnostics.json`：provider/op 與型別摘要、embedding 相似度。原 ORT traces 很大，隨 scratch 刪除；摘要由保存的 `check_encoders.py` 產生。
- `environment.json`、`gguf-metadata.json`、`q4-archive.json`：版本與 artifact fingerprint。
- `logs/`：公開樣本推論、apt、編譯、Vulkan strace 等原始證據。第一次 CUDA link failure 與後續 retry 皆保留，包含工具輸出的 CR 與行尾空白；不對 raw logs 作格式修剪。`git diff --check` 對作者文件／腳本執行，排除這個 generated logs 目錄。
- `results/cleanup.json`：實際清理命令、返回狀態及正式服務核對；`SHA256SUMS` 對保存資料作完整性檢查。

清理僅限 state 識別的 test／compiler containers、當次新增 image 及專用 scratch（含下載模型、音訊、source/build、pip/apt cache）；不要 prune 整台 Docker，不刪模型 volume、既有 runtime image 或正式服務。先保存 evidence，再停止 test processes／移除其資源，最後核對正式容器 ID／健康狀態、原始模型 hash 與殘留標籤。當次實際 scratch、container IDs 與清理結果見 cleanup.json。
