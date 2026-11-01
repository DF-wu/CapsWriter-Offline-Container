# Tesla P4 CUDA runtime 優化與部署紀錄

2026-11-01，issue [#18](https://github.com/DF-wu/CapsWriter-Offline-Container/issues/18)，分支 `perf/p4-cuda-sm61-runtime`。

公開互動報告：[P4 CUDA 升級實測](https://aionc.dfder.tw/raw/BWZMEN6EYF5QyQq/index.html)。

已將 **llama.cpp b7798 / CUDA 11.8 / SM61 decoder** 套用到原 CapsWriter 容器。沿用原 FP16 ONNX encoder 與 Q5_K_M GGUF，沒有替換或重新量化模型。三份權重 SHA-256 完全一致。

| 音訊 | 原服務 HTTP p50 | 已部署 CUDA HTTP p50 | 速度倍率 | 延遲減少 |
|---|---:|---:|---:|---:|
| 2 秒 | 1.2548 s | 0.3112 s | 4.03× | 75.2% |
| 5 秒 | 2.3956 s | 0.5057 s | 4.74× | 78.9% |
| 10 秒 | 4.1497 s | 0.7902 s | 5.25× | 81.0% |

原服務 decoder 處於 CPU fallback：Vulkan 未找到裝置，encoder 仍使用 CUDA。每長度每方案 3 次正式 HTTP 請求、另 1 次暖機，Immich 在計時期間暫停。時間包含 HTTP、音訊處理與推論，未包含客戶端放開按鍵到顯示文字。三個方案依序測量，樣本小且有時脈、溫度變化。

先前修好 EGL 的 Vulkan 與 CUDA 同場引擎測試，10 秒 p50 分別為 0.8649 s、0.5831 s，CUDA 少約 32.6%。它與此處 HTTP 測量範圍不同。[先前實驗](p4-exclusive-container-experiments-20261101.md)

## 修改與驗證

- 固定原始碼 commit 與 tar SHA-256，在免洗 CUDA 11.8 容器編譯 SM61；確認 cubins、可重定位 `$ORIGIN`、真實 driver 載入及不含 stub 路徑。
- 新增可驗證的 bundled CUDA archive 與每份函式庫 manifest，沿用原 downloader 的驗證及原子替換流程。
- NVIDIA 自動選 CUDA，保留 CPU、Vulkan 與低顯存 preset；選定 CUDA 後啟動失敗會中止，禁止不知情地以 CPU 提供服務。三個 inference package 會傳播 CUDA 載入錯誤。
- 實際啟動確認 `CUDA0`、29/29 layers、CUDA0 model/KV/compute buffer，模型 buffer 1193.93 MiB。
- 獨立免洗容器中，原 runtime CPU decoder 與新 CUDA decoder 使用相同參數及 greedy temperature=0，五份完整音訊的正規化文字 **5/5 相同**。
- 舊 HTTP worker 實際解碼使用預設 temperature=0.4，雖然 API 接受 temperature=0。HTTP 文字本來就存在隨機性；本次記錄該限制，沒有修改原行為。品質一致性以直接引擎 greedy 測試判定。
- 106 項相關回歸測試通過；全新清理後檢查確認部署程式、runtime 完整性、模型與設定雜湊及兩個服務 readiness。

正式採納的一輪為 **52 次推論**：36 次 HTTP 延遲（27 正式＋9 暖機）、10 次 greedy 品質、5 次正式服務完整音訊 smoke、1 次 Immich 恢復後的共存辨識。前兩輪分別因日誌讀取位置、HTTP 隨機性中止，均未套用正式 runtime；證據獨立保留，不混入採納數據。

## 服務與環境

CapsWriter 原 ID `d6ab96c622d1…`、原 image ID `b413de423317…`；Immich 原 ID `9b775b9cd582…`、原 image ID `113e61b9de60…`。Config、HostConfig、靜態網路設定指紋全部相同。CapsWriter Docker healthy、HTTP ready；Immich HTTP ping/pong 通過。完整驗證及共存請求通過後才標記 accepted。

所有編譯、安裝套件、音訊轉換與品質測試均在免洗容器內，沒有安裝 host CUDA/compiler/套件，沒有變更 driver、時脈、風扇或功耗。已移除本次剩餘 2 個容器、3 個臨時 image tag、31 筆精確指定的 build cache、暫存與重複 runtime archive cache；70 筆其他 cache 保留，未使用全域 prune。原始模型與其他服務資料保留。

## 優點、成本與後續

這個方案的加速已在此 P4 上測得，沿用現有權重、格式及介面。FP32 encoder 未改善整體表現且多耗顯存，Q6/Q8 和其他完整 runtime 未超越原 CUDA Q5 的整體評估；Tiro v2 也未通過替換條件，因此沒有採用。

熱推論的整卡顯存觀測中位數由約 1249 MiB 增至 2987 MiB。與 Immich 同時工作會競爭顯存、算力與散熱。已驗證恢復與一筆共存辨識，未測 Immich 滿載、長時間服務負載或客戶端 release-to-screen。

目前使用**原容器 writable layer 更新**，普通 restart 會保留。未來若從舊映像 recreate，更新會消失。完整 source Dockerfile 已整合 CUDA 建置與 EGL 依賴；本輪實際建置驗證的是原 deployed base 加 overlay，沒有重建完整 production Dockerfile、沒有合併 master 或替換 GHCR latest。私人回復備份與已驗證 CUDA bundle 保留於本機。

完整程式、逐筆資料、日誌、回復與清理證據：[實驗目錄](experiments/p4-cuda-deployment-20261101/README.md)。可分享頁面：[HTML](experiments/p4-cuda-deployment-20261101/site/index.html)。
