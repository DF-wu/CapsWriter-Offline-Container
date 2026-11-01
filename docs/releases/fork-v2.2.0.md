# fork-v2.2.0 — 發版草稿與檢查表

狀態：**準備中，尚未發布**。最新正式版為
[fork-v2.1.0](https://github.com/DF-wu/CapsWriter-Offline-Container/releases/tag/fork-v2.1.0)
（2026-10-31）。本草稿不代表 master 已合併、tag 已建立或 image／Windows artifacts
已發布。Canonical repo：`DF-wu/CapsWriter-Offline-Container`。

## 可使用的發版說明

fork 2.2.0 新增預設關閉的 Tesla P4 opt-in runtime，沿用上游 v2.6／`84912d5`、
Python 3.10–3.12 及 llama.cpp b7798 ABI。設定：

```dotenv
CAPSWRITER_TESLA_P4_OPTIMIZATION=true
```

- 未開啟時維持原 CPU／Vulkan 選擇，不註冊 bundled CUDA runtime。
- 開啟才使用固定 b7798／CUDA 11.8／SM61 decoder；CPU、低顯存 preset 與明確
  CPU／Vulkan 設定優先。實際 CUDA 裝置必須是 Tesla P4；其他卡不冒充實測支援。
- 權重保留原 FP16 ONNX encoder＋Q5_K_M GGUF。archive／DSO hashes 驗證與 atomic
  promotion、CUDA device guard、載入失敗拒絕 CPU fallback 都已整合。
- runtime 在 Docker build stage 內建置；使用者 host 不需要安裝 CUDA compiler，
  不變更時脈、功耗、風扇或權限。NVIDIA boost 是獨立且仍預設關閉的功能。
- fork runtime／package metadata 為 `2.2.0`；health/readiness 的 `fork_version`
  自動反映該版本，上游 `version: "2.6"` 維持原值。

使用、關閉、Portainer 注入、GPU 可見性及原 writable-layer hotfix 升級注意事項：
[Tesla P4 指南](../tesla-p4.md)。

### 效能證據範圍

歷史 P4 實測將原部署 CPU decoder fallback 的 HTTP p50，由 2／5／10 秒音訊的
1.255／2.396／4.150 秒降為 0.311／0.506／0.790 秒；五份 greedy 正規化輸出一致。
詳見[原始證據](../reports/p4-cuda-runtime-deployment-20261101.md)。這不是跨硬體保證，
也不是新版正式 image 的重新量測。GPU 顯存約增加 1.7 GiB；Immich 滿載、長時間
壓力與客戶端 release-to-screen 尚未驗收。既有 HTTP temperature 傳遞限制保留。

## 本候選版驗證記錄

本輪測試只使用免洗容器，未修改 host 套件、現行 CapsWriter／Immich 或 Portainer。
具體命令、結果與原生 P4 smoke 見 [2.2 驗證紀錄](validation-fork-v2.2.0.md)。

| 發版參照 | 狀態 |
|---|---|
| 候選版本 | `2.2.0`；預定 tag `fork-v2.2.0` |
| 同 repo PR | 由此分支建立，URL 記錄於 PR 與交付訊息 |
| 最終 merge SHA／tag target | PR 合併並驗證後填入，不能使用舊 release SHA |
| CI／portability／Windows package | 以 PR／最終 merge SHA 的實際 workflow 結果為準 |
| Server／Web immutable images、digest、SBOM／provenance | 尚未建置發佈；master 合併後的 publish workflow 提供 |
| GitHub Release | 僅草稿，最後需更新為已驗證的 merge SHA 與 artifacts |

## 正式發布前

- [ ] 同 repo PR 通過 review、CI、Linux／Windows portability 與 Windows package。
- [ ] 在最終 merge SHA 重跑 release gate，記錄每個 workflow URL 與 SHA。
- [ ] 建置並驗證 source server image 的 P4 開啟／關閉、CPU／Vulkan／low-VRAM 模式。
- [ ] 記錄該 SHA 的 server／Web `sha-<full-sha>` image tags、digest、SBOM／provenance。
- [ ] 附上 Windows ZIP、SHA256SUMS 與 relocation self-check 證據。
- [ ] 更新 draft release target 與 artifacts，確認 tag 指向同一個已驗證 merge SHA。
- [ ] 發布前確認影響範圍：合併 master 會自動啟動 image 發布與 guarded latest promotion。
- [ ] 獨立處理 Portainer 部署，保留全部注入參數、model/log volumes 與 rollback 資料。

這次授權範圍是 feature、in-repo PR 與 2.2 發版準備；尚未執行 master merge、
正式 Release 發布或現行服務 recreate。
