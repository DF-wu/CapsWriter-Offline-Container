# fork-v2.2.0 候選驗證紀錄（2026-11-01）

本輪驗證在免洗 Docker container 內執行，沒有安裝 host 套件。現行 CapsWriter、
Immich 與 Portainer 設定未變更。候選分支：`release/v2.2-tesla-p4-opt-in`。

## 已執行

- TDD：新測試先證實預設會誤選 CUDA、停用時仍讀 bundle、無旗標可直接選 CUDA、
  缺少 bundle／GPU 會回落、錯誤旗標不被拒絕，修正後 Docker server suite 82 項通過。
- 預設／false 保留 Vulkan，true 啟用 CUDA，CPU／low-VRAM／Vulkan override
  保留，未註冊 bundle 不影響原模式；CUDA 無旗標及非 P4 硬體會明確拒絕。

| 驗證 | 本輪實際結果 |
|---|---|
| Docker server suite | 82 tests，無失敗／skip |
| `scripts/tests` | 504 tests，19 skips，無失敗；可選桌面依賴／平台環境未齊，非完整 lifecycle 驗收 |
| 嚴格 API contract | 191 tests，無失敗／skip，包含 fork metadata 與上游版本契約 |
| Compose／Dockerfile focused suites | 19 tests，無失敗／skip |
| 文件連結與中英對應 | 81 Markdown files 通過 |
| 上游 divergence guard | 通過，80 upstream-tracked files；未新增未登記偏離 |
| 真實 Tesla P4 smoke | flag 未設定／false／true 三種皆通過（詳下方） |
| 唯讀獨立審查第一輪 | Critical／Important／Minor 均 0；後續覆蓋保護及最終狀態依 PR 記錄 |

### 命令與免洗邊界

基底為已存在的原 image ID：
`sha256:b413de423317afd354ddeaa17898fabbd4ae03bcb8040f261dc029ec451f5ea5`。
每個容器皆 `--rm --label capswriter.task=p4-flag-2.2`，workspace 掛載唯讀，
`PYTHONDONTWRITEBYTECODE=1`；沒有 model／log production volume 掛載。
一般 server、scripts 與 API 驗證在容器內執行：

```sh
python -m unittest discover -s docker/server/tests -q
python -m unittest discover -s scripts/tests -q
python -m unittest scripts.tests.test_compose_config scripts.tests.test_dockerfiles -q
python -m pip install --disable-pip-version-check --require-hashes --only-binary=:all: -r requirements/api-test.lock
python scripts/verify_api_contract.py
python scripts/check_docs.py
python scripts/check_upstream_divergence.py --require-base
```

Git 僅安裝在做 scripts／docs guard 的免洗容器；API 鎖定依賴僅安裝在另一個免洗容器。
首次直接使用原 server image 的 API suite 因缺少 `httpx` 失敗，補齊鎖定測試環境後
嚴格 runner 191 項全通過。第一次安裝嘗試因 appuser 對 `/opt/venv` 無寫入權而失敗，
改為僅該測試容器使用 root；正式 image 仍 `appuser`。另一次 focused 命令誤寫
`test_dockerfile_pins`（不存在），改為實際 `test_dockerfiles` 後 19 項全通過；
同次文件與 divergence guard 已先通過。這些失敗未被列為通過結果。

### 真實 P4 native smoke

以 `--gpus all` 暴露 P4，將先前驗證 bundle 唯讀掛載至
`/opt/capswriter/llama-cuda`，執行本分支的 entrypoint：

```sh
sh /workspace/docker/server/entrypoint.sh python /workspace/docker/server/tests/smoke_p4_runtime.py
```

- 未設定旗標：`selected llama backend: vulkan`，CUDA asset／manifest 未註冊。
- `CAPSWRITER_TESLA_P4_OPTIMIZATION=false`：同上。
- `CAPSWRITER_TESLA_P4_OPTIMIZATION=true`：`selected llama backend: cuda`；
  在容器臨時目錄驗 hash、解壓四份 DSO、搬移載入並通過 ctypes native guard。
  原生日誌：`Device 0: Tesla P4, compute capability 6.1, VMM: yes`，
  registry 為 `CUDA0, CPU`，device description 確認為 `Tesla P4`。

只驗證選擇、artifact 完整性、可搬移載入與真實裝置辨識，沒有載入 ASR 權重、
沒有重跑本輪效能或模型品質。未設定／false 的 smoke 沒有重新驗 Vulkan tensor 執行。
CPU／Vulkan overrides 在旗標開啟時也不讀取未使用的 CUDA metadata，新增測試先重現
損壞 bundle 誤阻擋啟動，再完成修正（red → green）。

### 發版範圍

PR workflow 結果以 GitHub 上該 head SHA 的實際狀態為準；本機驗證不是完整
Linux／Windows CI、Windows ZIP、source release image、GHCR digest 或長時間 ASR
壓力驗收。正式 merge SHA、tag 與 artifacts 待後續發布 gate。

未執行的完整 release image／Windows artifacts 不列為通過；歷史 benchmark 另見
[2026-11-01 部署紀錄](../reports/p4-cuda-runtime-deployment-20261101.md)。
