# v1 上游更新與遷移 / Upstream refresh and migration

[繁體中文 README](../readme.md) · [English README](../README.en.md)

## 這次更新的邊界

v1 整合上游至 `84912d5`，接受必要重構及模型更新，保留 fork 的 Linux／Docker
Server 與 HTTP API。Windows 為第一支援 Client 平台，主要流程是快捷鍵錄音 →
遠端 axolotl 辨識 → 輸入目前視窗。此文件不表示 axolotl 正式服務已升級。

v1 的目錄架構由 `util/` 移至 `core/`；fork Server 功能位於 `fork_server/`。
舊 `core_client.py`／`core_server.py` 已移除。自訂程式若直接 import 舊內部模組，
須改用目前實作；外部整合應優先透過文件化的 WebSocket／HTTP 介面。

| 用途 | 目前入口 | 設定來源 |
| --- | --- | --- |
| Windows 桌面 Client | `python start_client.py` | `config_client.py`、Client 熱詞與角色檔 |
| 原生 fork Server | `python start_server_universal.py` | `config_server.py`，另讀取入口支援的環境變數 |
| Linux Docker Server | container entrypoint → `start_server_docker.py` | Compose environment、`fork_server` runtime 與 `/app/models` |
| 上游桌面 Server | `python start_server.py` | 上游風格 `config_server.py`；fork HTTP 設定請用 universal 入口 |

原生環境保留 Python 3.10–3.12 相容性，建議以乾淨的 Python 3.12 環境安裝。
固定 Docker runtime 保持 Python 3.10。不要把上游 Python 3.14 tooling 的存在
理解為所有原生依賴已在 3.14 通過驗證。

### 刻意保留的上游差異

原生 llama runtime 與 ctypes binding 維持相容的 `b7798` ABI，包括四參數
penalties 呼叫。上游 `b10621` binding 與此 binary 不相容，不能只換其中一邊。
這次延後該原生引擎遷移，之後必須將 binary／binding／封裝同步更新，並重新跑
真實模型與各 backend 驗證。這是已知整合例外，不是宣稱與上游逐檔完全一致。

## 升級前保留可回復版本

1. 記錄目前 commit／release tag、實際 container image ID、Compose project 名稱、
   port、volume／bind mount 路徑，以及使用的模型與 backend。
2. 備份 `.env`、Compose override、`config_client.py`、`config_server.py`、
   `hot.txt`、`hot-rule.txt`、`hot-server.txt` 與自訂 `LLM/` 角色。備份放在 checkout
   外，保護其中的 token／金鑰；不要提交到 Git。
3. 保留 `models/` 或實際模型 mount、Client 錄音／日記，以及服務 log。模型不會
   因本次更新自動刪除；大模型可使用獨立 snapshot／唯讀原始副本，避免測試改寫。
4. 在另一份 checkout／環境測試；不要直接在正在提供服務的目錄覆寫或安裝依賴。

## Windows Client 遷移

使用新的設定模組，逐項搬移舊值；**不要把舊 `config_client.py` 整份蓋回新版**。
尤其檢查以下內容：

| 設定 | 遷移注意事項 |
| --- | --- |
| `ClientConfig.addr`／`port` | 遠端 Server 使用 `axolotl` 或可到達的 IP；addr 不含 `ws://`、路徑或 port。預設 `127.0.0.1` 是 Windows 自身。 |
| `shortcuts`／`threshold` | 保留習慣的鍵位、hold mode、suppress 與 enabled；先在普通文字編輯器確認短按與長按。 |
| `traditional_convert`／`traditional_locale` | 臺灣繁體可設 `True`／`'zh-tw'`；與辨識語言分開設定。 |
| `paste`／`restore_clip`／`paste_apps` | 驗證文字輸入及原剪貼簿保留行為；不同目標程式可能需要不同輸出方式。 |
| `save_audio`／`llm_enabled` | 明確保留自己的錄音儲存與角色功能偏好；自訂角色中的 API key 不可公開。 |
| 分段、熱詞及輸出格式 | 不盲目搬入過時的模型參數；先使用新版預設，再依測試調整。 |

在 PowerShell 的新 checkout 建立環境：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-client.txt
.\.venv\Scripts\python.exe start_client.py
```

Client 連遠端 Server 時不需在 Windows 載入 ASR 模型。音訊／檔案功能所需的
FFmpeg 與桌面依賴，依 [環境依賴說明](环境依赖安装说明.md) 確認。
修改 Python 設定後重新啟動 Client。

## Server 設定與模型遷移

同樣逐項搬移 `config_server.py`，不要覆蓋新版類別結構。Docker 則以 `.env` 與
Compose 傳入的 environment 為主；只編輯 `.env` 中未傳入 container 的變數不會
自動生效。以 `docker compose config` 檢查解析結果，並避免將含金鑰的輸出公開。

目前模型架構包含 Qwen3-ASR、Fun-ASR-Nano、SenseVoice 與 Paraformer。
舊模型 mount 可以保留，但新版 `ModelPaths` 與所選引擎可能需要不同的模型檔案
或子目錄；**保留 mount 路徑不等於所有既有模型可直接重用**。核對所選 backend
的檔名、tokenizer、ONNX 與 GGUF 組合，缺少時由模型 bootstrap／下載流程處理。
預留下載與解壓空間；不要刪除仍供舊版使用的模型。

- 在隔離 checkout 建立獨立 `.env`；沿用必要設定值，但不要複製 production token。
- 若使用 CPU，設 `CAPSWRITER_GPU_DEVICE_COUNT=0` 與
  `CAPSWRITER_INFERENCE_HARDWARE=cpu`；GPU 測試另記錄 driver 與 backend。
- 使用與正式服務不同的 ports、Compose project、image tag 與 model/log storage。
- v1 預設 image 為本機 build 的 `capswriter-offline-v1-local:source`。
  公開 `ghcr.io/df-wu/capswriter-offline-server:latest` 是 v2，不可替代。
- HTTP API 仍為選用。檢查目前 [API contract](HTTP_API.md) 與 runtime 限制，
  不要假設舊版 environment 預設值、安全限制或完整 OpenAI 功能均相同。
- HTTP caller 必須明確傳入 `model=whisper-1`；未知欄位、其他 model ID、
  `stream=true`、diarization 與 logprobs 會被拒絕。`prompt` 現在傳入 context，
  `language` 傳入引擎作為提示；實際效果依 backend 而異。
- `balanced`／`quality` Qwen preset 請改為 `default`。基礎 Compose 不再
  請求 GPU，NVIDIA／Intel／AMD 裝置須使用對應 GPU override。

## 驗收與回復

本次實際通過的測試與未驗證範圍見 [驗證紀錄](v1-refresh-validation.md)。

至少記錄：設定驗證、容器 build／啟動、所選模型載入、已知中英文音訊的
WebSocket 與啟用時的 HTTP 轉錄、取消及斷線後的恢復。Windows 另需實機驗證
啟動／退出、系統匣、快捷鍵、麥克風、剪貼簿與文字注入。Linux 單元測試及容器
測試不能代替 Windows 桌面驗證。

若出現退步，停止並移除**隔離測試服務**，回到保留的 commit／image 與對應設定、
模型 mount；不要拿新版設定／模型佈局覆蓋舊版後期待直接回復。本次工作沒有
升級正式服務，因此無須切換 axolotl 正式部署。

清理僅限此次建立的容器、image、network、volume、臨時虛擬環境及 build cache。
保留驗證紀錄、仍使用中的 branch／PR、既有模型、設定、正式 log 與備份。
不要執行無範圍的 Docker prune 或刪除未合併且仍有工作內容的 branch。

## English migration reference

The v1 refresh integrates upstream `84912d5` with the fork's hardened server,
API, and Docker runtime. It intentionally keeps the paired `b7798` native llama
ABI; upstream `b10621` bindings need a coordinated binary/binding migration.
Native Python 3.10–3.12 compatibility remains, using 3.12 for new setups; the
pinned Docker runtime stays on 3.10. Python 3.14 is not release-qualified here.

Back up configuration, hotwords, custom `LLM/` roles, recordings, model mounts,
and the previous source/image identity outside the checkout. Reapply individual
values to the new configuration modules rather than replacing them with old
files. The implementation has moved from `util/` to `core/`; old
`core_client.py` and `core_server.py` imports no longer work.

Use `start_client.py` on Windows. In `ClientConfig`, set `addr='axolotl'` (or a
reachable server IP), `port='6016'`, and optionally `traditional_convert=True`,
`traditional_locale='zh-tw'`. The address is a hostname, not a WebSocket URL.
Restart after changing Python configuration. A remote client does not need a
local ASR model. Native fork server operation uses `start_server_universal.py`;
Docker uses `start_server_docker.py` and the Compose environment.

Keep existing model mounts, but verify the new model layout and required files;
new engines may require additional downloads. Do not delete old model files
needed for rollback. Build the v1 local image; the public v2 `latest` image is
not a v1 substitute. Test with isolated ports, storage, project and image names
on axolotl, leaving existing services untouched. CPU-only hosts should set
both `CAPSWRITER_GPU_DEVICE_COUNT=0` and `CAPSWRITER_INFERENCE_HARDWARE=cpu`.

Report unit/container results separately from real-model and Windows desktop
results. Linux tests do not qualify hotkeys, microphone, clipboard, or text
injection on Windows. To roll back an isolated trial, stop only that trial and
reuse the saved source/image with its matching configuration and models. Clean
only resources created for the trial and retain the test report.
