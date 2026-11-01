# 單一 Compose 部署 / Single-file deployment

部署入口只有 [`docker-compose.yml`](../docker-compose.yml)，參數集中在
[`.env.example`](../.env.example)。Server service 名稱、原 model／log／settings
volume keys 都保留；不需要組合其他 YAML。適用 Docker Standalone／Compose，
不代表 Docker Swarm 的 GPU／profiles 部署已支援。

## 基本啟動

```sh
cp .env.example .env
cp hot-server.example.txt hot-server.txt
docker compose config --quiet
docker compose up -d
```

預設只啟動非 root Server，沒有 GPU 裝置要求，P4 優化、HTTP API、設定管理與
clock boost 都關閉。WebSocket 6016 與 HTTP 6017 的 host mapping 預設限 loopback；
6017 即使 API 關閉也會被 Docker 保留，但服務不監聽。若該 host port 已被占用，
請改 `CAPSWRITER_HTTP_API_PORT`，或在唯一 Compose 中移除未使用的 HTTP mapping。

Default startup runs only the non-root server. It requires no GPU, does not
enable P4/clock boost/API/settings, and publishes on loopback. The HTTP host
port is reserved even when its listener is disabled; change its port or remove
that mapping if unused and already occupied.

## Runtime 支援範圍

單一 Compose 統一的是部署入口；公開 server image 目前仍是 CUDA11.8／ORT1.18
與 llama b7798。P4 flag 開啟才用已驗證 SM61 CUDA decoder，其餘維持原 CPU／Vulkan
選擇。Intel／AMD 的 DRI 區塊提供 Vulkan decoder 存取，沒有自動安裝 OpenVINO、
SYCL、MIGraphX 或 HIP；encoder 仍需各自支援的 provider。

後續廠牌路由方向是 NVIDIA 優先相容 CUDA、Intel 原生 GPU 路徑、AMD 原生或明確
選用的社群 ROCm，auto 模式最後經完整模型驗證回 CPU。這需要不同 image／adapter
及硬體驗收，本 PR 尚未實作，詳見[一手來源研究](reports/gpu-vendor-qwen-runtime-policy-20261101.md)。

One Compose file does not make vendor runtime packages interchangeable. The
current image and P4 opt-in retain their existing support boundaries; the vendor
selection policy in the linked research is a follow-up, not a new support claim.

## Qwen prefill 與音訊長度 / Tuning

Batch／micro-batch／context 的容量與取捨見 [Qwen 調校指南](qwen-tuning.md)。
目前 Server 的 `CHUNK_SIZE=80` 會截斷超過上限的單次音訊，並非自動分段；
`N_BATCH` 過小可能讓長 prompt／音訊中止。先保留原基準，再逐項比較 latency、
顯存與完整輸出；變更 env 後須以原 Stack 配置重新部署。

See the guide's English summary before changing prefill settings: logical batch
capacity, physical micro-batches and context are separate limits. Alternatives
are trial configurations, not newly benchmarked P4 presets.

## NVIDIA／Tesla P4

在 `docker-compose.yml` 的 `BEGIN NVIDIA`／`END NVIDIA` 之間，取消每行開頭
`# ` 的註解，保留原縮排。這會啟用原本的 NVIDIA device reservation。不要同時
啟用 DRI 區塊；CPU 主機全部保持註解。`CAPSWRITER_GPU_DEVICE_COUNT` 可設 `all`
或正整數，host 仍需已配置的 NVIDIA Container Toolkit。

P4 另在 `.env` 設定：

```dotenv
CAPSWRITER_TESLA_P4_OPTIMIZATION=true
```

```sh
docker compose config --quiet
docker compose up -d capswriter-server
```

GPU block 是 Docker 的裝置存取；P4 flag 是應用程式的 runtime 選擇，兩者都有
才會使用 P4 CUDA。P4 不需要 root／NVIDIA_BOOST，詳見 [P4 指南](tesla-p4.md)。
只有 flag、沒有裝置會拒絕 P4 CUDA 啟動；其他 NVIDIA 卡保持 P4 flag 關閉。

Uncomment the NVIDIA block in the same file and leave DRI commented. This
preserves the existing GPU reservation mechanism; an application env flag alone
cannot grant GPU access. Only Tesla P4 opts into the separate P4 runtime flag.

## Intel／AMD `/dev/dri`

取消同檔 `BEGIN DRI` 區塊的註解，NVIDIA 區塊保持註解，並設定 host 實際 GID：

```sh
stat -c '%n %g' /dev/dri/renderD* /dev/dri/card*
```

將結果填入 `.env` 的 `CAPSWRITER_DRI_RENDER_GID`／`CAPSWRITER_DRI_VIDEO_GID`，
再用相同 `docker compose up -d` 命令。裝置節點、host driver 與權限仍須可用。

Uncomment DRI instead of NVIDIA, then set the actual host render/video GIDs.
Docker checks devices before starting the entrypoint; CPU fallback cannot repair
a missing host device or NVIDIA runtime. This is why hardware blocks stay
explicit rather than attaching every device by default.

## 模型與儲存

切換 Fun-ASR 不需另一份檔案：

```dotenv
CAPSWRITER_MODEL_TYPE=fun_asr_nano
```

預設 `CAPSWRITER_MODELS_SOURCE=capswriter-server-models` 保留既有 named volume。
若要 host 管理，先建立 writable 目錄，確認 image 的 `appuser` UID／GID，才改成：

```dotenv
CAPSWRITER_MODELS_SOURCE=./models
```

絕對 host 路徑也可。換 storage source 不會搬移模型；先備份並複製／確認完整性，
舊 volume 保留。其他 named volume 名稱不要任意填入，除非也在同檔 top-level
`volumes` 宣告。Bootstrap 需要整個目錄可寫，不能只讓 model 檔可讀。

The default model source keeps the existing named volume. A relative (`./models`)
or absolute bind source is operator-managed: prepare permissions first. Changing
the source does not migrate data. Preserve old storage and the Compose project
name, so existing volumes remain attached.

## Web 與設定管理

在 `.env` 設定：

```dotenv
CAPSWRITER_HTTP_API_ENABLE=true
CAPSWRITER_HTTP_API_KEY=replace-with-your-own-token
CAPSWRITER_SETTINGS_ENABLE=true
CAPSWRITER_HTTP_API_CORS_ORIGINS=http://127.0.0.1:8080,http://localhost:8080
```

```sh
docker compose --profile web up -d
```

或把 `COMPOSE_PROFILES=web` 寫入 `.env`，之後仍用 `docker compose up -d`。
只要 Web 指向另一台 Server，可單獨 `docker compose --profile web up -d capswriter-web`。
Web profile 本身不自動啟用 API、不注入公開 token；在 Web UI 輸入 token。
遠端瀏覽器的 API URL 要可達，跨 origin 需 CORS，遠端麥克風需 HTTPS。

`/app/settings` 使用持久化 named volume。每日設定優先序為
**非空 env → 保存 JSON → 預設**。唯一 Compose 保留未設定時的原預設；要讓 GUI
接管欄位，必須在 `.env` 明確清空，例如 `CAPSWRITER_MODEL_TYPE=`，再 recreate。
刪除整行會回到 Compose 預設，並非解鎖 GUI。完整欄位見 [設定說明](settings.md)。

Use the `web` profile for the optional published Web image. API and settings
management remain explicit env switches; the settings endpoint requires an API
key. Explicitly empty daily variables delegate to saved JSON; omitted variables
retain Compose defaults. Set `CAPSWRITER_WEB_IMAGE=capswriter-web:local` and use
`docker compose --profile web build capswriter-web` only when building Web locally.

## 選用 NVIDIA clock boost

保留在同檔 `BEGIN NVIDIA_BOOST` 區塊，預設全部註解。只有需要時才取消註解，
搭配 NVIDIA 裝置區塊、`CAPSWRITER_GPU_BOOST_ENABLED=true` 與支援的頻率。
該區塊沿用 root、指定 capabilities 與 150 秒停止寬限；不因整併而預設開啟。
完整限制與回復 appuser 的所有權注意事項見 [boost 指南](nvidia-boost.md)。

## 從舊多檔／Portainer 配置遷移

- 主檔與 example：都改用唯一 `docker-compose.yml`。
- gpu／igpu：分別搬到同檔 NVIDIA／DRI 註解區塊。
- web：改用 `web` profile。
- settings：改為 HTTP／SETTINGS env 開關，volume 已整合；GUI 欄位明確留空。
- models-bind：改用 `CAPSWRITER_MODELS_SOURCE`。
- fun-asr：改用 `CAPSWRITER_MODEL_TYPE`。
- nvidia-boost：搬到同檔 NVIDIA_BOOST 註解區塊。

Portainer 只指定這份 Compose，移除舊 additional Compose paths／`COMPOSE_FILE`；
保留原 Stack／project 名稱、所有注入 env、API key、網路／port、host paths 及
models/logs/settings volumes。將舊自訂 GPU／權限設定核對搬入同一檔，不能只
更換檔名而遺失裝置。若 Portainer 的 profile 選項不可用，可在這一份檔案移除
Web service 的 `profiles: [web]` 行來一同啟動，不需要新增 YAML。

先檢查 `docker compose config` 的結果再於維護時段 redeploy；`restart` 不套用新
env／mounts。不要用 `down -v` 遷移。本次原始碼整併不會自動修改已運行的 Stack。

For Portainer, keep the same stack/project identity, injected settings, GPU
mapping and volume sources. Use only this Compose path and remove old overlay
paths. Review the rendered configuration before redeploying; do not use
`down -v`. Existing services are not automatically upgraded by source changes.
