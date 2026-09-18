# 日常設定 / Daily settings

Windows Client 與 Server 分別保存設定：Client 管理此裝置的收音與輸出，Server
管理所有連線共用的辨識與資源上限。兩者都只在重新啟動對應程式後套用。
儲存不會自動中斷正在進行的錄音或重新啟動服務。

Client and server settings are separate. Saving changes takes effect after
restarting that component; saving never restarts the server automatically.
The existing Python configuration files remain available for advanced options.

## Windows 首次設定 / First Windows setup

使用已建置的程式執行 `start_client.exe --settings`，或依
[開發流程](development.md)安裝 desktop profile 後執行：

```powershell
python scripts/dev.py client --settings
```

1. 主機填 `axolotl` 或其可連線 IP，連接埠預設 `6016`；主機欄不含 `ws://`
   或 HTTP 路徑。按連線測試確認 WebSocket handshake。
2. 選擇麥克風與快捷鍵，設定按住／切換錄音、繁體轉換、貼上與剪貼簿還原。
3. 儲存後開啟／重新啟動 Client，按快捷鍵說話，確認文字輸入目標視窗。
   執行中也可從 tray 的設定選單開啟視窗。

Enter a hostname/IP and WebSocket port, test the connection, select a microphone
and shortcut, then save and restart the client. A successful handshake confirms
connectivity; it does not prove the server model is ready or the microphone works.
Windows microphone, hotkey suppression and foreground typing require a real
Windows desktop acceptance check.

設定預設存於 `%LOCALAPPDATA%\CapsWriter\client-settings.json`；可用
`CAPSWRITER_CLIENT_SETTINGS` 指定其他路徑。只有明確覆寫的值會儲存，其他值
繼續沿用 `config_client.py`。設定檔錯誤會顯示原因，不會靜默覆蓋。

## Server 圖形設定 / Server settings in the Web console

管理 API 預設關閉。使用本次程式碼建置的 Server 與 Web 後，Docker 可在原有
Compose 檔案後加上 `docker-compose.settings.yml`。請先設定 HTTP API key，
並在跨 origin 使用時設定 `CAPSWRITER_HTTP_API_CORS_ORIGINS`。

```sh
docker compose -f docker-compose.yml -f docker-compose.settings.yml -f docker-compose.web.yml config --quiet
docker compose -f docker-compose.yml -f docker-compose.settings.yml -f docker-compose.web.yml up -d
```

如有 GPU／模型 override，放在 `docker-compose.settings.yml` 之前。
設定 overlay 會取代模型 override 的內建選擇；例如 `docker-compose.fun-asr.yml`
中的固定模型會被清空。首次啟動要保持 Fun-ASR，請在 `.env` 明確保留
`CAPSWRITER_MODEL_TYPE=fun_asr_nano`（UI 會鎖定），或先建立含該模型的 Server
JSON 設定。GPU 裝置掛載仍保留。
此設定 overlay 會啟用 HTTP API 與設定管理，將 HTTP port 預設發布至 host
loopback，並以 `capswriter-server-settings` volume 保存 `/app/settings`。
遠端瀏覽器的 API base URL 必須是瀏覽器能連到的 Server 位址；部署方式見
[部署指南](zh-TW/deployment.md)。上述命令是維運操作範例，本次開發不會升級
axolotl 的既有服務。

Settings management is opt-in and requires the HTTP Bearer API key. Apply the
settings Compose override after any hardware/model overrides. It preserves
settings in a dedicated named volume and publishes HTTP to loopback by default.
It clears a model override's literal defaults too: to retain Fun-ASR on first
start, explicitly set `CAPSWRITER_MODEL_TYPE=fun_asr_nano` in `.env` (UI locked)
or seed the settings JSON. GPU device mappings remain intact.
Use a server image built from this change; an older released image does not
provide the settings API. Enter the API URL and key in the Web console, then
open Server settings.

### 環境變數優先 / Environment precedence

優先序為 **非空環境變數 → 儲存的 JSON → 預設值**。從 `.env.example` 複製的
每日參數已有非空值，因此會顯示為環境變數控制、不能在 UI 編輯。要交由 UI
管理，先把 `.env` 中對應的值清空或刪除，再重建容器；進階參數不受影響。

| UI 設定 | 對應環境變數 |
|---|---|
| 模型、硬體、Qwen 模式 | `CAPSWRITER_MODEL_TYPE`、`CAPSWRITER_INFERENCE_HARDWARE`、`CAPSWRITER_QWEN_PRESET` |
| Fun-ASR CPU 執行緒 | `CAPSWRITER_NUM_THREADS` |
| 數字與中英空格 | `CAPSWRITER_FORMAT_NUM`、`CAPSWRITER_FORMAT_SPELL` |
| HTTP 檔案／時長／逾時 | `CAPSWRITER_HTTP_API_MAX_UPLOAD_MB`、`CAPSWRITER_HTTP_API_MAX_AUDIO_SECONDS`、`CAPSWRITER_HTTP_API_TASK_TIMEOUT` |
| HTTP 同時處理／等待數 | `CAPSWRITER_HTTP_API_MAX_CONCURRENT_REQUESTS`、`CAPSWRITER_HTTP_API_MAX_PENDING_REQUESTS` |
| WebSocket 連線／時長 | `CAPSWRITER_SERVER_MAX_WEBSOCKET_CONNECTIONS`、`CAPSWRITER_SERVER_MAX_WEBSOCKET_TASK_SECONDS` |

Nonempty environment values take precedence and lock their fields in the UI.
Clear the corresponding values in `.env` and recreate the container to let
saved settings control them. The overlay removes Compose's implicit daily
defaults without ignoring your explicit environment choices.

介面分別顯示執行中的有效值、下次啟動值及設定來源。儲存後顯示待重啟；
請選擇沒有辨識作業的時段重新啟動 Server。多個視窗同時修改會要求重新讀取，
避免舊頁面覆蓋較新的設定。模型切換前需確認資產可用；`gpu` 仍沿用既有
「優先 GPU，失敗可回退 CPU」政策。

The UI distinguishes current and next-start values, reports validation errors,
and detects conflicting edits. Restart during an idle period after saving.
Model selection still requires suitable assets; GPU selection follows the
existing fallback policy. API keys and advanced engine settings are not exposed
or edited through this endpoint.

### Source / Windows native server

明確匯出 `CAPSWRITER_SETTINGS_ENABLE=true`、`CAPSWRITER_SETTINGS_PATH` 與
HTTP API key，並啟用 HTTP API，再啟動 `start_server_universal.py`。
路徑必須可寫。原生模式不提供容器硬體／Qwen preset 選項，未覆寫的模型、
格式化與 Fun-ASR 執行緒沿用 `config_server.py`。

Native servers use the same authenticated API with an explicitly configured
writable settings file. Container-only hardware controls are omitted. Disable
`CAPSWRITER_SETTINGS_ENABLE` to disable remote management; saved settings still
apply while `CAPSWRITER_SETTINGS_PATH` is configured.

## 備份與還原 / Backup and recovery

保留 Client JSON、Server settings volume、原有 Python 設定與 `.env`。
Server JSON 格式為 `{"version": 1, "values": {}}`；Client 為
`{"version": 1, "overrides": {}}`。要回到進階設定／預設值，可先備份 JSON，
將對應 values／overrides 清空，再重新啟動。不要刪除模型或錄音來重設設定。

Back up the JSON files along with your advanced configuration and environment.
Reset overrides after a backup and restart the affected component. Model and
recording data are independent and should be retained.
