# Web Console Client

> [文件首頁](README.md) · [English](../en/web-console.md) · [Server 與 Client 分工](server-and-clients.md)

Web Console 是 **Client**，不是 ASR Server。Static file 由 `8080` 提供；browser
再把音訊送到 CapsWriter Server 選用的 HTTP API `6017`。Browser 不會載入辨識
model。

![CapsWriter Web Console data flow：browser 輸入經 HTTP API 送到本機 ASR worker](../assets/web-console-architecture.svg)

## 分工

| Web Console 負責 | ASR Server 負責 |
|---|---|
| 麥克風／選檔、播放、UI state、history、download | Model download／load、FFmpeg decode、inference、hotword、readiness |
| Browser 本機 Web Speech TTS | 產生 speech-to-text 結果 |
| Page memory 內的 API root／key | 驗證 authentication、request limit、queue／deadline |

Browser TTS 是本機 browser／OS 功能，不是 Server TTS endpoint。

## 1. 準備 Server

啟用 HTTP 並設定 exact browser origin：

```dotenv
CAPSWRITER_HTTP_API_ENABLE=true
CAPSWRITER_HTTP_API_KEY=replace-with-a-long-random-token
CAPSWRITER_HTTP_API_PUBLISH_HOST=127.0.0.1
CAPSWRITER_HTTP_API_PORT=6017
CAPSWRITER_HTTP_API_CORS_ORIGINS=http://127.0.0.1:8080,http://localhost:8080,http://127.0.0.1:5173,http://localhost:5173
```

`8080`、`5173` 分別是 production container 與 Vite development origin；沒有
使用的 origin 應移除。

取消 `docker-compose.yml` 內 `ports:` 下 HTTP mapping 的註解，重建 Server 並確認
readiness：

```bash
docker compose up -d --force-recreate capswriter-server
curl http://127.0.0.1:6017/health
curl http://127.0.0.1:6017/ready
```

## 2. 啟動 production Web Client

Build 本機 static image，並只發布在 loopback：

```bash
docker compose -f docker-compose.web.yml up -d --build capswriter-web
curl http://127.0.0.1:8080/health
```

開啟 `http://127.0.0.1:8080`，確認 API root 是
`http://127.0.0.1:6017`，再把 Server token 輸入遮罩 key 欄位。

`CAPSWRITER_WEB_API_KEY` 會寫入公開 `/config.js`，建議保持空白。若 deployment
刻意發布 default key，還必須設定 `CAPSWRITER_WEB_ALLOW_PUBLIC_API_KEY=true`，並
接受所有能載入頁面的 browser 都可讀取該 key。

## 開發

```bash
cd client/web
npm ci --no-audit --no-fund
npm run dev
```

開啟 `http://127.0.0.1:5173`，把 API root 設為
`http://127.0.0.1:6017`，再輸入 Server token。如果 Server CORS allowlist 原本
沒有 `5173` exact origin，加入後必須先重建 Server 才能測試。

若只做沒有 model Server 的 UI 測試，可在另一個 terminal 執行
`npm run mock-api`。它只回固定資料，不是真實轉錄證據。

驗證：

```bash
npm run verify
npm run browser-smoke
```

## Browser 安全與隱私

- 麥克風需要 localhost 或 HTTPS，以及 browser／OS permission。
- `localhost` 與 `127.0.0.1` 是不同 CORS origin；必須允許頁面實際使用的 origin。
- 手動輸入的 API key 只留 page memory，不會存入 history。
- Settings 與最多 20 筆明文 transcript／raw history 使用 `localStorage`；共用電腦
  使用後應清除 site data。
- Download filename 會清理 path separator、control character 與 reserved name。
- Diagnostic／transcription fetch 拒絕 redirect，避免 private audio 與 Bearer key
  被重送到其他 origin。

## 常見問題

| 症狀 | 檢查 |
|---|---|
| Web page 開啟但 diagnostics 失敗 | Server HTTP `6017`、API root、key、`/ready` |
| CORS error | `CAPSWRITER_HTTP_API_CORS_ORIGINS` 是否包含 exact page origin |
| 麥克風 unavailable | Localhost／HTTPS、browser permission、OS permission |
| TTS 無聲 | Browser／OS voice 與 autoplay／audio policy |
| Upload 回 `413` | Server `max_upload_mb` 與所選檔案大小 |

Network exposure、runtime variable、upgrade 與 rollback 請接著讀
[部署](deployment.md#web-console-profile)及[支援與安全](support-security.md)。
