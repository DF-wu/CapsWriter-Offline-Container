# fork-v1.0.0-rc.1 Release notes

> [維護政策](maintenance.md) · [English](../en/release-notes.md) · [專案 README](../../readme.md)

Release candidate 日期：**2026-07-18**。這是隔離 legacy maintenance line 的
source-only GitHub pre-release；不是 v2 release，也不宣稱 real-device／model
qualification 已完成。

## 依角色列出的交付內容

### Server

- Legacy Linux bare-metal 與 Docker Server source。
- Port `6016` 的 WebSocket service。
- Port `6017` 的選用、transcription-only OpenAI 相容 HTTP API。
- Model bootstrap、GPU preference／CPU fallback、health check，以及 persistent
  model／hotword／log path。
- WebSocket、decoded audio、task／context、multipart、authentication、
  cancellation、routing、error／logging bounds。

此 release **不發布 v1 container image**。Compose 從 tagged source build
`capswriter-offline-v1-local:source`；公開 `capswriter-offline-server:latest`
屬於 v2。

### Client

- 保留相容性的 upstream-era Windows desktop source：`start_client.py`、tray、
  hotkey、麥克風、clipboard 與 text injection。
- Desktop Client 以 WebSocket `6016` 連接 v1 Server。

此 release **不附 Windows executable**，也不包含 v2 Web Console、無 GUI CLI、
Textual TUI 或 universal Windows package。

### 外部 API caller

相容 OpenAI SDK／curl caller 可以把 base URL 指向文件列出的 `whisper-1` file
transcription subset。這是 Server interface，不是 bundled v1 Client app；不實作
translation 或完整 OpenAI Audio API。

## 維護變更

- 限制 WebSocket frame、decoded chunk、segmentation geometry、task ID 與 context，
  不改變合法 Client wire format。
- 依 connection 隔離 recognition state，避免相同 task ID 合併或誤送逐字稿。
- 在 multipart parsing 前驗證並檢查 media type；限制 upload 並清理 cancellation。
- 從 error／log 遮蔽 reflected secret 與 private exception detail。
- 修正 subtitle timestamp carry。
- 更新 security-sensitive runtime dependency 與 focused regression。
- 移除 v1 指向 v2 `latest` image 的危險文件／default；v1 改為 build 自己的 local
  source image。

## Automated evidence

- Ubuntu 24.04 與 Windows 2022。
- Python 3.10 與 3.12。
- 每個 matrix leg 都執行完整 maintenance test suite 與 compile checks。
- Compose／entrypoint validation 只在 Ubuntu 24.04／Python 3.10 validation job
  執行。
- Maintenance source baseline 的 duplicate push／PR matrix 全部通過。

## Stable v1 release 前仍需

- Disposable v1 image build 與 cold model bootstrap。
- Model-backed 中文與英文 known-audio transcription。
- Target CPU／GPU backend 與 driver／runtime evidence。
- 真實 Windows desktop launch／exit、tray、hotkey、麥克風、clipboard、FFmpeg、
  model 與 child cleanup。
- 宣稱任何 v1 container image 前，建立獨立審查的 immutable v1 image workflow。

Active development、Web／CLI／TUI／universal package 請使用
[fork v2](https://github.com/DF-wu/CapsWriter-Offline-Container)。
