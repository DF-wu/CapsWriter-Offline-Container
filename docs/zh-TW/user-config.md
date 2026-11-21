# 升級不覆寫的設定（`user/`）

> [文件首頁](README.md) · 繁體中文 · [English](../en/user-config.md)

發行 ZIP 永遠不含 `user/` 資料夾。把自己的設定放在這裡，升級時就能直接把新版
ZIP **解壓覆蓋舊資料夾**：程式檔被替換，`user/` 原封不動。這適用於自訂內容已
放在 `user/` 後的升級。首次從沒有 `user/` 的舊版遷移，請解壓到**另一個新資料夾**，
再於新資料夾執行 `start_client.exe --import-config <舊資料夾>`，從完整的舊資料夾匯入；
匯入前不要覆蓋舊資料夾。

```text
CapsWriter-Offline/
├─ start_client.exe、internal/、core/ …  ← 升級時替換
├─ config_client.py、hot.txt、LLM/ …     ← 隨附預設值，升級時替換
├─ models/                               ← 不在 ZIP 內，保留
└─ user/                                 ← 不在 ZIP 內，保留
   ├─ README.txt
   ├─ config_client.py                   ← 只寫你要改的 Client 設定
   ├─ config_server.py                   ← 只寫你要改的 Server 設定（完整版）
   ├─ hot.txt、hot-rule.txt              ← 你的熱詞與正則規則
   ├─ hot-server.txt                     ← Server 熱詞（完整版）
   └─ LLM/                               ← 你的 LLM 角色（API key 放這裡）
```

## 建立方式

第一次執行 `start_client.exe`（或 `start_server.exe`）時會建立 `user/`，補上缺少的
檔案，且永不覆寫已存在的檔案：

| 檔案 | 第一次啟動建立 |
|---|---|
| `config_client.py`、`config_server.py` | 全部是註解、不改任何值的範本 |
| `hot.txt`、`hot-rule.txt`、`hot-server.txt` | 隨附預設檔的複本 |
| `LLM/` | 隨附角色的複本（僅在 `user/LLM/` 尚不存在時） |

之後請改 `user/` 內的檔案。托盤的「熱詞」與「新增熱詞」對話框已改為開啟／寫入
`user/hot.txt`。

## 設定檔只寫改動的部分

`user/config_client.py`、`user/config_server.py` 只放要改的值，類別與屬性名稱要和
程式資料夾的 `config_client.py`／`config_server.py` 相同；其他值照舊跟隨隨附預設，
包括之後版本新增的設定。

```python
# user/config_client.py
class ClientConfig:
    addr = '192.168.1.20'
    paste_apps = ['WeiXin.exe', 'Telegram.exe', 'Code.exe']
```

```python
# user/config_server.py
class ServerConfig:
    model_type = 'sensevoice'

class Qwen3ASRGGUFArgs:
    llm_use_gpu = False
```

拼錯的類別或屬性會在啟動時提示並略過，因此日後版本移除某個設定也不會讓程式
無法啟動；但檔案若不是合法 Python，會顯示檔名與錯誤並停止啟動。只有你指定的
屬性會改變：隨附檔裡由其他屬性推導出的值（例如由 `ModelPaths.model_dir` 組出的
模型路徑）也要一併明確設定。

## 優先順序

由低到高：

1. 隨附的 `config_client.py`／`config_server.py`。
2. `user/config_client.py`／`user/config_server.py`。
3. 設定視窗儲存的 Client 日常設定
   （`%LOCALAPPDATA%\CapsWriter\client-settings.json`，見[日常設定](../settings.md)）。
4. Server 環境變數（Docker 與 `CAPSWRITER_*`）。

設定視窗的「設定來源」頁會把來自你的檔案的值標示為 `user/config_client.py`。
LLM 角色依檔名合併：`user/LLM/<名稱>.py` 取代 `LLM/<名稱>.py`，其餘隨附角色照常載入。

`CAPSWRITER_USER_DIR` 可把這個資料夾改到別處（例如同步碟）。Docker Server 不使用
`user/`，請依 [Compose 指南](../compose.md)用環境變數與掛載檔案設定。

## 升級

以下方式適用於自訂內容已放在 `user/` 後；首次舊版遷移請依下方的新資料夾匯入步驟。

- **原地覆蓋**：結束 Client 與 Server，把新版 ZIP 解壓覆蓋舊資料夾，再重新啟動。
- **新資料夾**：把新版解壓到別處，再把 `user/`（完整版另含 `models/`）搬進去。

### 從沒有 `user/` 的舊版升級

舊版的修改寫在程式檔裡，升級會被覆寫。請把新版解壓到**新資料夾**，並從舊資料夾
匯入一次：

```powershell
.\start_client.exe --import-config "D:\CapsWriter-Offline-old"
```

匯入只把與新版預設值不同的設定寫入 `user/config_client.py`、`user/config_server.py`；
`hot.txt`、`hot-rule.txt`、`hot-server.txt` 與 LLM 角色只複製與隨附版本不同者。已在
`user/` 改過的檔案不會被覆寫（會列為「保留」），無法自動轉換的值會列出來請你手動
設定。僅 Client 版不匯入 Server 設定。
