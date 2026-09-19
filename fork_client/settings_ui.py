"""Native daily settings editor; runs in its own process, never the tray thread."""
from __future__ import annotations

import asyncio
import copy
import json
import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from .devices import device_label, input_device_choices
from .settings import (
    SettingsError, effective_settings, load_overrides, probe_connection,
    save_overrides, settings_path, update_overrides, validate,
)


class SettingsWindow:
    def __init__(self, root: tk.Tk, config):
        self.root = root
        self.config = config
        self.path = settings_path()
        self.saved = False
        self.pending = queue.Queue()
        self.closed = False
        self.widgets = {}
        self.variables = {}
        self.errors = {}
        self.touched = set()
        self.device_choices = {}
        self.shortcut_draft_dirty = False
        self.read_error = None
        try:
            self.overrides = load_overrides(self.path)
        except SettingsError as exc:
            self.overrides = {}
            self.read_error = str(exc)
        self.effective = effective_settings(config, self.overrides)
        self.shortcuts = copy.deepcopy(self.effective["shortcuts"])
        root.title("CapsWriter · 日常設定")
        root.geometry("820x750")
        root.minsize(720, 660)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Escape>", lambda _event: self.close())
        root.bind("<Control-s>", lambda _event: self.save())
        outer = ttk.Frame(root, padding=20)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="讓說話成為輸入", font=("Microsoft JhengHei UI", 18, "bold")).pack(anchor="w")
        ttk.Label(outer, text="按住快捷鍵說話 → axolotl 辨識 → 輸入目前視窗", padding=(0, 8)).pack(anchor="w")
        self.status = tk.StringVar(value="所有變更皆於重新啟動 Client 後生效。")
        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill="both", expand=True, pady=12)
        self.pages = {}
        for name in ("連線", "錄音與快捷鍵", "輸出", "設定來源"):
            page = ttk.Frame(self.notebook, padding=16)
            page.columnconfigure(1, weight=1)
            self.notebook.add(page, text=name)
            self.pages[name] = page
        self.connection_page()
        self.recording_page()
        self.output_page()
        self.sources_page()
        ttk.Label(outer, textvariable=self.status, wraplength=740).pack(fill="x", pady=(4, 12))
        actions = ttk.Frame(outer)
        actions.pack(fill="x")
        ttk.Button(actions, text="恢復 Python 設定", command=self.reset).pack(side="left")
        ttk.Button(actions, text="關閉", command=self.close).pack(side="right")
        ttk.Button(actions, text="儲存設定  Ctrl+S", command=self.save).pack(side="right", padx=8)
        self.poll_after_id = root.after(100, self.poll)
        if self.read_error:
            self.status.set("設定檔有錯誤；修正原檔後重開，或選擇「恢復 Python 設定」。")
            root.after_idle(lambda: messagebox.showerror("設定檔無法讀取", self.read_error, parent=root))
        elif not self.path.exists():
            self.status.set("首次使用：請確認 Server 主機為 axolotl 或其 IP，測試連線，再儲存設定。Windows 不需安裝辨識模型。")
        self.widgets["addr"].focus_set()

    def mark_dirty(self, key):
        self.touched.add(key)
        self.status.set("有尚未儲存的變更；儲存後請重新啟動 Client。")

    def field(self, page, row, key, label, help_text="", choices=None, boolean=False, editable=False):
        ttk.Label(page, text=label).grid(row=row, column=0, sticky="w", padx=(0, 16), pady=(6, 0))
        value = self.effective[key]
        if key == "input_device":
            label = device_label(value)
            self.device_choices[label] = value
            value = label
        variable = tk.BooleanVar(value=value) if boolean else tk.StringVar(value="" if value is None else str(value))
        self.variables[key] = variable
        if boolean:
            widget = ttk.Checkbutton(page, variable=variable)
        elif choices is not None:
            widget = ttk.Combobox(page, textvariable=variable, values=choices, state="normal" if editable else "readonly")
        else:
            widget = ttk.Entry(page, textvariable=variable)
        widget.grid(row=row, column=1, sticky="ew", pady=(6, 0))
        self.widgets[key] = widget
        error = tk.StringVar(value=help_text)
        self.errors[key] = (error, help_text)
        ttk.Label(page, textvariable=error, wraplength=600).grid(row=row + 1, column=0, columnspan=2, sticky="w", pady=(2, 8))
        variable.trace_add("write", lambda *_: self.mark_dirty(key))
        widget.bind("<FocusOut>", lambda _event: self.validate_form(focus=False))
        return widget

    def connection_page(self):
        page = self.pages["連線"]
        self.field(page, 0, "addr", "Server 主機", "填 axolotl 或其 IP；127.0.0.1 指這台 Windows，不是遠端 Server。")
        self.field(page, 2, "port", "WebSocket 連接埠", "預設 6016。HTTP／Web 轉錄服務使用不同連接埠。")
        self.test_button = ttk.Button(page, text="測試 Server 連線", command=self.test_connection)
        self.test_button.grid(row=4, column=0, columnspan=2, sticky="w", pady=12)
        self.connection_status = tk.StringVar(value="會測試 WebSocket 握手；不會錄音或上傳內容。")
        ttk.Label(page, textvariable=self.connection_status, wraplength=650).grid(row=5, column=0, columnspan=2, sticky="w")
        self.field(page, 6, "language", "辨識語言", "auto 自動偵測；支援的語言依 Server 模型而定。")
        self.field(page, 8, "context", "辨識提示詞", "例如常用人名、地名與專業術語；效果依 Server 模型而定。")

    def recording_page(self):
        page = self.pages["錄音與快捷鍵"]
        self.mic = self.field(page, 0, "input_device", "麥克風", "留空使用系統預設。同名同介面以裝置編號區分；編號可能變動，增減裝置後請重新選擇。", choices=("",), editable=True)
        ttk.Button(page, text="重新整理麥克風", command=self.refresh_devices).grid(row=2, column=0, sticky="w")
        self.device_names = tk.StringVar(value="尚未讀取裝置。")
        ttk.Label(page, textvariable=self.device_names, wraplength=480).grid(row=2, column=1, sticky="w")
        self.field(page, 3, "threshold", "長按觸發時間（秒）", "預設 0.3 秒；短按仍保留按鍵原本功能。")
        self.field(page, 5, "save_audio", "保留錄音檔", boolean=True)
        ttk.Label(page, text="快捷鍵（選取後編輯，再按「套用按鍵」）").grid(row=7, column=0, columnspan=2, sticky="w")
        self.shortcut_list = ttk.Treeview(page, columns=("key", "type", "mode", "enabled"), show="headings", height=3, selectmode="browse")
        for key, title in (("key", "按鍵"), ("type", "類型"), ("mode", "模式"), ("enabled", "啟用")):
            self.shortcut_list.heading(key, text=title)
            self.shortcut_list.column(key, width=130, stretch=True)
        self.shortcut_list.grid(row=8, column=0, columnspan=2, sticky="ew", pady=6)
        self.shortcut_list.bind("<<TreeviewSelect>>", self.select_shortcut)
        edit = ttk.Frame(page)
        edit.grid(row=9, column=0, columnspan=2, sticky="ew")
        self.key = tk.StringVar(value="caps_lock")
        self.kind = tk.StringVar(value="keyboard")
        self.hold = tk.BooleanVar(value=True)
        self.suppress = tk.BooleanVar(value=True)
        self.enabled = tk.BooleanVar(value=True)
        for variable in (self.key, self.kind, self.hold, self.suppress, self.enabled):
            variable.trace_add("write", self.mark_shortcut_draft)
        ttk.Label(edit, text="按鍵").grid(row=0, column=0, sticky="w")
        ttk.Entry(edit, textvariable=self.key, width=17).grid(row=0, column=1, padx=6)
        ttk.Label(edit, text="類型").grid(row=0, column=2, sticky="w")
        ttk.Combobox(edit, textvariable=self.kind, values=("keyboard", "mouse"), state="readonly", width=12).grid(row=0, column=3, padx=6)
        for column, (label, variable) in enumerate((("按住錄音", self.hold), ("阻擋原按鍵", self.suppress), ("啟用", self.enabled))):
            ttk.Checkbutton(edit, text=label, variable=variable).grid(row=1, column=column, sticky="w", pady=6)
        for column, (label, command) in enumerate((("套用按鍵", self.update_shortcut), ("新增", lambda: self.update_shortcut(new=True)), ("移除", self.remove_shortcut))):
            ttk.Button(edit, text=label, command=command).grid(row=2, column=column, sticky="w", pady=4)
        error = tk.StringVar(value="例如 caps_lock、f12；滑鼠使用 x1／x2。未勾「按住錄音」時，再按一次停止。")
        self.errors["shortcuts"] = (error, error.get())
        ttk.Label(page, textvariable=error, wraplength=660).grid(row=10, column=0, columnspan=2, sticky="w", pady=4)
        self.render_shortcuts()

    def mark_shortcut_draft(self, *_args):
        self.shortcut_draft_dirty = True

    def render_shortcuts(self, selected=0):
        self.shortcut_list.delete(*self.shortcut_list.get_children())
        for i, item in enumerate(self.shortcuts):
            self.shortcut_list.insert("", "end", iid=str(i), values=(item["key"], item["type"], "按住" if item["hold_mode"] else "切換", "是" if item["enabled"] else "否"))
        if self.shortcuts:
            self.shortcut_list.selection_set(str(min(selected, len(self.shortcuts) - 1)))
            self.select_shortcut()

    def select_shortcut(self, _event=None):
        selected = self.shortcut_list.selection()
        if selected:
            item = self.shortcuts[int(selected[0])]
            for variable, key in ((self.key, "key"), (self.kind, "type"), (self.hold, "hold_mode"), (self.suppress, "suppress"), (self.enabled, "enabled")):
                variable.set(item[key])
            self.shortcut_draft_dirty = False

    def update_shortcut(self, new=False):
        item = {"key": self.key.get().strip().lower(), "type": self.kind.get(), "hold_mode": self.hold.get(), "suppress": self.suppress.get(), "enabled": self.enabled.get()}
        proposed = copy.deepcopy(self.shortcuts)
        selected = self.shortcut_list.selection()
        if selected and not new:
            proposed[int(selected[0])] = item
        else:
            proposed.append(item)
        try:
            validate({"shortcuts": proposed})
        except SettingsError as exc:
            self.errors["shortcuts"][0].set(exc.errors["shortcuts"])
            return
        self.shortcuts = proposed
        self.mark_dirty("shortcuts")
        self.render_shortcuts(int(selected[0]) if selected and not new else len(proposed) - 1)
        self.errors["shortcuts"][0].set(self.errors["shortcuts"][1])

    def remove_shortcut(self):
        selected = self.shortcut_list.selection()
        if selected:
            proposed = copy.deepcopy(self.shortcuts)
            del proposed[int(selected[0])]
            try:
                validate({"shortcuts": proposed})
            except SettingsError as exc:
                self.errors["shortcuts"][0].set(exc.errors["shortcuts"])
                return
            self.shortcuts = proposed
            self.render_shortcuts()
            self.mark_dirty("shortcuts")

    def output_page(self):
        page = self.pages["輸出"]
        self.field(page, 0, "paste", "使用剪貼簿貼上", "關閉時逐字輸入；開啟時以 Ctrl+V 貼上。", boolean=True)
        self.field(page, 2, "restore_clip", "貼上後還原剪貼簿", boolean=True)
        self.field(page, 4, "traditional_convert", "轉為繁體中文", boolean=True)
        self.field(page, 6, "traditional_locale", "繁體地區", "zh-tw：臺灣；zh-hant：一般繁體；zh-hk：香港。", choices=("zh-tw", "zh-hant", "zh-hk"))
        self.field(page, 8, "trash_punc_thresh", "短句去尾端標點門檻", "單詞數低於此值時，移除進階設定指定的尾端標點。0 停用。")
        self.field(page, 10, "llm_enabled", "啟用 LLM 潤色", "需另於 LLM/ 角色檔配置服務；首次使用可先關閉。", boolean=True)

    def sources_page(self):
        page = self.pages["設定來源"]
        ttk.Label(page, text=f"日常設定：{self.path}\n進階設定：config_client.py\n此表為已儲存、下次啟動將採用的值；不代表其他執行中的 Client 已套用。", wraplength=650).pack(anchor="w", pady=(0, 12))
        self.sources = ttk.Treeview(page, columns=("key", "value", "source"), show="headings", height=15)
        for key, title, width in (("key", "設定", 160), ("value", "下次啟動值", 310), ("source", "來源", 130)):
            self.sources.heading(key, text=title)
            self.sources.column(key, width=width)
        self.sources.pack(fill="both", expand=True)
        self.render_sources()

    def render_sources(self):
        self.sources.delete(*self.sources.get_children())
        for key, value in effective_settings(self.config, self.overrides).items():
            self.sources.insert("", "end", values=(key, json.dumps(value, ensure_ascii=False), "日常設定" if key in self.overrides else "config_client.py"))

    def collect(self):
        values = {}
        errors = {}
        for key in self.touched:
            if key == "shortcuts":
                values[key] = copy.deepcopy(self.shortcuts)
                continue
            value = self.variables[key].get()
            if key in ("threshold", "trash_punc_thresh"):
                try:
                    value = float(value) if key == "threshold" else int(value)
                except ValueError:
                    errors[key] = "請輸入有效數字" if key == "threshold" else "請輸入整數"
                    continue
            if key == "input_device":
                value = self.device_choices.get(value, value.strip() or None)
            values[key] = value
        if errors:
            raise SettingsError(errors)
        return validate(values)

    def validate_form(self, focus=True):
        for variable, help_text in self.errors.values():
            variable.set(help_text)
        try:
            return self.collect()
        except SettingsError as exc:
            for key, message in exc.errors.items():
                if key in self.errors:
                    self.errors[key][0].set("錯誤：" + message)
            if focus:
                self.status.set("請修正欄位旁的錯誤後再儲存。")
                key = next(iter(exc.errors))
                if key in self.widgets:
                    widget = self.widgets[key]
                    self.notebook.select(widget.master)
                    widget.focus_set()
            return None

    def save(self):
        if self.read_error:
            messagebox.showerror("原設定檔需要修復", "為避免覆蓋無法讀取的設定，請先修正原檔，或選擇「恢復 Python 設定」。", parent=self.root)
            return
        if self.shortcut_draft_dirty:
            self.notebook.select(self.pages["錄音與快捷鍵"])
            self.status.set("快捷鍵編輯尚未套用；請先按「套用按鍵」或「新增」，再儲存設定。")
            return
        values = self.validate_form()
        if values is None:
            return
        try:
            self.overrides = update_overrides(values, self.path)
        except (OSError, SettingsError) as exc:
            messagebox.showerror("儲存失敗", str(exc), parent=self.root)
            return
        self.reload_fields()
        self.touched.clear()
        self.saved = True
        self.render_sources()
        self.status.set("已儲存。請結束並重新啟動 Client；執行中的錄音與設定不會被中途變更。")

    def reset(self):
        if not messagebox.askyesno("恢復 Python 設定", "移除此裝置的所有日常覆寫，回到 config_client.py？下次啟動生效。", parent=self.root):
            return
        try:
            save_overrides({}, self.path)
        except OSError as exc:
            messagebox.showerror("儲存失敗", str(exc), parent=self.root)
            return
        self.read_error = None
        self.overrides = {}
        self.reload_fields()
        self.render_sources()
        self.touched.clear()
        self.saved = True
        self.status.set("已恢復 Python 設定；請重新啟動 Client。")

    def reload_fields(self):
        self.effective = effective_settings(self.config, self.overrides)
        for key, variable in self.variables.items():
            value = self.effective[key]
            if key == "input_device":
                label = device_label(value)
                self.device_choices[label] = value
                value = label
            variable.set("" if value is None else value)
        self.shortcuts = copy.deepcopy(self.effective["shortcuts"])
        self.render_shortcuts()

    def test_connection(self):
        try:
            values = validate({key: self.variables[key].get() for key in ("addr", "port")})
        except SettingsError as exc:
            for key, message in exc.errors.items():
                self.errors[key][0].set("錯誤：" + message)
            return
        self.test_button.configure(state="disabled")
        self.connection_status.set("正在連線，最多約 6 秒……")

        def worker():
            try:
                asyncio.run(probe_connection(values["addr"], values["port"]))
                result = "連線成功：Server 接受 WebSocket 握手。辨識模型與麥克風仍需實際錄音驗證。"
            except Exception as exc:
                result = f"無法連線：{type(exc).__name__}: {exc}。請確認 Server 已啟動、主機／連接埠與防火牆。"
            self.pending.put(("connection", result))

        threading.Thread(target=worker, daemon=True, name="settings-connection-test").start()

    def refresh_devices(self):
        self.device_names.set("正在讀取麥克風……")

        def worker():
            try:
                import sounddevice
                choices = input_device_choices(sounddevice)
                result = (choices, f"找到 {len(choices)} 個麥克風，可從上方清單選擇。" if choices else "未找到麥克風；請檢查 Windows 麥克風權限及裝置連線。")
            except Exception as exc:
                result = ([], f"無法列出裝置：{exc}。仍可輸入裝置名稱或留空使用預設。")
            self.pending.put(("devices", result))

        threading.Thread(target=worker, daemon=True, name="settings-audio-devices").start()

    def poll(self):
        if self.poll_after_id is not None:
            self.root.after_cancel(self.poll_after_id)
            self.poll_after_id = None
        if self.closed:
            return
        try:
            while True:
                kind, result = self.pending.get_nowait()
                if kind == "connection":
                    self.connection_status.set(result)
                    self.test_button.configure(state="normal")
                else:
                    choices, message = result
                    self.device_choices.update({item["label"]: item["value"] for item in choices})
                    self.mic.configure(values=("", *(item["label"] for item in choices)))
                    self.device_names.set(message)
        except queue.Empty:
            pass
        self.poll_after_id = self.root.after(100, self.poll)

    def close(self):
        if (self.touched or self.shortcut_draft_dirty) and not messagebox.askyesno("尚未儲存", "捨棄尚未儲存的變更並關閉？", parent=self.root):
            return
        self.closed = True
        if self.poll_after_id is not None:
            self.root.after_cancel(self.poll_after_id)
            self.poll_after_id = None
        self.root.destroy()


def run_settings() -> bool:
    from config_client import ClientConfig

    root = tk.Tk()
    window = SettingsWindow(root, ClientConfig)
    root.mainloop()
    return window.saved
