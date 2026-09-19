import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { RefreshCw, Server, Save } from "lucide-react";
import { fetchServerSettings, saveServerSettings, safeErrorMessage, ServerSettingsError } from "./api/capswriter";
import type { ApiSettings, ServerSettingField, ServerSettingsResponse, ServerSettingValue } from "./types";

type Draft = string | boolean | null;
const formatValue = (value: ServerSettingValue | null) => value === null ? "自動" : typeof value === "boolean" ? value ? "開啟" : "關閉" : String(value);
const initialValue = (field: ServerSettingField): Draft => {
  const value = field.overridden_by_environment ? field.value : field.saved_value ?? field.default;
  return typeof value === "number" ? String(value) : value;
};
const sourceLabel = (source: string) => source === "environment" ? "環境變數" : source === "saved" ? "設定檔" : "預設值";
const endpointIdentity = (settings: ApiSettings) => JSON.stringify([settings.baseUrl, settings.apiKey]);

export interface ServerSettingsHandle {
  confirmEndpointChange: () => boolean;
}

const ServerSettings = forwardRef<ServerSettingsHandle, { settings: ApiSettings }>(function ServerSettings({ settings }, ref) {
  const [data, setData] = useState<ServerSettingsResponse | null>(null);
  const [edits, setEdits] = useState<Record<string, Draft>>({});
  const [phase, setPhase] = useState<"idle" | "loading" | "saving">("idle");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [conflict, setConflict] = useState(false);
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const request = useRef<AbortController | null>(null);
  const errorSummary = useRef<HTMLDivElement>(null);
  const endpoint = useRef(endpointIdentity(settings));
  const latestSettings = useRef(settings);
  const snapshotSettings = useRef<ApiSettings | null>(null);
  latestSettings.current = settings;
  const busy = phase !== "idle";
  const dirty = Object.keys(edits).length > 0;

  useImperativeHandle(ref, () => ({
    confirmEndpointChange: () => {
      if (!dirty && phase !== "saving") return true;
      return window.confirm(phase === "saving"
        ? "Server 設定仍在儲存。切換連線後仍會等待原連線的結果，是否繼續？"
        : "切換連線將捨棄尚未儲存的 Server 設定，是否繼續？");
    },
  }), [dirty, phase]);

  useEffect(() => () => request.current?.abort(), []);
  useEffect(() => { if (error) errorSummary.current?.focus(); }, [error]);
  useEffect(() => {
    const nextEndpoint = endpointIdentity(settings);
    if (endpoint.current === nextEndpoint) return;
    endpoint.current = nextEndpoint;
    // A write owns the endpoint captured when it began. Its completion clears
    // the old snapshot and reports the outcome after an accepted switch.
    if (phase === "saving") return;
    request.current?.abort(); request.current = null;
    snapshotSettings.current = null;
    setData(null); setEdits({}); setPhase("idle"); setError(""); setFieldErrors({}); setConflict(false);
    setMessage("連線已切換，請重新讀取 Server 設定。");
  }, [settings.baseUrl, settings.apiKey, phase]);

  const load = async () => {
    if (dirty && !window.confirm("重新讀取將捨棄尚未儲存的變更，是否繼續？")) return;
    const requestSettings = { ...settings };
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    snapshotSettings.current = null;
    setPhase("loading"); setError(""); setMessage(""); setData(null); setEdits({}); setFieldErrors({}); setConflict(false);
    try {
      const snapshot = await fetchServerSettings(requestSettings, controller.signal);
      if (request.current !== controller) return;
      snapshotSettings.current = requestSettings;
      setData(snapshot);
    } catch (reason) {
      if (request.current !== controller) return;
      if (reason instanceof ServerSettingsError && reason.status === 404) {
        setMessage("此 Server 尚未提供圖形設定。您仍可使用轉錄功能，並透過 Server 設定檔調整參數。");
      } else if (reason instanceof ServerSettingsError && reason.status === 401) {
        setError("無法讀取設定：請在連線區輸入有效的 API key，再重新讀取。");
      } else {
        setError(safeErrorMessage(reason, settings.apiKey, "讀取設定失敗"));
      }
    } finally {
      if (request.current === controller) { request.current = null; setPhase("idle"); }
    }
  };

  const change = (field: ServerSettingField, value: Draft) => {
    setEdits((previous) => {
      const next = { ...previous, [field.key]: value };
      if (value === initialValue(field) || (value === null && field.saved_value === null)) delete next[field.key];
      return next;
    });
    setFieldErrors((previous) => { const next = { ...previous }; delete next[field.key]; return next; });
    setMessage("");
  };

  const save = async () => {
    if (!data?.revision || busy || conflict) return;
    const requestSettings = snapshotSettings.current;
    if (!requestSettings) return;
    const values: Record<string, ServerSettingValue | null> = {};
    const invalid: Record<string, string> = {};
    for (const field of data.fields) {
      if (!(field.key in edits)) continue;
      const raw = edits[field.key];
      if (raw === null) { values[field.key] = null; continue; }
      if (field.type === "integer" || field.type === "number") {
        const value = Number(raw);
        if (String(raw).trim() === "" || !Number.isFinite(value) || (field.type === "integer" && !Number.isInteger(value))) invalid[field.key] = field.type === "integer" ? "請輸入整數。" : "請輸入有效數字。";
        else if (field.minimum !== undefined && value < field.minimum) invalid[field.key] = `不得小於 ${field.minimum}。`;
        else if (field.maximum !== undefined && value > field.maximum) invalid[field.key] = `不得大於 ${field.maximum}。`;
        else values[field.key] = value;
      } else values[field.key] = raw;
    }
    setFieldErrors(invalid);
    if (Object.keys(invalid).length) { setError("請修正下列欄位後再儲存。"); errorSummary.current?.focus(); return; }
    const controller = new AbortController();
    request.current = controller;
    setPhase("saving"); setError(""); setMessage("");
    try {
      const snapshot = await saveServerSettings(requestSettings, values, data.revision, controller.signal);
      if (request.current !== controller) return;
      const switched = endpointIdentity(latestSettings.current) !== endpointIdentity(requestSettings);
      snapshotSettings.current = switched ? null : requestSettings;
      setData(switched ? null : snapshot); setEdits({});
      setMessage(switched
        ? "切換前的 Server 已儲存設定。連線已切換，請重新讀取目前 Server 的設定。"
        : snapshot.restart_required ? "已儲存。請由管理者在適當時間重啟 Server，變更才會生效。" : "已儲存；目前沒有等待重啟的變更。");
    } catch (reason) {
      if (request.current !== controller) return;
      const switched = endpointIdentity(latestSettings.current) !== endpointIdentity(requestSettings);
      if (switched) {
        snapshotSettings.current = null;
        setData(null); setEdits({}); setConflict(false); setFieldErrors({});
        setError(`${safeErrorMessage(reason, requestSettings.apiKey, "儲存失敗")} 這是切換前 Server 的儲存結果；結果可能不確定。請切回原連線並重新讀取，確認原 Server 是否已儲存；目前連線的設定也需重新讀取。`);
        return;
      }
      if (reason instanceof ServerSettingsError && reason.status === 409) {
        setConflict(true);
        setError("設定已由其他使用者或程式更新。您的輸入已保留；請先重新讀取，再確認並套用變更。");
      } else {
        if (reason instanceof ServerSettingsError) setFieldErrors(reason.fields);
        else setConflict(true); // A lost response may follow a completed write; read before retrying.
        setError(`${safeErrorMessage(reason, requestSettings.apiKey, "儲存失敗")} 請確認設定；連線中斷時請先重新讀取。`);
      }
    } finally {
      if (request.current === controller) { request.current = null; setPhase("idle"); }
    }
  };

  const stopWaiting = () => {
    request.current?.abort(); request.current = null;
    snapshotSettings.current = null;
    setMessage(phase === "saving" ? "已停止等待。發出儲存請求的 Server 可能已儲存；若已切換連線，請切回原連線並重新讀取確認結果。" : "已取消讀取。");
    setData(null); setEdits({}); setPhase("idle");
  };

  return (
    <section className="panel server-settings-panel" aria-labelledby="server-settings-title">
      <div className="panel-heading split">
        <div className="heading-inline"><Server size={20} aria-hidden="true" /><h2 id="server-settings-title">Server 共用設定</h2></div>
        <button className="secondary-action" type="button" onClick={load} disabled={busy}>
          <RefreshCw size={18} aria-hidden="true" className={phase === "loading" ? "spin" : ""} />{phase === "loading" ? "讀取中" : "讀取 Server 設定"}
        </button>
      </div>
      <p className="settings-help">調整上方連線所指向的 Server，供所有 Client 共用。儲存後需重啟 Server；進階參數仍使用設定檔。</p>
      {!data && !error && !message && !busy ? <p className="settings-help">先填入 axolotl 的 API 位址與 API key，再讀取共用設定。Windows 快捷鍵、麥克風及文字輸出請在桌面 Client 設定。</p> : null}
      {error ? <div className="settings-notice error" role="alert" tabIndex={-1} ref={errorSummary}>
        <p>{error}</p>
        {Object.entries(fieldErrors).length ? <ul>{Object.entries(fieldErrors).map(([key, detail]) => <li key={key}><a href={`#server-setting-${key}`}>{data?.fields.find((field) => field.key === key)?.label ?? key}：{detail}</a></li>)}</ul> : null}
      </div> : null}
      {message ? <p className="settings-notice" role="status">{message}</p> : null}
      {data && (!data.enabled || data.available === false) ? <div className="settings-notice">
        <p>此 Server 未啟用圖形設定，轉錄功能仍可照常使用。</p>
        <p>管理者需設定 API key、啟用 <code>CAPSWRITER_SETTINGS_ENABLE</code>，並指定可寫入的 <code>CAPSWRITER_SETTINGS_PATH</code> 後重啟 Server。</p>
      </div> : null}
      {data?.enabled && data.available !== false ? <form noValidate onSubmit={(event) => { event.preventDefault(); void save(); }}>
        {data.restart_required ? <p className="settings-notice pending" role="status">有已儲存的變更等待重啟。下方「目前生效」仍是 Server 正在使用的值。</p> : null}
        <div className="server-settings-grid">
          {data.fields.map((field) => {
            const id = `server-setting-${field.key}`;
            const draft = field.key in edits ? edits[field.key] : initialValue(field);
            const value = draft === null ? field.default : draft;
            const disabled = busy || field.overridden_by_environment;
            const invalid = fieldErrors[field.key];
            const inputProps = { id, disabled, "aria-describedby": `${id}-help${invalid ? ` ${id}-error` : ""}`, "aria-invalid": Boolean(invalid) };
            return <div className="server-setting" key={field.key}>
              <label htmlFor={id}>{field.label}</label>
              {field.type === "boolean" ? <select {...inputProps} value={String(value)} onChange={(event) => change(field, event.target.value === "true")}><option value="true">開啟</option><option value="false">關閉</option></select>
                : field.choices?.length ? <select {...inputProps} value={String(value ?? "")} onChange={(event) => change(field, event.target.value)}>{field.choices.map((choice) => <option key={String(choice)} value={String(choice)}>{String(choice)}</option>)}</select>
                : <input {...inputProps} type={field.type === "string" ? "text" : "number"} value={String(value ?? "")} placeholder={field.default === null ? "自動" : undefined} min={field.minimum} max={field.maximum} step={field.type === "integer" ? 1 : "any"} onChange={(event) => change(field, event.target.value)} />}
              <div id={`${id}-help`} className="settings-help">
                <p>{field.description}</p>
                <p>目前生效：<strong>{formatValue(field.value)}</strong> · 預設：{formatValue(field.default)}</p>
                <p>已儲存：{field.saved_value === null ? "未覆寫" : formatValue(field.saved_value)}{field.restart_required ? "（等待重啟）" : ""}</p>
                <p>目前來源：{sourceLabel(field.source)}</p>
                <p>下次啟動：{formatValue(field.next_value !== undefined ? field.next_value : field.overridden_by_environment ? field.value : field.saved_value ?? field.default)}（{sourceLabel(field.next_source ?? (field.overridden_by_environment ? "environment" : field.saved_value !== null ? "saved" : "default"))}）</p>
                {field.overridden_by_environment ? <p>由環境變數 <code>{field.environment_variable}</code> 控制；請在 Server 調整。</p> : null}
                {field.key in edits && draft === null ? <p>待儲存：移除覆寫，恢復預設值。</p> : null}
              </div>
              {invalid ? <p id={`${id}-error`} className="field-error">{invalid}</p> : null}
              <button className="secondary-action reset-setting" type="button" disabled={disabled || (field.saved_value === null && !(field.key in edits))} onClick={() => change(field, null)}>恢復預設：{field.label}</button>
            </div>;
          })}
        </div>
        <div className="action-row settings-actions">
          <button className="primary-action" type="submit" disabled={busy || !dirty || conflict}><Save size={18} aria-hidden="true" />{phase === "saving" ? "儲存中" : "儲存 Server 設定"}</button>
          <button className="secondary-action" type="button" disabled={busy || !dirty} onClick={() => { setEdits({}); setFieldErrors({}); setError(""); setMessage("已捨棄未儲存的變更。"); }}>捨棄變更</button>
          <span className="settings-help" aria-live="polite">{dirty ? `${Object.keys(edits).length} 個欄位尚未儲存` : "沒有未儲存的變更"}</span>
        </div>
      </form> : null}
      {busy ? <button className="secondary-action" type="button" onClick={stopWaiting}>{phase === "saving" ? "停止等待儲存" : "取消讀取"}</button> : null}
    </section>
  );
});

export default ServerSettings;
