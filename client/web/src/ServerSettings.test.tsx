import { act, fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import ServerSettings from "./ServerSettings";
import type { ApiSettings, ServerSettingField, ServerSettingsResponse } from "./types";

const settings: ApiSettings = { baseUrl: "http://localhost:6017", apiKey: "test-key", model: "", language: "", prompt: "", responseFormat: "text" };
const upload: ServerSettingField = {
  key: "max_upload_mb", label: "檔案上限 (MB)", description: "HTTP 單次上傳的大小上限。", type: "integer",
  value: 100, saved_value: null, default: 100, source: "default", next_value: 100, next_source: "default",
  minimum: 1, maximum: 1024, restart_required: false, environment_variable: "CAPSWRITER_HTTP_API_MAX_UPLOAD_MB", overridden_by_environment: false,
};
const snapshot = (fields = [upload]): ServerSettingsResponse => ({ enabled: true, revision: "revision-one", restart_required: fields.some((field) => field.restart_required), fields });
const response = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });
const read = () => userEvent.click(screen.getByRole("button", { name: "讀取 Server 設定" }));
const save = () => userEvent.click(screen.getByRole("button", { name: "儲存 Server 設定" }));

afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });

describe("Server settings", () => {
  it("saves only changed values with authentication and revision, retaining the running value", async () => {
    const fetchMock = vi.fn(async (_url: RequestInfo | URL, init?: RequestInit) => init?.method === "PATCH"
      ? response(snapshot([{ ...upload, saved_value: 200, next_value: 200, next_source: "saved", restart_required: true }]))
      : response(snapshot()));
    vi.stubGlobal("fetch", fetchMock);
    render(<ServerSettings settings={settings} />);
    expect(fetchMock).not.toHaveBeenCalled();
    await read();
    fireEvent.change(screen.getByLabelText(upload.label), { target: { value: "200" } });
    await save();
    expect(fetchMock.mock.calls[1]?.[1]).toMatchObject({ method: "PATCH", headers: { Authorization: "Bearer test-key" }, body: JSON.stringify({ values: { max_upload_mb: 200 }, revision: "revision-one" }) });
    expect(await screen.findByText("已儲存。請由管理者在適當時間重啟 Server，變更才會生效。")).toBeTruthy();
    expect(screen.getByText("下次啟動：200（設定檔）")).toBeTruthy();
    expect(screen.getByText("目前來源：預設值")).toBeTruthy();
    expect(screen.getByText("100", { selector: "strong" })).toBeTruthy();
  });

  it("validates bounds before sending a write and focuses the error summary", async () => {
    const fetchMock = vi.fn(async () => response(snapshot()));
    vi.stubGlobal("fetch", fetchMock);
    render(<ServerSettings settings={settings} />);
    await read();
    fireEvent.change(screen.getByLabelText(upload.label), { target: { value: "2048" } });
    await save();
    const alert = screen.getByRole("alert");
    expect(within(alert).getByRole("link").textContent).toContain("不得大於 1024");
    expect(document.activeElement).toBe(alert);
    expect(screen.getByLabelText(upload.label).getAttribute("aria-invalid")).toBe("true");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("locks environment-controlled fields and clears saved overrides with null", async () => {
    const locked = { ...upload, key: "locked", label: "鎖定欄位", value: 500, overridden_by_environment: true, source: "environment" };
    const fetchMock = vi.fn(async (_url: RequestInfo | URL, init?: RequestInit) => response(init?.method === "PATCH" ? snapshot() : snapshot([{ ...upload, saved_value: 200 }, locked])));
    vi.stubGlobal("fetch", fetchMock);
    render(<ServerSettings settings={settings} />);
    await read();
    expect((screen.getByLabelText("鎖定欄位") as HTMLInputElement).disabled).toBe(true);
    await userEvent.click(screen.getByRole("button", { name: "恢復預設：檔案上限 (MB)" }));
    expect(screen.getByText("待儲存：移除覆寫，恢復預設值。")).toBeTruthy();
    await save();
    expect(fetchMock.mock.calls[1]?.[1]?.body).toBe(JSON.stringify({ values: { max_upload_mb: null }, revision: "revision-one" }));
  });

  it("preserves a conflicting draft and requires an explicit reload before retrying", async () => {
    vi.stubGlobal("fetch", vi.fn(async (_url: RequestInfo | URL, init?: RequestInit) => init?.method === "PATCH" ? response({ error: { message: "conflict" } }, 409) : response(snapshot())));
    render(<ServerSettings settings={settings} />);
    await read();
    fireEvent.change(screen.getByLabelText(upload.label), { target: { value: "200" } });
    await save();
    expect(screen.getByRole("alert").textContent).toContain("您的輸入已保留");
    expect((screen.getByLabelText(upload.label) as HTMLInputElement).value).toBe("200");
    expect((screen.getByRole("button", { name: "儲存 Server 設定" }) as HTMLButtonElement).disabled).toBe(true);
    vi.spyOn(window, "confirm").mockReturnValue(false);
    await read();
    expect((screen.getByLabelText(upload.label) as HTMLInputElement).value).toBe("200");
    vi.mocked(window.confirm).mockReturnValue(true);
    await read();
    expect((screen.getByLabelText(upload.label) as HTMLInputElement).value).toBe("100");
  });

  it.each([404, 401, 200])("explains unavailable settings (%s) without showing editable fields", async (status) => {
    vi.stubGlobal("fetch", vi.fn(async () => response(status === 200 ? { enabled: false, revision: null, fields: [], restart_required: false } : { detail: "Unavailable" }, status)));
    render(<ServerSettings settings={settings} />);
    await read();
    expect(screen.queryByRole("button", { name: "儲存 Server 設定" })).toBeNull();
    expect(screen.getByText(status === 401 ? /無法讀取設定/ : status === 404 ? /尚未提供圖形設定/ : /未啟用圖形設定/)).toBeTruthy();
  });

  it("ignores a late load response after cancellation", async () => {
    let complete!: (result: Response) => void;
    vi.stubGlobal("fetch", vi.fn(() => new Promise<Response>((resolve) => { complete = resolve; })));
    render(<ServerSettings settings={settings} />);
    await read();
    await userEvent.click(screen.getByRole("button", { name: "取消讀取" }));
    await act(async () => { complete(response(snapshot())); });
    expect(screen.getByText("已取消讀取。")).toBeTruthy();
    expect(screen.queryByLabelText(upload.label)).toBeNull();
  });

  it("rejects malformed server metadata rather than rendering an editable form", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => response(snapshot([{ ...upload, default: {} } as unknown as ServerSettingField]))));
    render(<ServerSettings settings={settings} />);
    await read();
    expect(screen.getByRole("alert").textContent).toContain("設定欄位格式不正確");
    expect(screen.queryByLabelText(upload.label)).toBeNull();
  });
});
