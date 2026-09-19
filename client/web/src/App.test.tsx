import userEvent from "@testing-library/user-event";
import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { StrictMode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import {
  WEB_SETTING_LIMITS,
  loadHistory,
  saveHistory,
} from "./lib/storage";

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((done, fail) => {
    resolve = done;
    reject = fail;
  });
  return { promise, resolve, reject };
}

const serverSettingsSnapshot = (savedValue: number | null = null) => ({
  enabled: true,
  revision: "revision-one",
  restart_required: savedValue !== null,
  fields: [{
    key: "max_upload_mb",
    label: "檔案上限 (MB)",
    description: "HTTP 單次上傳的大小上限。",
    type: "integer",
    value: 100,
    saved_value: savedValue,
    default: 100,
    source: "default",
    next_value: savedValue ?? 100,
    next_source: savedValue === null ? "default" : "saved",
    minimum: 1,
    maximum: 1024,
    restart_required: savedValue !== null,
    environment_variable: "CAPSWRITER_HTTP_API_MAX_UPLOAD_MB",
    overridden_by_environment: false,
  }],
});

const jsonResponse = (body: unknown) => new Response(JSON.stringify(body), {
  headers: { "Content-Type": "application/json" },
});

describe("App", () => {
  const originalMediaDevices = Object.getOwnPropertyDescriptor(navigator, "mediaDevices");
  const originalClipboard = Object.getOwnPropertyDescriptor(navigator, "clipboard");

  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    if (originalMediaDevices) {
      Object.defineProperty(navigator, "mediaDevices", originalMediaDevices);
    } else {
      delete (navigator as unknown as { mediaDevices?: MediaDevices }).mediaDevices;
    }
    if (originalClipboard) {
      Object.defineProperty(navigator, "clipboard", originalClipboard);
    } else {
      delete (navigator as unknown as { clipboard?: Clipboard }).clipboard;
    }
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("renders the primary workbench regions", () => {
    const { container } = render(<App />);

    expect(screen.getByRole("heading", { name: "Web Console" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "連線" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "音訊" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "轉錄" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "TTS" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "歷史" })).toBeTruthy();
    expect(screen.getByRole("textbox", { name: "轉錄內容" })).toBeTruthy();
    expect(
      Array.from(container.querySelectorAll(".workspace-column"), (column) =>
        Array.from(column.querySelectorAll("h2"), (heading) => heading.textContent),
      ),
    ).toEqual([["連線"], ["音訊", "轉錄"], ["TTS", "歷史"]]);
  });

  it("renders settings controls with bounded input lengths", () => {
    render(<App />);

    expect((screen.getByLabelText("API root") as HTMLInputElement).maxLength).toBe(WEB_SETTING_LIMITS.baseUrl);
    expect((screen.getByLabelText("API key") as HTMLInputElement).maxLength).toBe(WEB_SETTING_LIMITS.apiKey);
    expect((screen.getByLabelText("語言") as HTMLInputElement).maxLength).toBe(WEB_SETTING_LIMITS.language);
    expect((screen.getByLabelText("模型") as HTMLInputElement).maxLength).toBe(WEB_SETTING_LIMITS.model);
    expect((screen.getByLabelText("Prompt") as HTMLTextAreaElement).maxLength).toBe(WEB_SETTING_LIMITS.prompt);
  });

  it.each([
    ["API root", "http://localhost:6017", "http://other:6017"],
    ["API key", "", "replacement-key"],
  ])("guards dirty Server settings when changing %s", async (label, original, replacement) => {
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse(serverSettingsSnapshot())));
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: "讀取 Server 設定" }));
    fireEvent.change(screen.getByLabelText("檔案上限 (MB)"), { target: { value: "200" } });

    fireEvent.change(screen.getByLabelText(label), { target: { value: replacement } });

    expect(confirm).toHaveBeenCalledWith("切換連線將捨棄尚未儲存的 Server 設定，是否繼續？");
    expect((screen.getByLabelText(label) as HTMLInputElement).value).toBe(original);
    expect((screen.getByLabelText("檔案上限 (MB)") as HTMLInputElement).value).toBe("200");

    confirm.mockReturnValue(true);
    fireEvent.change(screen.getByLabelText(label), { target: { value: replacement } });

    expect((screen.getByLabelText(label) as HTMLInputElement).value).toBe(replacement);
    expect(await screen.findByText("連線已切換，請重新讀取 Server 設定。")).toBeTruthy();
    expect(screen.queryByLabelText("檔案上限 (MB)")).toBeNull();
  });

  it("resets a clean Server snapshot without confirmation when the endpoint changes", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse(serverSettingsSnapshot())));
    const confirm = vi.spyOn(window, "confirm");
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: "讀取 Server 設定" }));
    expect(screen.getByLabelText("檔案上限 (MB)")).toBeTruthy();

    fireEvent.change(screen.getByLabelText("API root"), { target: { value: "http://other:6017" } });

    expect(await screen.findByText("連線已切換，請重新讀取 Server 設定。")).toBeTruthy();
    expect(screen.queryByLabelText("檔案上限 (MB)")).toBeNull();
    expect(confirm).not.toHaveBeenCalled();
  });

  it("finishes an in-flight write against its original root before requiring a reload", async () => {
    const write = deferred<Response>();
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) =>
      init?.method === "PATCH" ? write.promise : Promise.resolve(jsonResponse(serverSettingsSnapshot())));
    vi.stubGlobal("fetch", fetchMock);
    vi.spyOn(window, "confirm").mockReturnValue(true);
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: "讀取 Server 設定" }));
    fireEvent.change(screen.getByLabelText("檔案上限 (MB)"), { target: { value: "200" } });
    await userEvent.click(screen.getByRole("button", { name: "儲存 Server 設定" }));

    fireEvent.change(screen.getByLabelText("API root"), { target: { value: "http://other:6017" } });
    expect(fetchMock.mock.calls[1]?.[0]).toBe("http://localhost:6017/v1/settings");

    await act(async () => { write.resolve(jsonResponse(serverSettingsSnapshot(200))); });

    expect(await screen.findByText("切換前的 Server 已儲存設定。連線已切換，請重新讀取目前 Server 的設定。")).toBeTruthy();
    expect(screen.queryByLabelText("檔案上限 (MB)")).toBeNull();
  });

  it("keeps an uncertain write on its original key and explains the reload after switching", async () => {
    localStorage.setItem("capswriter.web.settings", JSON.stringify({
      baseUrl: "http://localhost:6017",
      model: "",
      language: "",
      prompt: "",
      responseFormat: "text",
    }));
    const write = deferred<Response>();
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) =>
      init?.method === "PATCH" ? write.promise : Promise.resolve(jsonResponse(serverSettingsSnapshot())));
    vi.stubGlobal("fetch", fetchMock);
    vi.spyOn(window, "confirm").mockReturnValue(true);
    render(<App />);
    fireEvent.change(screen.getByLabelText("API key"), { target: { value: "original-key" } });
    await userEvent.click(screen.getByRole("button", { name: "讀取 Server 設定" }));
    fireEvent.change(screen.getByLabelText("檔案上限 (MB)"), { target: { value: "200" } });
    await userEvent.click(screen.getByRole("button", { name: "儲存 Server 設定" }));

    fireEvent.change(screen.getByLabelText("API key"), { target: { value: "replacement-key" } });
    expect(fetchMock.mock.calls[1]?.[1]).toMatchObject({ headers: { Authorization: "Bearer original-key" } });

    await act(async () => { write.reject(new TypeError("response lost")); });

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("response lost");
    expect(alert.textContent).toContain("這是切換前 Server 的儲存結果；結果可能不確定。");
    expect(alert.textContent).toContain("請切回原連線並重新讀取，確認原 Server 是否已儲存");
    expect((screen.getByLabelText("API key") as HTMLInputElement).value).toBe("replacement-key");
    expect(screen.queryByLabelText("檔案上限 (MB)")).toBeNull();
  });

  it("requires confirmation before irreversibly clearing transcript history", async () => {
    saveHistory([
      {
        id: "history-1",
        createdAt: "2026-07-17T00:00:00.000Z",
        sourceName: "private.wav",
        durationSeconds: 1,
        format: "text",
        text: "private transcript",
        raw: "private transcript",
      },
    ]);
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    const user = userEvent.setup();
    render(<App />);
    const clear = screen.getByRole("button", { name: "清除全部歷史" });

    await user.click(clear);

    expect(confirm).toHaveBeenCalledWith(
      "清除全部 1 筆轉錄歷史？此動作無法復原。",
    );
    expect(screen.getByText("private.wav")).toBeTruthy();
    expect(loadHistory()).toHaveLength(1);

    confirm.mockReturnValue(true);
    await user.click(clear);

    expect(screen.getByText("No records")).toBeTruthy();
    expect(loadHistory()).toEqual([]);
  });

  it("opens the audio file picker from the keyboard", async () => {
    const user = userEvent.setup();
    const { container } = render(<App />);
    const input = container.querySelector(".file-input");
    expect(input).toBeInstanceOf(HTMLInputElement);
    const click = vi.spyOn(input as HTMLInputElement, "click").mockImplementation(() => {});

    screen.getByRole("button", { name: "選擇音訊檔" }).focus();
    await user.keyboard("{Enter}");

    expect(click).toHaveBeenCalledOnce();
  });

  it("loads an audio file selected through the upload input", async () => {
    const createObjectURL = vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:meeting");
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
    const { container } = render(<App />);
    const input = container.querySelector(".file-input");
    expect(input).toBeInstanceOf(HTMLInputElement);
    const file = new File(["RIFF"], "meeting.wav", { type: "audio/wav" });

    await userEvent.upload(input as HTMLInputElement, file);

    expect(await screen.findByText("已載入 meeting.wav")).toBeTruthy();
    expect(screen.getByRole("button", { name: "meeting.wav" })).toBeTruthy();
    expect(createObjectURL).toHaveBeenCalledWith(file);
  });

  it("rejects selected files above the readiness upload limit before preview creation", async () => {
    const createObjectURL = vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:too-large");
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith("/health")) {
          return new Response(JSON.stringify({ status: "ok", model: "mock_asr", version: "dev" }));
        }
        if (url.endsWith("/ready")) {
          return new Response(
            JSON.stringify({
              status: "ok",
              model: "mock_asr",
              version: "dev",
              checks: {
                task_router_bound: true,
                recognizer_process_alive: true,
                ffmpeg_available: true,
              },
              config: {
                auth_enabled: false,
                max_upload_mb: 1,
                task_timeout: 600,
                max_concurrent_requests: 2,
                cors_enabled: true,
                cors_origins_count: 1,
              },
            }),
          );
        }
        if (url.endsWith("/v1/models")) {
          return new Response(
            JSON.stringify({
              object: "list",
              data: [{ id: "mock_asr", object: "model", owned_by: "capswriter-offline", created: 0 }],
            }),
          );
        }
        return new Response("not found", { status: 404 });
      }),
    );
    const { container } = render(<App />);
    const input = container.querySelector(".file-input");
    expect(input).toBeInstanceOf(HTMLInputElement);

    await userEvent.click(screen.getByRole("button", { name: "檢查服務" }));
    expect(await screen.findByText("1 MB / 2 slots")).toBeTruthy();

    await userEvent.upload(
      input as HTMLInputElement,
      new File([new Uint8Array(1024 * 1024 + 1)], "too-large.wav", { type: "audio/wav" }),
    );

    expect(await screen.findByText("音訊超過 server 上限 1 MB，請選擇較小檔案")).toBeTruthy();
    expect(screen.getByRole("button", { name: "選擇音訊檔" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "too-large.wav" })).toBeNull();
    expect(createObjectURL).not.toHaveBeenCalled();
  });

  it("rejects loaded audio above the readiness upload limit before transcription", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/health")) {
        return new Response(JSON.stringify({ status: "ok", model: "mock_asr", version: "dev" }));
      }
      if (url.endsWith("/ready")) {
        return new Response(
          JSON.stringify({
            status: "ok",
            model: "mock_asr",
            version: "dev",
            checks: {
              task_router_bound: true,
              recognizer_process_alive: true,
              ffmpeg_available: true,
            },
            config: {
              auth_enabled: false,
              max_upload_mb: 1,
              task_timeout: 600,
              max_concurrent_requests: 2,
              cors_enabled: true,
              cors_origins_count: 1,
            },
          }),
        );
      }
      if (url.endsWith("/v1/models")) {
        return new Response(
          JSON.stringify({
            object: "list",
            data: [{ id: "mock_asr", object: "model", owned_by: "capswriter-offline", created: 0 }],
          }),
        );
      }
      if (url.endsWith("/v1/audio/transcriptions")) {
        throw new Error("unexpected transcription request");
      }
      return new Response("not found", { status: 404 });
    });
    vi.stubGlobal("fetch", fetchMock);
    vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:oversized");
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
    const { container } = render(<App />);
    const input = container.querySelector(".file-input");
    expect(input).toBeInstanceOf(HTMLInputElement);
    const file = new File([new Uint8Array(1024 * 1024 + 1)], "oversized.wav", { type: "audio/wav" });

    await userEvent.upload(input as HTMLInputElement, file);
    expect(await screen.findByText("已載入 oversized.wav")).toBeTruthy();
    await userEvent.click(screen.getByRole("button", { name: "檢查服務" }));
    expect(await screen.findByText("1 MB / 2 slots")).toBeTruthy();

    fetchMock.mockClear();
    await userEvent.click(screen.getByRole("button", { name: "轉錄" }));

    expect(await screen.findByText("音訊超過 server 上限 1 MB，請選擇較小檔案")).toBeTruthy();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("locks audio replacement while transcription is running", async () => {
    const response = deferred<Response>();
    vi.stubGlobal("fetch", vi.fn(() => response.promise));
    vi.spyOn(URL, "createObjectURL")
      .mockReturnValueOnce("blob:meeting")
      .mockReturnValueOnce("blob:other");
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
    const { container } = render(<App />);
    const input = container.querySelector(".file-input");
    expect(input).toBeInstanceOf(HTMLInputElement);

    await userEvent.upload(
      input as HTMLInputElement,
      new File(["RIFF"], "meeting.wav", { type: "audio/wav" }),
    );
    await userEvent.click(screen.getByRole("button", { name: "轉錄" }));

    expect(await screen.findByText("轉錄中")).toBeTruthy();
    expect((screen.getByRole("button", { name: "meeting.wav" }) as HTMLButtonElement).disabled).toBe(true);
    expect((input as HTMLInputElement).disabled).toBe(true);

    fireEvent.change(input as HTMLInputElement, {
      target: { files: [new File(["RIFF"], "other.wav", { type: "audio/wav" })] },
    });

    expect(screen.getByRole("button", { name: "meeting.wav" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "other.wav" })).toBeNull();

    response.resolve(
      new Response(JSON.stringify({ text: "done" }), {
        headers: { "Content-Type": "application/json" },
      }),
    );

    expect(await screen.findByText("完成：4 字")).toBeTruthy();
  });

  it("ignores stale transcription results after cancel and audio replacement", async () => {
    const response = deferred<Response>();
    vi.stubGlobal("fetch", vi.fn(() => response.promise));
    vi.spyOn(URL, "createObjectURL")
      .mockReturnValueOnce("blob:meeting")
      .mockReturnValueOnce("blob:other");
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
    const { container } = render(<App />);
    const input = container.querySelector(".file-input");
    expect(input).toBeInstanceOf(HTMLInputElement);

    await userEvent.upload(
      input as HTMLInputElement,
      new File(["RIFF"], "meeting.wav", { type: "audio/wav" }),
    );
    await userEvent.click(screen.getByRole("button", { name: "轉錄" }));
    expect(await screen.findByText("轉錄中")).toBeTruthy();

    await userEvent.click(screen.getByRole("button", { name: "取消" }));
    expect(await screen.findByText("已取消")).toBeTruthy();
    await userEvent.upload(
      input as HTMLInputElement,
      new File(["RIFF"], "other.wav", { type: "audio/wav" }),
    );
    expect(await screen.findByText("已載入 other.wav")).toBeTruthy();

    response.resolve(
      new Response(JSON.stringify({ text: "done" }), {
        headers: { "Content-Type": "application/json" },
      }),
    );
    await Promise.resolve();
    await Promise.resolve();

    expect(screen.getByRole("button", { name: "other.wav" })).toBeTruthy();
    expect(screen.queryByText("完成：4 字")).toBeNull();
    expect(screen.queryByText("done")).toBeNull();
  });

  it("keeps drag highlight while moving inside the upload target", () => {
    render(<App />);
    const uploadTarget = screen.getByRole("button", { name: "選擇音訊檔" });
    const label = uploadTarget.querySelector("span");
    expect(label).toBeInstanceOf(HTMLSpanElement);

    fireEvent.dragEnter(uploadTarget);
    expect(uploadTarget.className).toContain("dragging");
    fireEvent.dragEnter(label as HTMLSpanElement);
    fireEvent.dragLeave(label as HTMLSpanElement);
    expect(uploadTarget.className).toContain("dragging");
    fireEvent.dragLeave(uploadTarget);
    expect(uploadTarget.className).not.toContain("dragging");
  });

  it("shows readiness diagnostics after checking the server", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith("/health")) {
          return new Response(JSON.stringify({ status: "ok", model: "mock_asr", version: "dev" }));
        }
        if (url.endsWith("/ready")) {
          return new Response(
            JSON.stringify({
              status: "ok",
              model: "mock_asr",
              version: "dev",
              checks: {
                task_router_bound: true,
                recognizer_process_alive: true,
                ffmpeg_available: true,
              },
              config: {
                auth_enabled: false,
                max_upload_mb: 100,
                task_timeout: 600,
                max_concurrent_requests: 2,
                cors_enabled: true,
                cors_origins_count: 1,
              },
            }),
          );
        }
        if (url.endsWith("/v1/models")) {
          return new Response(
            JSON.stringify({
              object: "list",
              data: [{ id: "mock_asr", object: "model", owned_by: "capswriter-offline", created: 0 }],
            }),
          );
        }
        return new Response("not found", { status: 404 });
      }),
    );

    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: "檢查服務" }));

    expect(await screen.findByText("服務正常：mock_asr vdev")).toBeTruthy();
    expect(screen.getByText("100 MB / 2 slots")).toBeTruthy();
    expect(screen.getByText("off")).toBeTruthy();
  });

  it("keeps the console mounted when readiness is missing nested checks", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith("/health")) {
          return new Response(JSON.stringify({ status: "ok", model: "mock_asr", version: "dev" }));
        }
        if (url.endsWith("/ready")) {
          return new Response(
            JSON.stringify({ status: "ok", model: "mock_asr", version: "dev" }),
          );
        }
        if (url.endsWith("/v1/models")) {
          return new Response(
            JSON.stringify({
              object: "list",
              data: [{ id: "mock_asr", object: "model", owned_by: "capswriter-offline", created: 0 }],
            }),
          );
        }
        return new Response("not found", { status: 404 });
      }),
    );

    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: "檢查服務" }));

    expect(
      await screen.findByText(
        "服務檢查部分失敗：Ready: HTTP 200: Invalid /ready response: checks must be an object",
      ),
    ).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Web Console" })).toBeTruthy();
    expect(screen.getAllByRole("region")).toHaveLength(6);
    expect(screen.getByRole("region", { name: "Server 共用設定" })).toBeTruthy();
    expect(screen.getByText("Router").parentElement?.textContent).toBe("Router-");
  });

  it("settles readiness diagnostics when rendered in React StrictMode", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith("/health")) {
          return new Response(JSON.stringify({ status: "ok", model: "mock_asr", version: "dev" }));
        }
        if (url.endsWith("/ready")) {
          return new Response(
            JSON.stringify({
              status: "ok",
              model: "mock_asr",
              version: "dev",
              checks: {
                task_router_bound: true,
                recognizer_process_alive: true,
                ffmpeg_available: true,
              },
              config: {
                auth_enabled: false,
                max_upload_mb: 100,
                task_timeout: 600,
                max_concurrent_requests: 2,
                cors_enabled: true,
                cors_origins_count: 1,
              },
            }),
          );
        }
        if (url.endsWith("/v1/models")) {
          return new Response(
            JSON.stringify({
              object: "list",
              data: [{ id: "mock_asr", object: "model", owned_by: "capswriter-offline", created: 0 }],
            }),
          );
        }
        return new Response("not found", { status: 404 });
      }),
    );

    render(
      <StrictMode>
        <App />
      </StrictMode>,
    );
    await userEvent.click(screen.getByRole("button", { name: "檢查服務" }));

    expect(await screen.findByText("服務正常：mock_asr vdev")).toBeTruthy();
    expect(screen.getByText("100 MB / 2 slots")).toBeTruthy();
  });

  it("keeps partial readiness diagnostics when model listing fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith("/health")) {
          return new Response(JSON.stringify({ status: "ok", model: "mock_asr", version: "dev" }));
        }
        if (url.endsWith("/ready")) {
          return new Response(
            JSON.stringify({
              status: "ok",
              model: "mock_asr",
              version: "dev",
              checks: {
                task_router_bound: true,
                recognizer_process_alive: true,
                ffmpeg_available: true,
              },
              config: {
                auth_enabled: true,
                max_upload_mb: 100,
                task_timeout: 600,
                max_concurrent_requests: 2,
                cors_enabled: true,
                cors_origins_count: 1,
              },
            }),
          );
        }
        if (url.endsWith("/v1/models")) {
          return new Response(JSON.stringify({ detail: "Missing API key" }), { status: 401 });
        }
        return new Response("not found", { status: 404 });
      }),
    );

    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: "檢查服務" }));

    expect(await screen.findByText("服務檢查部分失敗：Models: HTTP 401: Missing API key")).toBeTruthy();
    expect(screen.getByText("100 MB / 2 slots")).toBeTruthy();
    expect(screen.getByText("enabled")).toBeTruthy();
  });

  it("redacts a peer-reflected API key before rendering diagnostic errors", async () => {
    const apiKey = "sk-render-reflection-secret";
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(
          JSON.stringify({
            error: {
              message: `Rejected Authorization: Bearer ${apiKey}\u0000\u001b[31m`,
            },
          }),
          { status: 401 },
        ),
      ),
    );

    render(<App />);
    await userEvent.type(screen.getByLabelText("API key"), apiKey);
    await userEvent.click(screen.getByRole("button", { name: "檢查服務" }));

    const status = await screen.findByText(/服務檢查部分失敗/);
    expect(status.textContent).toContain("Bearer [REDACTED]");
    expect(status.textContent).not.toContain(apiKey);
    expect(status.textContent).not.toMatch(/[\u0000-\u001f\u007f-\u009f]/u);
    expect((screen.getByLabelText("API key") as HTMLInputElement).value).toBe(apiKey);
  });

  it("aborts stale server diagnostics when a newer check starts", async () => {
    const requests: Array<{
      pending: ReturnType<typeof deferred<Response>>;
      signal?: AbortSignal | null;
      url: string;
    }> = [];
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
        const pending = deferred<Response>();
        init?.signal?.addEventListener("abort", () => {
          pending.reject(new DOMException("aborted", "AbortError"));
        });
        requests.push({ pending, signal: init?.signal, url: String(input) });
        return pending.promise;
      }),
    );

    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: "檢查服務" }));
    expect(requests).toHaveLength(3);

    await userEvent.click(screen.getByRole("button", { name: "檢查服務" }));
    expect(requests).toHaveLength(6);
    expect(requests.slice(0, 3).every((request) => request.signal?.aborted)).toBe(true);
    expect(requests.slice(3).every((request) => request.signal?.aborted)).toBe(false);

    for (const request of requests.slice(3)) {
      if (request.url.endsWith("/health")) {
        request.pending.resolve(new Response(JSON.stringify({ status: "ok", model: "new_asr", version: "dev" })));
      } else if (request.url.endsWith("/ready")) {
        request.pending.resolve(
          new Response(
            JSON.stringify({
              status: "ok",
              model: "new_asr",
              version: "dev",
              checks: {
                task_router_bound: true,
                recognizer_process_alive: true,
                ffmpeg_available: true,
              },
              config: {
                auth_enabled: false,
                max_upload_mb: 100,
                task_timeout: 600,
                max_concurrent_requests: 2,
                cors_enabled: true,
                cors_origins_count: 1,
              },
            }),
          ),
        );
      } else if (request.url.endsWith("/v1/models")) {
        request.pending.resolve(
          new Response(
            JSON.stringify({
              object: "list",
              data: [{ id: "new_asr", object: "model", owned_by: "capswriter-offline", created: 0 }],
            }),
          ),
        );
      }
    }

    expect(await screen.findByText("服務正常：new_asr vdev")).toBeTruthy();
    expect(screen.getAllByText("new_asr").length).toBeGreaterThan(0);
  });

  it("stops active recording resources on unmount", async () => {
    const track = { stop: vi.fn() } as unknown as MediaStreamTrack;
    const stream = { getTracks: vi.fn(() => [track]) } as unknown as MediaStream;
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia: vi.fn(async () => stream) },
    });

    const recorders: MockRecorder[] = [];
    class MockRecorder {
      static isTypeSupported = vi.fn(() => true);
      state: RecordingState = "inactive";
      mimeType = "audio/webm";
      ondataavailable: ((event: BlobEvent) => void) | null = null;
      onstop: ((event: Event) => void) | null = null;

      constructor() {
        recorders.push(this);
      }

      start = vi.fn(() => {
        this.state = "recording";
      });

      stop = vi.fn(() => {
        this.state = "inactive";
        this.onstop?.(new Event("stop"));
      });
    }
    vi.stubGlobal("MediaRecorder", MockRecorder);

    const { unmount } = render(<App />);
    await userEvent.click(screen.getByRole("button", { name: "錄音" }));

    expect(await screen.findByText("錄音中")).toBeTruthy();
    expect(recorders).toHaveLength(1);

    unmount();

    expect(recorders[0].stop).toHaveBeenCalledTimes(1);
    expect(track.stop).toHaveBeenCalledTimes(1);
  });

  it("coalesces recording starts while microphone permission is pending", async () => {
    const pendingStream = deferred<MediaStream>();
    const track = { stop: vi.fn() } as unknown as MediaStreamTrack;
    const stream = { getTracks: vi.fn(() => [track]) } as unknown as MediaStream;
    const getUserMedia = vi.fn(() => pendingStream.promise);
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia },
    });

    const recorders: MockRecorder[] = [];
    class MockRecorder {
      static isTypeSupported = vi.fn(() => true);
      state: RecordingState = "inactive";
      mimeType = "audio/webm";
      ondataavailable: ((event: BlobEvent) => void) | null = null;
      onstop: ((event: Event) => void) | null = null;

      constructor() {
        recorders.push(this);
      }

      start = vi.fn(() => {
        this.state = "recording";
      });

      stop = vi.fn(() => {
        this.state = "inactive";
        this.onstop?.(new Event("stop"));
      });
    }
    vi.stubGlobal("MediaRecorder", MockRecorder);

    const { unmount } = render(<App />);
    const record = screen.getByRole("button", { name: "錄音" });
    act(() => {
      record.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      record.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(getUserMedia).toHaveBeenCalledOnce();
    expect((screen.getByRole("button", { name: "準備中" }) as HTMLButtonElement).disabled).toBe(true);

    await act(async () => {
      pendingStream.resolve(stream);
      await pendingStream.promise;
    });

    expect(await screen.findByText("錄音中")).toBeTruthy();
    expect(recorders).toHaveLength(1);

    unmount();
    expect(track.stop).toHaveBeenCalledOnce();
  });

  it("waits for recorder shutdown and blocks transcription before allowing another start", async () => {
    const firstTrack = { stop: vi.fn() } as unknown as MediaStreamTrack;
    const secondTrack = { stop: vi.fn() } as unknown as MediaStreamTrack;
    const streams = [
      { getTracks: vi.fn(() => [firstTrack]) } as unknown as MediaStream,
      { getTracks: vi.fn(() => [secondTrack]) } as unknown as MediaStream,
    ];
    const getUserMedia = vi
      .fn<() => Promise<MediaStream>>()
      .mockResolvedValueOnce(streams[0])
      .mockResolvedValueOnce(streams[1]);
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia },
    });

    const recorders: MockRecorder[] = [];
    class MockRecorder {
      static isTypeSupported = vi.fn(() => true);
      state: RecordingState = "inactive";
      mimeType = "audio/webm";
      ondataavailable: ((event: BlobEvent) => void) | null = null;
      onstop: ((event: Event) => void) | null = null;

      constructor() {
        recorders.push(this);
      }

      start = vi.fn(() => {
        this.state = "recording";
      });

      stop = vi.fn(() => {
        this.state = "inactive";
      });

      finishStop() {
        this.onstop?.(new Event("stop"));
      }
    }
    vi.stubGlobal("MediaRecorder", MockRecorder);
    vi.spyOn(URL, "createObjectURL")
      .mockReturnValueOnce("blob:existing")
      .mockReturnValueOnce("blob:recording");
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    const { container, unmount } = render(<App />);
    const input = container.querySelector(".file-input");
    expect(input).toBeInstanceOf(HTMLInputElement);
    await userEvent.upload(
      input as HTMLInputElement,
      new File(["RIFF"], "existing.wav", { type: "audio/wav" }),
    );

    await userEvent.click(screen.getByRole("button", { name: "錄音" }));
    expect(await screen.findByText("錄音中")).toBeTruthy();
    expect((screen.getByRole("button", { name: "轉錄" }) as HTMLButtonElement).disabled).toBe(true);
    expect((input as HTMLInputElement).disabled).toBe(true);

    await userEvent.click(
      within(screen.getByRole("region", { name: "音訊" })).getByRole("button", {
        name: "停止",
      }),
    );
    const stopping = screen.getByRole("button", { name: "停止中" });
    expect((stopping as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole("button", { name: "轉錄" }) as HTMLButtonElement).disabled).toBe(true);
    stopping.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    expect(getUserMedia).toHaveBeenCalledOnce();
    expect(fetchMock).not.toHaveBeenCalled();

    act(() => recorders[0].finishStop());

    expect(firstTrack.stop).toHaveBeenCalledOnce();
    expect((screen.getByRole("button", { name: "轉錄" }) as HTMLButtonElement).disabled).toBe(false);
    await userEvent.click(screen.getByRole("button", { name: "錄音" }));
    expect(await screen.findByText("錄音中")).toBeTruthy();
    expect(getUserMedia).toHaveBeenCalledTimes(2);
    expect(recorders).toHaveLength(2);

    unmount();
    expect(secondTrack.stop).toHaveBeenCalledOnce();
  });

  it("releases the microphone stream when recorder setup fails", async () => {
    const track = { stop: vi.fn() } as unknown as MediaStreamTrack;
    const stream = { getTracks: vi.fn(() => [track]) } as unknown as MediaStream;
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia: vi.fn(async () => stream) },
    });

    class FailingRecorder {
      static isTypeSupported = vi.fn(() => true);

      constructor() {
        throw new Error("recorder unavailable");
      }
    }
    vi.stubGlobal("MediaRecorder", FailingRecorder);

    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: "錄音" }));

    expect(await screen.findByText("recorder unavailable")).toBeTruthy();
    expect(track.stop).toHaveBeenCalledTimes(1);
  });

  it("shows an error when transcript copy is denied", async () => {
    const writeText = vi.fn(async () => {
      throw new Error("clipboard denied");
    });
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });

    const { container } = render(<App />);
    const output = container.querySelector(".transcript-output");
    expect(output).toBeInstanceOf(HTMLTextAreaElement);

    await userEvent.type(output as HTMLTextAreaElement, "hello");
    await userEvent.click(screen.getByRole("button", { name: "複製" }));

    expect(writeText).toHaveBeenCalledWith("hello");
    expect(await screen.findByText("clipboard denied")).toBeTruthy();
  });
});
