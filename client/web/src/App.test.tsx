import userEvent from "@testing-library/user-event";
import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

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
    render(<App />);

    expect(screen.getByRole("heading", { name: "Web Console" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "連線" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "音訊" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "轉錄" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "TTS" })).toBeTruthy();
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
