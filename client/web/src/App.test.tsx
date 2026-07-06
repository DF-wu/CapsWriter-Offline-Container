import userEvent from "@testing-library/user-event";
import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

describe("App", () => {
  const originalMediaDevices = Object.getOwnPropertyDescriptor(navigator, "mediaDevices");

  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    if (originalMediaDevices) {
      Object.defineProperty(navigator, "mediaDevices", originalMediaDevices);
    } else {
      delete (navigator as unknown as { mediaDevices?: MediaDevices }).mediaDevices;
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
});
