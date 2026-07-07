import { fireEvent, render, screen } from "@testing-library/react";
import type * as ReactModule from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

type SpeakTextOptions = {
  text: string;
  voiceURI: string;
  rate: number;
  pitch: number;
  onEnd: () => void;
  onError: (message: string) => void;
};

const stateProbe = vi.hoisted(() => ({
  unmounted: false,
  lateSetState: vi.fn(),
}));

const speechMock = vi.hoisted(() => ({
  loadVoices: vi.fn<() => Promise<SpeechSynthesisVoice[]>>(),
  speakText: vi.fn<(options: SpeakTextOptions) => void>(),
}));

vi.mock("react", async (importOriginal) => {
  const actual = await importOriginal<typeof ReactModule>();
  return {
    ...actual,
    useState<T>(
      initialState: T | (() => T),
    ): [T, ReactModule.Dispatch<ReactModule.SetStateAction<T>>] {
      const [value, setValue] = actual.useState(initialState);
      const guardedSetValue: ReactModule.Dispatch<ReactModule.SetStateAction<T>> = (next) => {
        if (stateProbe.unmounted) {
          stateProbe.lateSetState();
        }
        setValue(next);
      };
      return [value, guardedSetValue];
    },
  };
});

vi.mock("./lib/speech", () => speechMock);

import App from "./App";

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

describe("App voice loading lifecycle", () => {
  const originalMediaDevices = Object.getOwnPropertyDescriptor(navigator, "mediaDevices");

  afterEach(() => {
    if (originalMediaDevices) {
      Object.defineProperty(navigator, "mediaDevices", originalMediaDevices);
    } else {
      delete (navigator as unknown as { mediaDevices?: MediaDevices }).mediaDevices;
    }
    stateProbe.unmounted = false;
    stateProbe.lateSetState.mockClear();
    speechMock.loadVoices.mockReset();
    speechMock.speakText.mockReset();
    vi.unstubAllGlobals();
  });

  it("ignores voice loading that resolves after unmount", async () => {
    const pendingVoices = deferred<SpeechSynthesisVoice[]>();
    speechMock.loadVoices.mockReturnValueOnce(pendingVoices.promise);

    const { unmount } = render(<App />);
    expect(speechMock.loadVoices).toHaveBeenCalledOnce();

    stateProbe.unmounted = true;
    unmount();
    pendingVoices.resolve([
      {
        voiceURI: "voice-1",
        name: "Voice 1",
        lang: "en-US",
      } as SpeechSynthesisVoice,
    ]);
    await pendingVoices.promise;
    await Promise.resolve();

    expect(stateProbe.lateSetState).not.toHaveBeenCalled();
  });

  it("ignores speech callbacks after unmount", async () => {
    speechMock.loadVoices.mockResolvedValueOnce([
      {
        voiceURI: "voice-1",
        name: "Voice 1",
        lang: "en-US",
      } as SpeechSynthesisVoice,
    ]);

    const { unmount } = render(<App />);
    await screen.findByRole("option", { name: "Voice 1 (en-US)" });
    fireEvent.change(screen.getByLabelText("文字"), { target: { value: "hello" } });
    fireEvent.click(screen.getByRole("button", { name: "播放" }));
    expect(speechMock.speakText).toHaveBeenCalledOnce();

    const options = speechMock.speakText.mock.calls[0][0];
    stateProbe.unmounted = true;
    unmount();
    options.onEnd();
    options.onError("late speech error");

    expect(stateProbe.lateSetState).not.toHaveBeenCalled();
  });

  it("stops media streams that resolve after unmount", async () => {
    const pendingStream = deferred<MediaStream>();
    const track = { stop: vi.fn() } as unknown as MediaStreamTrack;
    const stream = { getTracks: vi.fn(() => [track]) } as unknown as MediaStream;
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia: vi.fn(() => pendingStream.promise) },
    });

    const recorders: unknown[] = [];
    class MockRecorder {
      static isTypeSupported = vi.fn(() => true);

      constructor() {
        recorders.push(this);
      }
    }
    vi.stubGlobal("MediaRecorder", MockRecorder);
    speechMock.loadVoices.mockResolvedValueOnce([]);

    const { unmount } = render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "錄音" }));

    stateProbe.unmounted = true;
    unmount();
    pendingStream.resolve(stream);
    await pendingStream.promise;
    await Promise.resolve();

    expect(track.stop).toHaveBeenCalledOnce();
    expect(recorders).toHaveLength(0);
    expect(stateProbe.lateSetState).not.toHaveBeenCalled();
  });

  it("ignores transcription results that resolve after unmount", async () => {
    const pendingResponse = deferred<Response>();
    vi.stubGlobal("fetch", vi.fn(() => pendingResponse.promise));
    vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:meeting");
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
    speechMock.loadVoices.mockResolvedValueOnce([]);

    const { container, unmount } = render(<App />);
    const input = container.querySelector(".file-input");
    expect(input).toBeInstanceOf(HTMLInputElement);
    fireEvent.change(input as HTMLInputElement, {
      target: { files: [new File(["RIFF"], "meeting.wav", { type: "audio/wav" })] },
    });
    await screen.findByText("已載入 meeting.wav");
    fireEvent.click(screen.getByRole("button", { name: "轉錄" }));
    await screen.findByText("轉錄中");

    stateProbe.unmounted = true;
    unmount();
    pendingResponse.resolve(
      new Response(JSON.stringify({ text: "done" }), {
        headers: { "Content-Type": "application/json" },
      }),
    );
    await pendingResponse.promise;
    await Promise.resolve();
    await Promise.resolve();

    expect(stateProbe.lateSetState).not.toHaveBeenCalled();
  });
});
