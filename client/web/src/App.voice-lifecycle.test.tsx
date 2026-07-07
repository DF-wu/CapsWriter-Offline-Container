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
  afterEach(() => {
    stateProbe.unmounted = false;
    stateProbe.lateSetState.mockClear();
    speechMock.loadVoices.mockReset();
    speechMock.speakText.mockReset();
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
});
