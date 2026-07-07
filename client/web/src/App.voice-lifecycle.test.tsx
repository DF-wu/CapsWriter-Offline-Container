import { render } from "@testing-library/react";
import type * as ReactModule from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

const stateProbe = vi.hoisted(() => ({
  unmounted: false,
  lateSetState: vi.fn(),
}));

const speechMock = vi.hoisted(() => ({
  loadVoices: vi.fn<() => Promise<SpeechSynthesisVoice[]>>(),
  speakText: vi.fn(),
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
});
