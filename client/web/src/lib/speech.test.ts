import { afterEach, describe, expect, it, vi } from "vitest";
import { loadVoices } from "./speech";

const originalSpeechSynthesis = Object.getOwnPropertyDescriptor(window, "speechSynthesis");

function installSpeechSynthesis(voices: SpeechSynthesisVoice[] = []) {
  const synth = {
    onvoiceschanged: null as ((this: SpeechSynthesis, ev: Event) => unknown) | null,
    getVoices: vi.fn(() => voices),
  };
  Object.defineProperty(window, "speechSynthesis", {
    configurable: true,
    value: synth,
  });
  return synth;
}

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
  if (originalSpeechSynthesis) {
    Object.defineProperty(window, "speechSynthesis", originalSpeechSynthesis);
  } else {
    delete (window as unknown as { speechSynthesis?: SpeechSynthesis }).speechSynthesis;
  }
});

describe("loadVoices", () => {
  it("restores the previous voiceschanged handler after timeout", async () => {
    vi.useFakeTimers();
    const previous = vi.fn();
    const synth = installSpeechSynthesis([]);
    synth.onvoiceschanged = previous;

    const promise = loadVoices();
    expect(synth.onvoiceschanged).not.toBe(previous);

    vi.advanceTimersByTime(500);
    await expect(promise).resolves.toEqual([]);
    expect(synth.onvoiceschanged).toBe(previous);
    expect(previous).not.toHaveBeenCalled();
  });

  it("calls and restores the previous voiceschanged handler on voice events", async () => {
    const voices = [{ voiceURI: "voice-1", lang: "en-US" } as SpeechSynthesisVoice];
    const previous = vi.fn();
    const synth = installSpeechSynthesis(voices);
    synth.onvoiceschanged = previous;

    const promise = loadVoices();
    synth.onvoiceschanged?.call(synth as unknown as SpeechSynthesis, new Event("voiceschanged"));

    await expect(promise).resolves.toEqual(voices);
    expect(previous).toHaveBeenCalledOnce();
    expect(synth.onvoiceschanged).toBe(previous);
  });
});
