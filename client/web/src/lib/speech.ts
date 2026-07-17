export function loadVoices(): Promise<SpeechSynthesisVoice[]> {
  if (!("speechSynthesis" in window)) {
    return Promise.resolve([]);
  }
  const synth = window.speechSynthesis;
  const existing = synth.getVoices();
  if (existing.length > 0) {
    return Promise.resolve(existing);
  }
  return new Promise((resolve) => {
    const previousHandler = synth.onvoiceschanged;
    let settled = false;
    let timeout = 0;
    const finish = () => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timeout);
      synth.onvoiceschanged = previousHandler;
      resolve(synth.getVoices());
    };
    timeout = window.setTimeout(finish, 500);
    synth.onvoiceschanged = function onVoicesChanged(event: Event) {
      try {
        previousHandler?.call(synth, event);
      } finally {
        finish();
      }
    };
  });
}

export function speakText(options: {
  text: string;
  voiceURI: string;
  rate: number;
  pitch: number;
  onEnd: () => void;
  onError: (message: string) => void;
}): void {
  const text = options.text.trim();
  if (!text) {
    options.onError("沒有可播放的文字");
    return;
  }
  if (!("speechSynthesis" in window)) {
    options.onError("此瀏覽器不支援語音合成");
    return;
  }
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  const voice = window.speechSynthesis
    .getVoices()
    .find((item) => item.voiceURI === options.voiceURI);
  if (voice) {
    utterance.voice = voice;
    utterance.lang = voice.lang;
  }
  utterance.rate = options.rate;
  utterance.pitch = options.pitch;
  utterance.onend = options.onEnd;
  utterance.onerror = () => options.onError("播放失敗");
  window.speechSynthesis.speak(utterance);
}
