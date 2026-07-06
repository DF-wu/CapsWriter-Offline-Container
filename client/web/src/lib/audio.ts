export interface BrowserAudio {
  blob: Blob;
  name: string;
  objectUrl: string;
  durationSeconds: number | null;
}

const RECORDER_TYPES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/ogg;codecs=opus",
  "audio/mp4",
];

export function chooseRecorderMimeType(): string {
  if (typeof MediaRecorder === "undefined") {
    return "";
  }
  return RECORDER_TYPES.find((type) => MediaRecorder.isTypeSupported(type)) ?? "";
}

export function extensionForMimeType(mimeType: string): string {
  if (mimeType.includes("ogg")) return "ogg";
  if (mimeType.includes("mp4")) return "m4a";
  if (mimeType.includes("wav")) return "wav";
  return "webm";
}

export function formatDuration(seconds: number | null): string {
  if (seconds === null || Number.isNaN(seconds)) {
    return "--:--";
  }
  const safe = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(safe / 60);
  const rest = safe % 60;
  return `${minutes}:${rest.toString().padStart(2, "0")}`;
}

export function revokeAudio(audio: BrowserAudio | null): void {
  if (audio?.objectUrl) {
    URL.revokeObjectURL(audio.objectUrl);
  }
}

export function fileToBrowserAudio(file: File): BrowserAudio {
  return {
    blob: file,
    name: file.name,
    objectUrl: URL.createObjectURL(file),
    durationSeconds: null,
  };
}
