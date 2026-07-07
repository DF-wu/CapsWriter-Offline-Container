import type { ResponseFormat, TranscriptionResult } from "../types";

const WINDOWS_RESERVED_FILENAME = /^(con|prn|aux|nul|com[1-9]|lpt[1-9])(\..*)?$/i;

export function extensionForFormat(format: ResponseFormat): string {
  if (format === "verbose_json" || format === "json") return "json";
  if (format === "srt") return "srt";
  if (format === "vtt") return "vtt";
  return "txt";
}

export function serialiseResult(result: TranscriptionResult): string {
  return typeof result.raw === "string"
    ? result.raw
    : JSON.stringify(result.raw, null, 2);
}

export function safeDownloadFilename(filename: string): string {
  const safe = filename
    .replace(/[\x00-\x1f\x7f/\\<>:"|?*]+/g, "-")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^[ .-]+|[ .-]+$/g, "");
  if (!safe) return "capswriter-download.txt";
  return WINDOWS_RESERVED_FILENAME.test(safe) ? `capswriter-${safe}` : safe;
}

export function downloadText(filename: string, text: string, type = "text/plain;charset=utf-8"): void {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = safeDownloadFilename(filename);
  try {
    document.body.appendChild(link);
    link.click();
  } finally {
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  }
}

export function timestampSlug(date = new Date()): string {
  return date.toISOString().replace(/[:.]/g, "-");
}
