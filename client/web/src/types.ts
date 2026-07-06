export type ResponseFormat = "json" | "text" | "srt" | "verbose_json" | "vtt";

export interface ApiSettings {
  baseUrl: string;
  apiKey: string;
  model: string;
  language: string;
  prompt: string;
  responseFormat: ResponseFormat;
}

export interface HealthResponse {
  status: string;
  model: string;
  version: string;
}

export interface ModelListResponse {
  object: "list";
  data: Array<{
    id: string;
    object: string;
    owned_by: string;
    created: number;
  }>;
}

export interface VerboseSegment {
  id: number;
  seek?: number;
  start: number;
  end: number;
  text: string;
}

export interface VerboseWord {
  word: string;
  start: number;
  end: number;
}

export interface VerboseTranscription {
  task?: string;
  language?: string | null;
  duration?: number;
  text: string;
  segments?: VerboseSegment[];
  words?: VerboseWord[];
}

export interface TranscriptionResult {
  text: string;
  format: ResponseFormat;
  raw: string | VerboseTranscription | { text?: string };
  contentType: string;
}

export interface TranscriptRecord {
  id: string;
  createdAt: string;
  sourceName: string;
  durationSeconds: number | null;
  format: ResponseFormat;
  text: string;
  raw: TranscriptionResult["raw"];
}
