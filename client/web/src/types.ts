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

export interface ReadinessResponse {
  status: "ok" | "degraded" | string;
  model: string;
  version: string;
  checks: {
    task_router_bound: boolean;
    ffmpeg_available: boolean;
    [key: string]: boolean;
  };
  config: {
    auth_enabled: boolean;
    max_upload_mb: number;
    task_timeout: number;
    max_concurrent_requests: number;
    cors_enabled: boolean;
    cors_origins_count: number;
    [key: string]: boolean | number;
  };
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
