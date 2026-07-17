/// <reference types="vite/client" />

import type { ApiSettings } from "./types";

declare global {
  interface Window {
    __CAPSWRITER_WEB_CONFIG__?: Partial<ApiSettings>;
  }
}

export {};
