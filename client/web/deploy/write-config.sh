#!/bin/sh
set -eu

js_escape() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

api_base="$(js_escape "${CAPSWRITER_WEB_API_BASE:-http://localhost:6017}")"
api_key="$(js_escape "${CAPSWRITER_WEB_API_KEY:-}")"
model="$(js_escape "${CAPSWRITER_WEB_MODEL:-whisper-1}")"
language="$(js_escape "${CAPSWRITER_WEB_LANGUAGE:-}")"
prompt="$(js_escape "${CAPSWRITER_WEB_PROMPT:-}")"
response_format="$(js_escape "${CAPSWRITER_WEB_RESPONSE_FORMAT:-verbose_json}")"

cat > /usr/share/nginx/html/config.js <<EOF
window.__CAPSWRITER_WEB_CONFIG__ = {
  baseUrl: "$api_base",
  apiKey: "$api_key",
  model: "$model",
  language: "$language",
  prompt: "$prompt",
  responseFormat: "$response_format"
};
EOF
chmod 0644 /usr/share/nginx/html/config.js
