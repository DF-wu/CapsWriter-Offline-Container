#!/bin/sh
set -eu

js_escape() {
  printf '%s' "$1" | awk '
    function escape_line(value, escaped, position, char) {
      escaped = ""
      for (position = 1; position <= length(value); position++) {
        char = substr(value, position, 1)
        if (char == "\\") {
          escaped = escaped "\\\\"
        } else if (char == "\"") {
          escaped = escaped "\\\""
        } else if (char == "\r") {
          escaped = escaped "\\r"
        } else {
          escaped = escaped char
        }
      }
      return escaped
    }
    BEGIN { first = 1 }
    {
      if (!first) {
        printf "\\n"
      }
      printf "%s", escape_line($0)
      first = 0
    }
  '
}

config_path="${CAPSWRITER_WEB_CONFIG_PATH:-/usr/share/nginx/html/config.js}"
api_base="$(js_escape "${CAPSWRITER_WEB_API_BASE:-http://localhost:6017}")"
api_key="$(js_escape "${CAPSWRITER_WEB_API_KEY:-}")"
allow_public_api_key="${CAPSWRITER_WEB_ALLOW_PUBLIC_API_KEY:-false}"
model="$(js_escape "${CAPSWRITER_WEB_MODEL:-whisper-1}")"
language="$(js_escape "${CAPSWRITER_WEB_LANGUAGE:-}")"
prompt="$(js_escape "${CAPSWRITER_WEB_PROMPT:-}")"
response_format="$(js_escape "${CAPSWRITER_WEB_RESPONSE_FORMAT:-verbose_json}")"

case "$allow_public_api_key" in
  true|TRUE|True|1|yes|YES|Yes|on|ON|On)
    allow_public_api_key=true
    ;;
  false|FALSE|False|0|no|NO|No|off|OFF|Off|"")
    allow_public_api_key=false
    ;;
  *)
    echo "CAPSWRITER_WEB_ALLOW_PUBLIC_API_KEY must be true or false" >&2
    exit 64
    ;;
esac

if [ -n "${CAPSWRITER_WEB_API_KEY:-}" ] && [ "$allow_public_api_key" != "true" ]; then
  echo "CAPSWRITER_WEB_API_KEY is written to public /config.js; set CAPSWRITER_WEB_ALLOW_PUBLIC_API_KEY=true to opt in" >&2
  exit 64
fi

mkdir -p "$(dirname "$config_path")"
cat > "$config_path" <<EOF
window.__CAPSWRITER_WEB_CONFIG__ = {
  baseUrl: "$api_base",
  apiKey: "$api_key",
  model: "$model",
  language: "$language",
  prompt: "$prompt",
  responseFormat: "$response_format"
};
EOF
chmod 0644 "$config_path"
