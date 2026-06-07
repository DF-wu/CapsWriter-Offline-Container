# Universal ASR Client

This repository now includes a universal Expo client at [`apps/capswriter-client`](../apps/capswriter-client).

## Product Scope

The client is designed as an operational workbench, not a landing page:

- browser UI for desktop and mobile users,
- Android native app path through Expo and EAS,
- optional iOS support through the same codebase,
- CapsWriter local ASR integration,
- OpenAI-compatible ASR, chat, responses, and TTS integrations,
- streaming Chat Completions and Responses output,
- Markdown export/share for transcripts, raw ASR payloads, and conversations,
- editable provider parameters and reusable templates.
- per-provider diagnostics for ASR, conversation, and TTS `/v1/models` endpoints, with model IDs selectable into the matching provider settings.

## Architecture

```
apps/capswriter-client/
├── app/                    Expo Router entry
├── src/components/         main workbench UI
├── src/data/templates.ts   provider templates
├── src/lib/                OpenAI-compatible API, platform, storage
├── src/state/              persisted settings provider
└── src/types/              shared client types
```

The app calls standard OpenAI-compatible paths:

| Feature | Endpoint |
|---|---|
| ASR | `POST /v1/audio/transcriptions` |
| Chat Completions | `POST /v1/chat/completions` |
| Responses | `POST /v1/responses` |
| TTS | `POST /v1/audio/speech` |
| Provider probe | `GET /v1/models` |

Streaming support uses server-sent events:

- Chat Completions: `choices[0].delta.content`
- Responses: `response.output_text.delta`

CapsWriter's local HTTP API is the default ASR template:

```text
http://localhost:6017/v1
```

On physical phones, replace `localhost` with the server LAN IP.

## Development

```bash
cd apps/capswriter-client
npm install
npm run typecheck
npx expo-doctor
npm run web
```

Android:

```bash
npm run android:go          # Expo Go
npm run prebuild:android    # verifies native Android project generation
npm run android             # native Android run
npx eas build --platform android --profile preview
```

iOS is optional:

```bash
npm run ios:go              # Expo Go
npm run ios:native          # native run on macOS
```

## Current Notes

- Mock provider scripts under `apps/capswriter-client/scripts/` verify ASR uploads, TTS speech, provider diagnostics, and both streaming API modes without calling external services.
- Android native config disables background audio services; Expo config resolves to `RECORD_AUDIO` and `MODIFY_AUDIO_SETTINGS` only.
- Native API keys use `expo-secure-store` when available.
- Web API keys use `localStorage`, which is convenient for local work but not suitable for shared machines.
