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
- editable provider parameters and reusable templates,
- advanced provider-specific headers, ASR form fields, and JSON body overrides for OpenAI-compatible variants,
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

Each provider profile supports extra headers. ASR also supports extra multipart form fields, while Chat Completions, Responses, and TTS support extra JSON body fields merged into the request payload.

Streaming support uses server-sent events:

- Chat Completions: `choices[0].delta.content`
- Responses: `response.output_text.delta`

CapsWriter's local HTTP API is the default ASR template:

```text
http://localhost:6017/v1
```

On physical phones, replace `localhost` with the server LAN IP.
For Android emulators, use the built-in `Android Emulator Host` template to route provider URLs through `10.0.2.2`.

## Development

```bash
cd apps/capswriter-client
npm install
npm run typecheck
npx expo-doctor
npm run verify:android-config
npm run verify:android-build
npm run verify:android-runtime
npm run web
```

Android:

```bash
npm run android:go          # Expo Go
npm run prebuild:android    # verifies native Android project generation
npm run verify:android-build # builds :app:assembleDebug when JDK/Android SDK exist
npm run verify:android-runtime # installs and launches the APK on an online adb device
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
- Android native config allows cleartext HTTP for local CapsWriter/LAN endpoints and pins the generated Gradle wrapper to `gradle-8.14.3-bin.zip`, avoiding the React Native Gradle plugin Foojay resolver incompatibility seen with Gradle 9.x.
- `npm run verify:android-build` requires JDK 17+ plus an Android SDK exposed through `ANDROID_HOME`, `ANDROID_SDK_ROOT`, or `android/local.properties` `sdk.dir`.
- `npm run verify:android-runtime` requires one online adb device or emulator. Headless or VM hosts without an online adb device or emulator acceleration cannot run this check.
- Background audio services are disabled through the `expo-audio` config plugin.
- Native API keys use `expo-secure-store` when available.
- Web API keys use `localStorage`, which is convenient for local work but not suitable for shared machines.
