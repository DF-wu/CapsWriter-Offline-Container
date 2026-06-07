# CapsWriter Client

Universal ASR workbench for CapsWriter and OpenAI-compatible services.

This app targets:

- Web browsers on desktop and mobile
- Android native builds through Expo
- iOS through Expo Go or EAS builds

## Features

- Record microphone audio and send it to `POST /v1/audio/transcriptions`
- Upload audio/video files for transcription
- Use CapsWriter's local Whisper-compatible HTTP API
- Use any OpenAI-compatible ASR provider
- Send transcript text to either `POST /v1/chat/completions` or `POST /v1/responses`
- Stream Chat Completions and Responses replies over server-sent events
- Generate speech with `POST /v1/audio/speech`
- Check ASR, chat, and TTS providers with `GET /v1/models`
- Persist provider settings and API keys locally
- Apply provider templates for CapsWriter local, cloud-compatible, and LM Studio/Ollama-style setups

## Run

```bash
cd apps/capswriter-client
npm install
npm run web
```

Android with Expo Go:

```bash
npm run android:go
```

Android native run with a local Android SDK/device:

```bash
npm run prebuild:android
npm run android
```

Android preview APK with EAS:

```bash
npx eas build --platform android --profile preview
```

iOS is optional and uses the same app:

```bash
npm run ios:go       # Expo Go
npm run ios:native   # native run on macOS
```

## CapsWriter Setup

Start the CapsWriter server HTTP API:

```bash
CAPSWRITER_HTTP_API_ENABLE=true \
CAPSWRITER_HTTP_API_BIND=0.0.0.0 \
CAPSWRITER_HTTP_API_PORT=6017 \
python start_server_docker.py
```

Then use this ASR base URL in the app:

```text
http://YOUR_SERVER_IP:6017/v1
```

For Android emulators, `localhost` points at the emulator itself. Use your LAN IP or `10.0.2.2` when the server is on the host machine.

## Provider Settings

The app exposes the parameters users normally need to tune:

- ASR: base URL, API key, model, `response_format`, language, prompt, temperature, timeout
- Conversation: base URL, API key, API mode, model, system prompt, temperature, top P, penalties, max output tokens, history, streaming, timeout
- TTS: base URL, API key, model, voice, output format, speed, instructions, timeout

Streaming is enabled per conversation profile. Chat Completions reads `choices[].delta.content`; Responses reads `response.output_text.delta`.

Provider checks run independently for the ASR, conversation, and TTS base URLs. Each check reports HTTP status and model IDs returned by `/v1/models`; returned model IDs can be applied directly to that provider.

## Verification

Run static checks:

```bash
npm run typecheck
npx expo-doctor
```

Run the browser smoke tests from the repository root:

```bash
python /home/df/.agents/skills/webapp-testing/scripts/with_server.py \
  --server "cd apps/capswriter-client && BROWSER=none npx expo start --web --port 8081 --host localhost" --port 8081 \
  --timeout 120 \
  -- python apps/capswriter-client/scripts/verify-web.py
```

Run the ASR upload integration check:

```bash
python /home/df/.agents/skills/webapp-testing/scripts/with_server.py \
  --server "cd apps/capswriter-client && BROWSER=none npx expo start --web --port 8081 --host localhost" --port 8081 \
  --server "python apps/capswriter-client/scripts/mock-openai-compatible.py --port 8099" --port 8099 \
  --timeout 120 \
  -- python apps/capswriter-client/scripts/verify-asr-mock.py
```

Run the streaming provider integration check:

```bash
python /home/df/.agents/skills/webapp-testing/scripts/with_server.py \
  --server "cd apps/capswriter-client && BROWSER=none npx expo start --web --port 8081 --host localhost" --port 8081 \
  --server "python apps/capswriter-client/scripts/mock-openai-compatible.py --port 8099" --port 8099 \
  --timeout 120 \
  -- python apps/capswriter-client/scripts/verify-openai-mock.py
```

## Security

On Android and iOS, settings are stored with `expo-secure-store` when available. On web, settings are stored in browser `localStorage`; avoid saving production cloud API keys on shared machines.

The Android native config requests microphone access and audio setting control only. Background audio recording/playback services are disabled in the `expo-audio` config plugin.
