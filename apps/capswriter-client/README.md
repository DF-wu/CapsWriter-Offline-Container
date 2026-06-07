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
- Generate speech with `POST /v1/audio/speech`
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
npm run android
```

Android preview APK with EAS:

```bash
npx eas build --platform android --profile preview
```

iOS is optional and uses the same app:

```bash
npm run ios
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
- Conversation: base URL, API key, API mode, model, system prompt, temperature, max output tokens, history, stream flag
- TTS: base URL, API key, model, voice, output format, speed, instructions

The conversation stream flag is stored as part of the provider profile. The current UI sends non-streaming requests for maximum provider compatibility across web and native fetch implementations.

## Security

On Android and iOS, settings are stored with `expo-secure-store` when available. On web, settings are stored in browser `localStorage`; avoid saving production cloud API keys on shared machines.
