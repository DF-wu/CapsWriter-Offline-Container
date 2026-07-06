# CapsWriter No-GUI CLI

See [../../docs/cli-client.md](../../docs/cli-client.md) for complete usage, architecture, verification, and cleanup instructions.

Quick start:

```bash
python client/cli/capswriter_cli.py health --base-url http://127.0.0.1:6017
python client/cli/capswriter_cli.py transcribe sample.wav --format text
python client/cli/capswriter_cli.py speak "CapsWriter transcription completed."
python client/cli/scripts/build_zipapp.py
python client/cli/dist/capswriter-cli.pyz --help
python client/cli/scripts/verify.py
```
