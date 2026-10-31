"""Append model hashes and runtime versions before disposable cleanup."""
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

root = Path('/bench')
path = root / 'results/environment.json'
env = json.loads(path.read_text())
env['captured_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
env['python'] = platform.python_version()
env['packages'] = {}
for package in ['numpy', 'onnx', 'onnxruntime-gpu', 'soundfile']:
    env['packages'][package] = importlib.metadata.version(package)
env['gpu_query'] = subprocess.run(
    ['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total,temperature.gpu,clocks.current.graphics',
     '--format=csv'], check=True, capture_output=True, text=True).stdout
env['model_artifacts'] = []
original = Path('/models-original/Qwen3-ASR/Qwen3-ASR-1.7B')
expected = {
    'qwen3_asr_encoder_frontend.onnx': 'fc288ba9a0ca732251b33628c803cd5188d3dd77a2cd7336f0f0fd37969ed4cd',
    'qwen3_asr_encoder_backend.onnx': '4b0b1e34e8393f29222006aae0719283b0ff28b61f289b6ff8377b3ddbda27e3',
    'qwen3_asr_llm.gguf': '89e198470f41d487ab5b6b8a36a35400b3b1534451c4499236983050fe9193e5'}


def digest(file):
    with file.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else streamed_digest(stream)


def streamed_digest(stream):
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        h.update(chunk)
    return h.hexdigest()


for label, directory in [('original-read-only', original), ('q4-full', root / 'models/q4-full')]:
    for filename in expected:
        file = directory / filename
        sha = digest(file)
        if label == 'original-read-only':
            assert sha == expected[filename], (file, sha)
        env['model_artifacts'].append({'variant': label, 'filename': filename,
                                       'bytes': file.stat().st_size, 'sha256': sha})
env['fp32_promotion_reproduction'] = {}
for part in ['frontend', 'backend']:
    reproduced = root / ('promoted-reproduced-' + part + '.onnx')
    used = root / 'models/fp32-promoted' / ('qwen3_asr_encoder_' + part + '.onnx')
    sha = digest(reproduced)
    assert sha == digest(used), ('promotion reproduction mismatch', part)
    env['fp32_promotion_reproduction'][part] = {'byte_identical': True, 'sha256': sha}
path.write_text(json.dumps(env, indent=2) + '\n')
print(json.dumps({key: env[key] for key in ['packages', 'model_artifacts', 'fp32_promotion_reproduction']}, indent=2))
