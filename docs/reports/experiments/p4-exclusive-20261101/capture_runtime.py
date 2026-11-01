"""Run inside the idle test container after all measurements, before cleanup."""
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

root = Path('/bench')
expected_original = {
    'qwen3_asr_encoder_frontend.onnx': 'fc288ba9a0ca732251b33628c803cd5188d3dd77a2cd7336f0f0fd37969ed4cd',
    'qwen3_asr_encoder_backend.onnx': '4b0b1e34e8393f29222006aae0719283b0ff28b61f289b6ff8377b3ddbda27e3',
    'qwen3_asr_llm.gguf': '89e198470f41d487ab5b6b8a36a35400b3b1534451c4499236983050fe9193e5'}


def artifact(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


env = {'captured_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'python': platform.python_version(),
       'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'onnx', 'onnxruntime-gpu', 'soundfile']},
       'model_artifacts': [], 'libraries': [], 'source_files': [],
       'gpu_query': subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total',
                                           '--format=csv'], text=True)}
for name, expected in expected_original.items():
    item = artifact(Path('/models-original/Qwen3-ASR/Qwen3-ASR-1.7B') / name)
    assert item['sha256'] == expected, item
    env['model_artifacts'].append(dict(item, variant='original-read-only', matches_prior_experiment=True))
for manifest in ['downloads.json', 'extra-models.json']:
    for model in json.loads((root / 'results' / manifest).read_text()):
        path = root / 'downloads' / model.get('download_relative_path', model['filename'])
        item = artifact(path)
        assert item['sha256'] == model['sha256'] and item['bytes'] == model['bytes'], item
        env['model_artifacts'].append(dict(item, variant=manifest, url=model['url']))
for variant in ['q4-full', 'fp32-promoted']:
    for path in sorted((root / 'models' / variant).glob('*.onnx')):
        env['model_artifacts'].append(dict(artifact(path), variant=variant))
for directory in ['vulkan-bin', 'cuda-build/bin', 'transcribe-build/lib']:
    for path in sorted((root / directory).glob('*.so*')):
        if path.is_file() and not path.is_symlink():
            env['libraries'].append(artifact(path))
for path in [root / 'transcribe-source/include/transcribe.h',
             root / 'transcribe-source/ggml/src/ggml-cuda/mmq-config-pascal-dp4a.cuh',
             root / 'transcribe-source/ggml/src/ggml-cuda/mmq.cuh',
             root / 'llama-source/ggml/src/ggml-cuda/mmq.cuh',
             root / 'cuda-build/CMakeCache.txt', root / 'transcribe-build/CMakeCache.txt']:
    env['source_files'].append(artifact(path))
env['ffmpeg'] = subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0]
(root / 'results/environment.json').write_text(json.dumps(env, indent=2) + '\n')
print(json.dumps({'verified_original_models': len(expected_original),
                  'verified_artifacts': len(env['model_artifacts']),
                  'libraries': len(env['libraries']), 'packages': env['packages']}))
