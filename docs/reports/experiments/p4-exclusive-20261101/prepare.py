"""Prepare disposable resources before stopping the Portainer-owned service."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

REPO = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
STATE = Path('/tmp/capswriter-p4-exclusive-state.json')
ORIGINAL = '/models-original/Qwen3-ASR/Qwen3-ASR-1.7B'


def command(argv, log=None, **kwargs):
    if log:
        with open(log, 'a') as stream:
            return subprocess.run(argv, check=True, stdout=stream, stderr=subprocess.STDOUT, **kwargs)
    return subprocess.run(argv, check=True, capture_output=True, text=True, **kwargs)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprints(info):
    sections = {'Config': info['Config'], 'HostConfig': info['HostConfig']}
    network_keys = ['NetworkID', 'Aliases', 'Links', 'IPAMConfig', 'DriverOpts', 'GwPriority']
    sections['NetworkConfiguration'] = {
        name: {k: network.get(k) for k in network_keys}
        for name, network in info['NetworkSettings']['Networks'].items()}
    return {name: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            for name, value in sections.items()}


def main():
    resume = '--resume' in sys.argv
    if resume:
        state = json.loads(STATE.read_text()); root = Path(state['scratch'])
        assert state['label'] == 'p4-exclusive-20261101' and root.name.startswith('capswriter-p4-exclusive-')
    else:
        assert not STATE.exists(), 'An experiment state already exists; inspect it first.'
        root = Path(tempfile.mkdtemp(prefix='capswriter-p4-exclusive-'))
        for name in ['src', 'audio', 'logs', 'results', 'models', 'downloads', 'vulkan-bin', 'build-source']:
            (root / name).mkdir()
        production = json.loads(command(['docker', 'inspect', 'capswriter-server']).stdout)[0]
        state = {'scratch': str(root), 'container': 'capswriter-p4-exclusive-test-20261101',
             'compiler_container': 'capswriter-p4-exclusive-compiler-20261101',
             'label': 'p4-exclusive-20261101', 'base_image': production['Image'],
             'production_id': production['Id'], 'production_name': 'capswriter-server',
             'model_volume': 'capswriter_capswriter-server-models', 'created_images': [],
             'compiler_image': 'nvidia/cuda:11.8.0-devel-ubuntu22.04',
             'config_fingerprints_before': fingerprints(production),
             'production_before': {'id': production['Id'], 'image': production['Image'],
                                   'started_at': production['State']['StartedAt'],
                                   'health': production['State']['Health']['Status']}}
        assert production['State']['Running'] and production['State']['Health']['Status'] == 'healthy'
        STATE.write_text(json.dumps(state, indent=2) + '\n');STATE.chmod(0o600)
        (root / 'results/production-before.json').write_text(json.dumps({k: state[k] for k in ['production_before', 'config_fingerprints_before']}, indent=2))
    shutil.copyfile(HERE / 'run_engine.py', root / 'run_engine.py')
    for name in ['promote_encoder.py', 'analyze.py']:
        shutil.copyfile(REPO / 'docs/reports/experiments/p4-20261101' / name, root / name)
    if not resume:
        with (root / 'source.tar').open('wb') as stream:
            subprocess.run(['git', 'archive', 'd3b334e95c4d958f6d7812cb94c14a33fff427d3'], cwd=REPO, stdout=stream, check=True)
        with tarfile.open(root / 'source.tar') as archive:
            archive.extractall(root / 'src', filter='data')
        command(['docker', 'cp', 'capswriter-server:/app/core/server/engines/llama/bin/.', str(root / 'vulkan-bin')])
    link = root / 'src/core/server/engines/llama/bin'
    if link.is_symlink():link.unlink()
    elif link.exists():shutil.rmtree(link)
    link.symlink_to('/bench/vulkan-bin')
    def download(url, filename, expected=None):
        target = root / 'downloads' / filename
        if target.exists() and expected and digest(target) == expected:
            return {'filename': filename, 'url': url, 'bytes': target.stat().st_size, 'sha256': expected}
        request = urllib.request.Request(url, headers={'User-Agent': 'CapsWriter-P4-Research/1.0'})
        with urllib.request.urlopen(request, timeout=120) as response, target.open('wb') as stream:
            shutil.copyfileobj(response, stream, 1024 * 1024)
        sha = digest(target)
        if expected:assert sha == expected, (filename, sha)
        print('DOWNLOADED', filename, target.stat().st_size, sha, flush=True)
        return {'filename': filename, 'url': url, 'bytes': target.stat().st_size, 'sha256': sha}
    tasks = [
        ('https://api.github.com/repos/ggml-org/llama.cpp/tarball/c301172f660a1fe0b42023da990bf7385d69adb4', 'llama-b7798.tar.gz', '4420071ba535356cb62f8cd65edb924dee321942d45b7e5e402df1f2906e1031'),
        ('https://github.com/HaujetZhao/CapsWriter-Offline/releases/download/models/Qwen3-ASR-1.7B-q4_k.zip', 'q4.zip', '9b3d2a66a4a26a0404c32085ec838b7c482495a7827919a5aa674de617c2757b'),
        ('https://huggingface.co/csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/resolve/68818b2313fe77bd06f6a7c5068ff3ef59d02b8a/test_wavs/noise2.wav', 'noise2.wav', '33f85268f7fbad6b3152b9ab051edab1a85082fde66bccff61d7f5ef7b437e58'),
        ('https://huggingface.co/csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/resolve/68818b2313fe77bd06f6a7c5068ff3ef59d02b8a/test_wavs/codeswitch.wav', 'codeswitch.wav', None)]
    manifest = json.loads((REPO / 'docs/reports/tesla-p4-asr-model-artifacts-20261101.json').read_text())
    handy = next(r for r in manifest['huggingface'] if r['repo'] == 'handy-computer/Qwen3-ASR-1.7B-gguf')
    for model in handy['files']:
        if any(q in model['name'] for q in ['Q4_K_M', 'Q5_K_M', 'Q6_K', 'Q8_0']):
            tasks.append((model['url'], model['name'], model['publisher_lfs_sha256']))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download, *args) for args in tasks]
        if resume:
            info = json.loads(command(['docker', 'inspect', state['container']]).stdout)[0]
            assert info['Config']['Labels']['capswriter.experiment'] == state['label']
            command(['docker', 'rm', '-f', state['container']])
        command(['docker', 'run', '-d', '--name', state['container'], '--label', 'capswriter.experiment='+state['label'],
                 '--user', '0',
                 '--gpus', 'all', '--cpus', '6', '--memory', '16g', '--pids-limit', '512',
                 '-e', 'NVIDIA_DRIVER_CAPABILITIES=all', '-e', 'OMP_NUM_THREADS=4', '-e', 'OPENBLAS_NUM_THREADS=4',
                 '-e', 'PYTHONPATH=/bench/src', '-v', str(root)+':/bench', '-v', state['model_volume']+':/models-original:ro',
                 '-w', '/bench/src', '--entrypoint', '/usr/bin/tail', state['base_image'], '-f', '/dev/null'])
        command(['docker', 'exec', state['container'], 'apt-get', 'update'], root / 'logs/runtime-apt.log')
        command(['docker', 'exec', '-e', 'DEBIAN_FRONTEND=noninteractive', state['container'], 'apt-get', 'install', '-y', '--no-install-recommends', 'libegl1'], root / 'logs/runtime-apt.log')
        command(['docker', 'exec', state['container'], 'python', '-m', 'pip', 'install', '--no-cache-dir', 'onnx==1.16.2'], root / 'logs/runtime-pip.log')
        existing = subprocess.run(['docker', 'image', 'inspect', state['compiler_image']], capture_output=True)
        if existing.returncode:
            state['created_images'].append(state['compiler_image']);STATE.write_text(json.dumps(state, indent=2))
            command(['docker', 'pull', state['compiler_image']], root / 'logs/compiler-pull.log')
        compiler_exists = subprocess.run(['docker', 'inspect', state['compiler_container']], capture_output=True)
        if compiler_exists.returncode:
            command(['docker', 'run', '-d', '--name', state['compiler_container'], '--label', 'capswriter.experiment='+state['label'],
                     '--cpus', '8', '--memory', '12g', '-v', str(root)+':/bench', '--entrypoint', 'tail',
                     state['compiler_image'], '-f', '/dev/null'])
        else:
            info = json.loads(compiler_exists.stdout)[0]
            assert info['Config']['Labels']['capswriter.experiment'] == state['label']
        command(['docker', 'exec', state['compiler_container'], 'apt-get', 'update'], root / 'logs/compiler-apt.log')
        command(['docker', 'exec', '-e', 'DEBIAN_FRONTEND=noninteractive', state['compiler_container'], 'apt-get', 'install', '-y', '--no-install-recommends', 'cmake', 'ninja-build', 'g++', 'python3'], root / 'logs/compiler-apt.log')
        downloads = [future.result() for future in futures]
    (root / 'results/downloads.json').write_text(json.dumps(downloads, indent=2) + '\n')
    with tarfile.open(root / 'downloads/llama-b7798.tar.gz') as archive:
        archive.extractall(root / 'build-source', filter='data')
    source = next((root / 'build-source').iterdir())
    (root / 'llama-source').unlink(missing_ok=True)
    (root / 'llama-source').symlink_to(source.relative_to(root))
    command(['docker', 'exec', state['compiler_container'], 'cmake', '-S', '/bench/llama-source', '-B', '/bench/cuda-build', '-G', 'Ninja',
             '-DGGML_CUDA=ON', '-DCMAKE_CUDA_ARCHITECTURES=61', '-DCMAKE_BUILD_TYPE=Release', '-DLLAMA_CURL=OFF',
             '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_BUILD_EXAMPLES=OFF', '-DLLAMA_BUILD_TOOLS=ON'], root / 'logs/cmake-b7798.log')
    command(['docker', 'exec', state['compiler_container'], 'ln', '-sf', 'libcuda.so', '/usr/local/cuda/lib64/stubs/libcuda.so.1'])
    command(['docker', 'exec', '-e', 'LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs', state['compiler_container'], 'cmake', '--build', '/bench/cuda-build', '--target', 'llama', 'llama-quantize', '-j8'], root / 'logs/build-b7798.log')
    command(['docker', 'exec', state['compiler_container'], '/usr/local/cuda/bin/cuobjdump', '--list-elf', '/bench/cuda-build/bin/libggml-cuda.so.0.9.5'], root / 'logs/cubins-b7798.log')
    (root / 'models/q4-full').mkdir(exist_ok=True)
    with zipfile.ZipFile(root / 'downloads/q4.zip') as archive:
        for item in archive.infolist():
            if Path(item.filename).name in ['qwen3_asr_encoder_frontend.onnx', 'qwen3_asr_encoder_backend.onnx', 'qwen3_asr_llm.gguf']:
                with archive.open(item) as inp, (root / 'models/q4-full' / Path(item.filename).name).open('wb') as out:
                    shutil.copyfileobj(inp, out)
    for variant in ['q4', 'fp32-promoted']:(root / 'models' / variant).mkdir(exist_ok=True)
    for part in ['frontend', 'backend']:
        name='qwen3_asr_encoder_'+part+'.onnx'
        (root / 'models/q4' / name).symlink_to(ORIGINAL+'/'+name)
        command(['docker', 'exec', state['container'], 'python', '/bench/promote_encoder.py', ORIGINAL+'/'+name, '/bench/models/fp32-promoted/'+name], root / 'logs/promote.log')
    (root / 'models/q4/qwen3_asr_llm.gguf').symlink_to('/bench/models/q4-full/qwen3_asr_llm.gguf')
    (root / 'models/fp32-promoted/qwen3_asr_llm.gguf').symlink_to(ORIGINAL+'/qwen3_asr_llm.gguf')
    shutil.copyfile(root / 'downloads/noise2.wav', root / 'audio/noise2.wav')
    # Pin FFmpeg to the runtime's 4.4.2. Host 9.0.2 writes a different WAV
    # fingerprint; the existing experiment's manifest pins whole-file bytes.
    command(['docker', 'exec', state['container'], 'ffmpeg', '-y', '-loglevel', 'error',
             '-i', '/bench/downloads/codeswitch.wav', '-ar', '16000', '-ac', '1', '/bench/audio/codeswitch-16k.wav'])
    samples = json.loads((REPO / 'docs/reports/experiments/p4-20261101/samples.json').read_text())
    samples = [s for s in samples if s.get('phase') == 'latency' or s['id'] in ['noise2', 'codeswitch']]
    (root / 'samples.json').write_text(json.dumps(samples, ensure_ascii=False, indent=2) + '\n')
    code = "import soundfile as sf;from pathlib import Path;y,sr=sf.read('/bench/audio/noise2.wav',dtype='float32');assert sr==16000;[(sf.write('/bench/audio/mandarin-'+str(n)+'s.wav',y[:16000*n],16000)) for n in [2,5,10]]"
    command(['docker', 'exec', state['container'], 'python', '-c', code])
    for sample in samples:assert digest(root / sample['path'].removeprefix('/bench/')) == sample['audio_sha256'], sample['id']
    state['base_preparation_complete'] = True
    STATE.write_text(json.dumps(state, indent=2) + '\n')
    print('BASE PREPARATION COMPLETE', root, flush=True)


if __name__ == '__main__':
    main()
