"""Build the alternative runtime independently of the b7798 ABI."""
import json
from pathlib import Path
import subprocess

state = json.loads(Path('/tmp/capswriter-p4-exclusive-state.json').read_text())
root = Path(state['scratch']);container = state['compiler_container']
source = root / 'transcribe-source'
commit = '5700f127e38dd53c4735ea23ef743b4b538eacd4'
if not source.exists():
    with open(root / 'logs/transcribe-fetch.log', 'a') as stream:
        for command in [['git', 'init', str(source)],
                        ['git', '-C', str(source), 'fetch', '--depth', '1',
                         'https://github.com/handy-computer/transcribe.cpp', commit],
                        ['git', '-C', str(source), 'checkout', '--detach', 'FETCH_HEAD']]:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == commit
flags = ['-DTRANSCRIBE_CUDA=ON', '-DCMAKE_CUDA_ARCHITECTURES=61', '-DCMAKE_BUILD_TYPE=Release',
         '-DTRANSCRIBE_METAL=OFF', '-DTRANSCRIBE_VULKAN=OFF', '-DTRANSCRIBE_BUILD_TESTS=OFF',
         '-DTRANSCRIBE_BUILD_EXAMPLES=OFF', '-DTRANSCRIBE_BUILD_TOOLS=OFF',
         '-DTRANSCRIBE_BUILD_SHARED=ON', '-DTRANSCRIBE_USE_SYSTEM_BLAS=OFF',
         '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=/bench/transcribe-build/lib',
         '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=/bench/transcribe-build/bin']
with open(root / 'logs/cmake-transcribe.log', 'a') as stream:
    subprocess.run(['docker', 'exec', container, 'cmake', '-S', '/bench/transcribe-source',
                    '-B', '/bench/transcribe-build', '-G', 'Ninja', *flags],
                   stdout=stream, stderr=subprocess.STDOUT, check=True)
with open(root / 'logs/build-transcribe.log', 'a') as stream:
    subprocess.run(['docker', 'exec', '-e', 'LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs', container,
                    'cmake', '--build', '/bench/transcribe-build', '--target', 'transcribe', '-j8'],
                   stdout=stream, stderr=subprocess.STDOUT, check=True)
with open(root / 'logs/cubins-transcribe.log', 'w') as stream:
    subprocess.run(['docker', 'exec', container, '/usr/local/cuda/bin/cuobjdump', '--list-elf',
                    '/bench/transcribe-build/lib/libggml-cuda.so'],stdout=stream,stderr=subprocess.STDOUT,check=True)
(root / 'results/transcribe-build.json').write_text(json.dumps({
    'commit': commit, 'flags': flags,
    'target': 'transcribe', 'compiler_only_stub_ld_library_path': True},indent=2)+'\n')
print('TRANSCRIBE BUILD COMPLETE',flush=True)
