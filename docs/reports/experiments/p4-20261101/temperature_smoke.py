"""Fixed-seed temperature=0.4 smoke check, after all serial matrix work."""
import json
from pathlib import Path
import subprocess

state = json.loads(Path('/tmp/capswriter-p4-experiment-state.json').read_text())
root = Path(state['scratch'])
samples = [dict(sample, phase='regression') for sample in json.loads((root / 'samples.json').read_text())
           if sample.get('phase') == 'latency']
(root / 'regression-samples.json').write_text(json.dumps(samples, ensure_ascii=False, indent=2) + '\n')
for config, backend in [('shipped-cpu-fallback', 'cpu-fallback'), ('vulkan-f16-q5', 'vulkan'),
                        ('cuda-f16-q5', 'cuda')]:
    link = root / 'src/core/server/engines/llama/bin'
    link.unlink()
    link.symlink_to('/bench/' + ('cuda-build/bin' if backend == 'cuda' else 'vulkan-bin'))
    args = ['docker', 'exec']
    if backend != 'cuda':
        args += ['-e', 'VK_ICD_FILENAMES=' + ('/bench/no-such-icd.json' if backend == 'cpu-fallback'
                                            else '/etc/vulkan/icd.d/nvidia_icd.json')]
    args += [state['container'], 'python', '/bench/run_engine.py',
             '--model-dir', '/models-original/Qwen3-ASR/Qwen3-ASR-1.7B',
             '--samples', '/bench/regression-samples.json',
             '--output', '/bench/results/temperature04.jsonl', '--config-id', config,
             '--phase', 'regression', '--repeats', '1', '--warmup', '1', '--temperature', '0.4']
    if backend == 'cpu-fallback':
        args += ['--allow-cpu-decoder']
    with open(root / 'logs' / (config + '-temperature04.log'), 'w') as log:
        subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
    print(config, 'temperature=0.4 completed', flush=True)
