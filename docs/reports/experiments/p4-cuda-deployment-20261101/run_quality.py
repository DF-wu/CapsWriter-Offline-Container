"""Greedy quality control; the existing HTTP worker currently uses temp=0.4.

Run only in a disposable container, one engine at a time. Baseline loads the
original native libraries and CPU decoder; candidate loads the tested CUDA
bundle. Both use the original encoder/decoder files and identical parameters.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf

from core.server.engines.qwen_asr_gguf.asr_engine import QwenASREngine
from core.server.engines.qwen_asr_gguf.inference.schema import ASREngineConfig

p = argparse.ArgumentParser()
p.add_argument('--backend', choices=['cpu', 'cuda'], required=True)
p.add_argument('--output', required=True)
p.add_argument('--run-id', required=True)
a = p.parse_args()
cfg = ASREngineConfig(
    model_dir='/app/models/Qwen3-ASR/Qwen3-ASR-1.7B',
    encoder_frontend_fn='qwen3_asr_encoder_frontend.onnx',
    encoder_backend_fn='qwen3_asr_encoder_backend.onnx', llm_fn='qwen3_asr_llm.gguf',
    onnx_provider='CUDA', llm_use_gpu=a.backend == 'cuda', n_ctx=2048,
    n_batch=4096, n_ubatch=512, n_threads=4, n_threads_batch=4,
    flash_attn=True, verbose=False, chunk_size=80,
)
adapter = QwenASREngine(cfg)
rows = []
for sample in json.loads(Path('/bench/samples.json').read_text()):
    if sample.get('phase') == 'latency':
        continue
    y, sr = sf.read(sample['path'], dtype='float32'); assert sr == 16000
    np.random.seed(1729)
    stream = adapter.create_stream(); stream.accept_waveform(sr, y)
    start = time.perf_counter()
    adapter.decode_stream(stream, language=sample.get('language'), temperature=0)
    text = stream.result.text
    assert text.strip(), sample['id']
    rows.append({'id': sample['id'], 'text': text, 'reference': sample['reference'],
                 'wall_s': time.perf_counter()-start, 'temperature': 0,
                 'backend': a.backend, 'run_id': a.run_id})
    Path(a.output).write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(json.dumps({'greedy_quality': sample['id'], 'backend': a.backend, 'ok': True}), flush=True)
adapter.cleanup()
