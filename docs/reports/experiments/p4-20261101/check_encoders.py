"""Separate untimed embedding / ORT provider diagnostic, run after matrix.

This uses the same encoder options and weights as the benchmark. Profiling
overhead is deliberately excluded from the latency experiment.
"""
import collections
import gc
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf

from core.server.engines.qwen_asr_gguf.inference.encoder import QwenAudioEncoder

ROOT = Path('/bench')
ORIGINAL = Path('/models-original/Qwen3-ASR/Qwen3-ASR-1.7B')
VARIANTS = {'f16': ORIGINAL, 'f32-promoted': ROOT / 'models/fp32-promoted',
            'w4': ROOT / 'models/q4-full'}
SESSION = ort.InferenceSession
active_variant = ''


def profiled_session(*args, **kwargs):
    options = kwargs['sess_options']
    options.enable_profiling = True
    options.profile_file_prefix = str(ROOT / 'profiles' / active_variant)
    return SESSION(*args, **kwargs)


def summarize_profile(path):
    totals = collections.defaultdict(lambda: {'calls': 0, 'duration_us': 0})
    type_examples = {}
    for event in json.loads(Path(path).read_text()):
        args = event.get('args', {})
        provider = args.get('provider')
        if not provider or not event.get('name', '').endswith('_kernel_time'):
            continue
        key = provider + '/' + args.get('op_name', '?')
        totals[key]['calls'] += 1
        totals[key]['duration_us'] += event.get('dur', 0)
        if key not in type_examples:
            type_examples[key] = {k: args[k] for k in ['input_type_shape', 'output_type_shape'] if k in args}
    return {'by_provider_op': dict(totals), 'type_examples': type_examples}


def graph_info(path):
    model = onnx.load(path)
    try:
        onnx.checker.check_model(model)
        checked = 'passed'
    except onnx.checker.ValidationError as error:
        checked = str(error)
    return {'checker': checked,
            'input_types': {v.name: onnx.helper.printable_type(v.type) for v in model.graph.input},
            'initializer_types': dict(collections.Counter(onnx.TensorProto.DataType.Name(t.data_type) for t in model.graph.initializer)),
            'operators': dict(collections.Counter(n.op_type for n in model.graph.node))}


def main():
    global active_variant
    (ROOT / 'profiles').mkdir(exist_ok=True)
    ort.InferenceSession = profiled_session
    samples = [s for s in json.loads((ROOT / 'samples.json').read_text()) if s.get('phase') == 'latency']
    baseline, report = {}, {}
    for name, directory in VARIANTS.items():
        active_variant = name
        frontend, backend = [directory / ('qwen3_asr_encoder_' + part + '.onnx') for part in ['frontend', 'backend']]
        info = {'graphs': {'frontend': graph_info(frontend), 'backend': graph_info(backend)}, 'samples': {}}
        encoder = QwenAudioEncoder(str(frontend), str(backend), onnx_provider='CUDA', verbose=False)
        for sample in samples:
            audio, rate = sf.read(sample['path'], dtype='float32')
            assert rate == 16000
            embedding = encoder.encode(audio)[0].astype(np.float64)
            assert np.isfinite(embedding).all()
            if name == 'f16':
                baseline[sample['id']] = embedding.copy()
            reference = baseline[sample['id']]
            assert reference.shape == embedding.shape
            row_cosines = np.sum(reference * embedding, axis=1) / (
                np.linalg.norm(reference, axis=1) * np.linalg.norm(embedding, axis=1))
            info['samples'][sample['id']] = {
                'shape': list(embedding.shape), 'finite': True,
                'cosine': float(np.vdot(reference, embedding) / (np.linalg.norm(reference) * np.linalg.norm(embedding))),
                'min_frame_cosine': float(row_cosines.min()),
                'relative_l2': float(np.linalg.norm(reference - embedding) / np.linalg.norm(reference)),
                'max_abs_difference': float(np.abs(reference - embedding).max())}
        info['profiles'] = {part: summarize_profile(session.end_profiling()) for part, session in
                            [('frontend', encoder.sess_fe), ('backend', encoder.sess_be)]}
        report[name] = info
        del encoder
        gc.collect()
    (ROOT / 'results/encoder-diagnostics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({name: {'samples': info['samples'], 'backend_ops': info['profiles']['backend']['by_provider_op']} for name, info in report.items()}, indent=2))


if __name__ == '__main__':
    main()
