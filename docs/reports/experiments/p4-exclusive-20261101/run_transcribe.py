"""Warm, persistent C-API benchmark of the pinned transcribe.cpp build.

The public structs below mirror include/transcribe.h at 5700f127e38d; their
size is checked against the initializer before any inference is attempted.
This measures a complete replacement engine, not a drop-in GGUF conversion.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import soundfile as sf


class LoadParams(C.Structure):
    _fields_ = [('struct_size', C.c_uint64), ('backend', C.c_int),
                ('device', C.c_void_p), ('no_mmap', C.c_bool)]


class SessionParams(C.Structure):
    _fields_ = [('struct_size', C.c_uint64), ('n_threads', C.c_int),
                ('kv_type', C.c_int), ('n_ctx', C.c_int32)]


class RunParams(C.Structure):
    _fields_ = [('struct_size', C.c_uint64), ('task', C.c_int), ('timestamps', C.c_int),
                ('pnc', C.c_int), ('itn', C.c_int), ('diarize', C.c_int),
                ('language', C.c_char_p), ('target_language', C.c_char_p),
                ('keep_special_tags', C.c_bool), ('family', C.c_void_p),
                ('spec_k_drafts', C.c_int32), ('vocabulary', C.c_void_p),
                ('n_vocabulary', C.c_int32), ('prompt', C.c_char_p), ('prefix', C.c_char_p)]


class Timings(C.Structure):
    _fields_ = [('struct_size', C.c_uint64), ('load_ms', C.c_float), ('mel_ms', C.c_float),
                ('encode_ms', C.c_float), ('decode_ms', C.c_float)]


def gpu_snapshot():
    return subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu,clocks.current.graphics,memory.used,utilization.gpu,power.draw,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.hw_thermal_slowdown',
                           '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout.strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--config-id', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--samples', default='/bench/samples.json')
    p.add_argument('--phase', default='latency')
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--warmup', type=int, default=3)
    a = p.parse_args()
    lib = C.CDLL('/bench/transcribe-build/lib/libtranscribe.so')

    def bind(name, args, result):
        f = getattr(lib, name);f.argtypes = args;f.restype = result
        return f

    def initialized(name, cls):
        obj = cls();bind(name, [C.POINTER(cls)], None)(C.byref(obj))
        assert obj.struct_size == C.sizeof(cls), (name, obj.struct_size, C.sizeof(cls))
        return obj

    status_string = bind('transcribe_status_string', [C.c_int], C.c_char_p)

    def checked(status):
        if status != 0:raise RuntimeError(status_string(status).decode())

    checked(bind('transcribe_init_backends_default', [], C.c_int)())
    lp = initialized('transcribe_model_load_params_init', LoadParams)
    lp.backend = 5  # TRANSCRIBE_BACKEND_CUDA: require CUDA, never AUTO fallback.
    model = C.c_void_p();session = C.c_void_p()
    t = time.perf_counter()
    checked(bind('transcribe_model_load_file', [C.c_char_p, C.POINTER(LoadParams), C.POINTER(C.c_void_p)], C.c_int)(
        a.model.encode(), C.byref(lp), C.byref(model)))
    sp = initialized('transcribe_session_params_init', SessionParams)
    sp.n_threads = 4;sp.n_ctx = 2048
    checked(bind('transcribe_session_init', [C.c_void_p, C.POINTER(SessionParams), C.POINTER(C.c_void_p)], C.c_int)(
        model, C.byref(sp), C.byref(session)))
    load_s = time.perf_counter() - t
    backend = bind('transcribe_model_backend', [C.c_void_p], C.c_char_p)(model).decode()
    assert 'cuda' in backend.lower(), backend
    run = bind('transcribe_run', [C.c_void_p, C.POINTER(C.c_float), C.c_int, C.POINTER(RunParams)], C.c_int)
    text = bind('transcribe_full_text', [C.c_void_p], C.c_char_p)
    aborted = bind('transcribe_was_aborted', [C.c_void_p], C.c_bool)
    get_timings = bind('transcribe_get_timings', [C.c_void_p, C.POINTER(Timings)], C.c_int)
    reset_timings = bind('transcribe_reset_timings', [C.c_void_p], None)
    samples = [s for s in json.loads(Path(a.samples).read_text()) if s.get('phase', 'quality') == a.phase]
    assert samples
    loaded = {}
    for sample in samples:
        audio, rate = sf.read(sample['path'], dtype='float32')
        assert rate == 16000 and audio.ndim == 1
        loaded[sample['id']] = np.ascontiguousarray(audio)
    run_id = str(time.time_ns())
    metadata = {'type': 'metadata', 'run_id': run_id, 'config': a.config_id, 'load_s': load_s,
                'runtime': 'transcribe.cpp', 'runtime_commit': '5700f127e38dd53c4735ea23ef743b4b538eacd4',
                'version': bind('transcribe_version', [], C.c_char_p)().decode(),
                'backend': backend, 'ggml_devices': [backend], 'parameters': vars(a),
                'n_threads': 4, 'n_ctx': 2048, 'sampling': 'runtime greedy default',
                'library_path': '/bench/transcribe-build/lib/libtranscribe.so',
                'public_struct_bytes': {cls.__name__: C.sizeof(cls) for cls in [LoadParams, SessionParams, RunParams, Timings]}}
    try:
        with open(a.output, 'a', buffering=1) as stream:
            stream.write(json.dumps(metadata, ensure_ascii=False) + '\n')
            sequence = [(True, i, samples[i % len(samples)]) for i in range(a.warmup)] + [
                (False, rep, sample) for rep in range(a.repeats) for sample in samples]
            for warmup, repeat, sample in sequence:
                rp = initialized('transcribe_run_params_init', RunParams)
                rp.language = sample['language'].encode() if sample.get('language') else None
                rp.timestamps = 0;rp.spec_k_drafts = 0
                audio = loaded[sample['id']];reset_timings(session)
                before = gpu_snapshot();start = time.perf_counter()
                status = run(session, audio.ctypes.data_as(C.POINTER(C.c_float)), len(audio), C.byref(rp))
                wall_s = time.perf_counter() - start
                output = (text(session) or b'').decode('utf-8', errors='replace')
                timings = initialized('transcribe_timings_init', Timings)
                checked(get_timings(session, C.byref(timings)))
                is_aborted = bool(aborted(session))
                row = {'type': 'sample', 'run_id': run_id, 'config': a.config_id, 'phase': a.phase,
                       'id': sample['id'], 'repeat': repeat, 'warmup': warmup, 'wall_s': wall_s,
                       'audio_s': len(audio)/16000, 'text': output, 'reference': sample.get('reference'),
                       'status': 'ok' if status == 0 and output.strip() and not is_aborted else 'error',
                       'runtime_status': status, 'error': None if status == 0 else status_string(status).decode(),
                       'gpu_before': before, 'gpu_after': gpu_snapshot(), 'aborted': is_aborted,
                       'mel_s': timings.mel_ms/1000, 'encoder_s': timings.encode_ms/1000,
                       'decoder_s': timings.decode_ms/1000}
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                print(json.dumps({k: row[k] for k in ['config', 'id', 'warmup', 'wall_s', 'status']}), flush=True)
    finally:
        bind('transcribe_session_free', [C.c_void_p], None)(session)
        bind('transcribe_model_free', [C.c_void_p], None)(model)


if __name__ == '__main__':
    main()
