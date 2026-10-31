"""Validate saved matrix evidence and derive quality / thermal audit.

Usage: python audit_results.py RESULTS_DIRECTORY
Requires only Python's standard library. It writes audit.json.
"""
import collections
import json
import math
from pathlib import Path
import re
import statistics
import sys
import unicodedata


def normalized(text):
    text = unicodedata.normalize('NFKC', text).lower()
    text = re.sub(r'\bmr\.', 'mister', text)
    return text


def characters(text):
    return ''.join(c for c in normalized(text)
                   if not unicodedata.category(c).startswith(('P', 'Z')) and not c.isspace())


def words(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", normalized(text))


def distance(a, b):
    row = list(range(len(b) + 1))
    for i, c in enumerate(a, 1):
        next_row = [i]
        for j, d in enumerate(b, 1):
            next_row.append(min(row[j] + 1, next_row[-1] + 1, row[j - 1] + (c != d)))
        row = next_row
    return row[-1]


def main():
    root = Path(sys.argv[1])
    configs = ['shipped-cpu-fallback', 'vulkan-f16-q5', 'cuda-f16-q5',
               'cuda-f32-q5', 'cuda-f16-q4', 'cuda-int4-q4']
    samples = [json.loads(line) for config in configs
               for line in (root / (config + '.jsonl')).read_text().splitlines()]
    measured = [r for r in samples if r['type'] == 'sample' and not r['warmup']
                and r['phase'] in ('latency', 'quality')]
    quality_baseline = {r['id']: r['text'] for r in measured
                        if r['config'] == 'shipped-cpu-fallback' and r['phase'] == 'quality'}
    assert len(quality_baseline) == 5
    result = {}
    for config in configs:
        latency = [r for r in measured if r['config'] == config and r['phase'] == 'latency']
        quality = [r for r in measured if r['config'] == config and r['phase'] == 'quality']
        assert len(latency) == 30 and len(quality) == 5, (config, len(latency), len(quality))
        assert sorted(collections.Counter(r['id'] for r in latency).values()) == [10, 10, 10]
        assert len({r['id'] for r in quality}) == 5
        for r in latency + quality:
            assert r['status'] == 'ok' and r['text'].strip() and not r['aborted'], r
            assert math.isfinite(r['wall_s']) and r['wall_s'] > 0
        snapshots = [r[k].split(', ') for r in latency for k in ['gpu_before', 'gpu_after']]
        assert all(len(s) == 7 for s in snapshots)
        numeric = lambda index: [float(s[index]) for s in snapshots]
        by_id = {}
        for r in quality:
            ref, hyp = characters(r['reference']), characters(r['text'])
            by_id[r['id']] = {
                'text': r['text'], 'reference': r['reference'],
                'raw_matches_cpu': r['text'] == quality_baseline[r['id']],
                'normalized_matches_cpu': hyp == characters(quality_baseline[r['id']]),
                'character_edits': distance(ref, hyp), 'reference_characters': len(ref),
                'cer': distance(ref, hyp) / len(ref)}
            if r['id'].startswith('libri-'):
                ref_words, hyp_words = words(r['reference']), words(r['text'])
                by_id[r['id']].update(word_edits=distance(ref_words, hyp_words),
                                     reference_words=len(ref_words),
                                     wer=distance(ref_words, hyp_words) / len(ref_words))
        blocks = {}
        # The original baseline was captured before run_id was added. It is
        # one documented process, rather than an interleaved GPU block.
        for run_id in sorted({r.get('run_id', 'baseline-initial-process') for r in latency}):
            rs = [r for r in latency if r.get('run_id', 'baseline-initial-process') == run_id]
            blocks[run_id] = {
                'samples': len(rs),
                'median_wall_s': {sid: statistics.median(r['wall_s'] for r in rs if r['id'] == sid)
                                  for sid in sorted({r['id'] for r in rs})}}
        result[config] = {'latency_n': len(latency), 'quality_n': len(quality),
                          'retry_calls': sum(max(0, r['decode_attempts'] - 1) for r in latency + quality),
                          'quality': by_id, 'latency_blocks': blocks,
                          'snapshots': {'n': len(snapshots),
                                        'temperature_c': [min(numeric(0)), max(numeric(0))],
                                        'graphics_mhz': [min(numeric(1)), max(numeric(1))],
                                        'gpu_memory_mib': [min(numeric(2)), max(numeric(2))],
                                        'sw_thermal_active': sum(s[5] == 'Active' for s in snapshots),
                                        'hw_thermal_active': sum(s[6] == 'Active' for s in snapshots)}}
    schedule = json.loads((root / 'schedule.json').read_text())
    assert len(schedule) == 15 and all(s['exit'] == 0 for s in schedule)
    regression_path = root / 'temperature04.jsonl'
    if regression_path.exists():
        regression = [json.loads(line) for line in regression_path.read_text().splitlines()]
        calls = [r for r in regression if r['type'] == 'sample' and not r['warmup']]
        assert len(calls) == 9 and all(r['status'] == 'ok' for r in calls)
        result['temperature04_smoke'] = {'n': len(calls), 'failures': 0,
                                       'retry_calls': sum(max(0, r['decode_attempts'] - 1) for r in calls),
                                       'note': 'One fixed seed per clip/config; not a performance distribution.'}
    (root / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('Validated 180 latency and 30 quality calls, 15 successful GPU blocks.')
    print(json.dumps({k: v.get('snapshots', v) for k, v in result.items()}, indent=2))


if __name__ == '__main__':
    main()
