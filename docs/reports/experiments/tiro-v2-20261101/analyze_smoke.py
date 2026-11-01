"""Offline analysis with per-sample quality gates, no benchmark side effects."""
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'p4-exclusive-20261101'))
from analyze_exclusive import edits, norm, words


def stats(values):
    values=sorted(values)
    return {'n':len(values),'p50':statistics.median(values),'min':values[0],'max':values[-1]}


def main():
    root=Path(sys.argv[1]);samples=json.loads((root.parent/'samples.json').read_text())
    configs=json.loads((root/'configs.json').read_text());summary={}
    acceptance=json.loads((root/'acceptance.json').read_text())
    assert acceptance['normal_completion'] and len(acceptance['blocks'])==8
    accepted={(x['config'],x['phase'],x['run_id']) for x in acceptance['blocks'] if x['accepted']}
    assert len(accepted)==8
    quality_samples=[s for s in samples if s.get('phase','quality')=='quality']
    for config in configs:
        cid=config['id'];rows=[json.loads(l) for l in (root/(cid+'.jsonl')).read_text().splitlines()]
        metas=[x for x in rows if x['type']=='metadata']
        calls=[x for x in rows if x['type']=='sample' and not x['warmup']]
        assert all((cid,x['phase'],x['run_id']) in accepted for x in calls)
        q=[x for x in calls if x['phase']=='quality'];lat=[x for x in calls if x['phase']=='latency']
        assert len(q)==14 and len(lat)==9, (cid,len(q),len(lat))
        item={'id':cid,'quality':{},'latency':{},'groups':{},'metadata':metas,
              'measured_calls':len(calls),'failures':sum(x['status']!='ok' for x in calls),
              'aborts':sum(bool(x.get('aborted')) for x in calls),
              'extra_decode_attempts':sum(max(0,x.get('decode_attempts',1)-1) for x in calls)}
        for sample in quality_samples:
            x=next(x for x in q if x['id']==sample['id'])
            ref=norm(sample['reference']);n=edits(ref,norm(x['text']))
            row={'id':x['id'],'text':x['text'],'reference':sample['reference'],
                 'condition':sample.get('condition','Mandarin' if x['id']=='noise2' else 'multilingual'),
                 'cer_edits':n,'reference_characters':len(ref),'cer':n/len(ref),
                 'wall_s':x['wall_s'],'status':x['status']}
            if x['id'].startswith('libri-'):
                wref=words(sample['reference']);n=edits(wref,words(x['text']))
                row.update(wer_edits=n,reference_words=len(wref),wer=n/len(wref))
            item['quality'][x['id']]=row
        for condition in sorted({x['condition'] for x in item['quality'].values()}):
            group=[x for x in item['quality'].values() if x['condition']==condition]
            g={'n':len(group),'cer':sum(x['cer_edits'] for x in group)/sum(x['reference_characters'] for x in group)}
            if all('wer' in x for x in group):
                g['word_errors']=sum(x['wer_edits'] for x in group)
                g['reference_words']=sum(x['reference_words'] for x in group)
                g['wer']=g['word_errors']/g['reference_words']
            item['groups'][condition]=g
        for sid in sorted({x['id'] for x in lat}):
            xs=[x for x in lat if x['id']==sid]
            good=[x['wall_s'] for x in xs if x['status']=='ok' and math.isfinite(x['wall_s'])]
            item['latency'][sid]={'wall_s':stats(good) if good else None,
                                  'texts':list(dict.fromkeys(x['text'] for x in xs)),
                                  'n_generate':list(dict.fromkeys(x.get('n_generate') for x in xs))}
        snapshots=[x[k].split(', ') for x in calls for k in ['gpu_before','gpu_after']]
        item['thermal']={'snapshots':len(snapshots),
            'temperature_c':[min(int(x[0]) for x in snapshots),max(int(x[0]) for x in snapshots)],
            'graphics_mhz':[min(int(x[1]) for x in snapshots),max(int(x[1]) for x in snapshots)],
            'gpu_mib':[min(int(x[2]) for x in snapshots),max(int(x[2]) for x in snapshots)],
            'sw_thermal_active':sum(x[5]=='Active' for x in snapshots),
            'hw_thermal_active':sum(x[6]=='Active' for x in snapshots)}
        summary[cid]=item
    for cid,item in summary.items():
        item['versus']={}
        for baseline in ['original-q5','tiro-f16']:
            worse=[];better=[]
            for sid,row in item['quality'].items():
                base=summary[baseline]['quality'][sid]
                metric='wer' if 'wer' in row else 'cer'
                if row[metric]>base[metric]+1e-12:worse.append(sid)
                if row[metric]<base[metric]-1e-12:better.append(sid)
            item['versus'][baseline]={'worse_samples':worse,'better_samples':better,
                'passes_small_sample_quality_gate':not worse and item['failures']==0}
    (root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({cid:{'groups':v['groups'],'quality_gate':v['versus']['original-q5'],
                          'latency':{k:x['wall_s'] for k,x in v['latency'].items()}}
                      for cid,v in summary.items()},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
