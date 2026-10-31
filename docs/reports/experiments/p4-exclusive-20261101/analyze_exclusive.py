"""Recompute latency, quality, thermal evidence and baseline improvements."""
import collections
import json
import math
from pathlib import Path
import re
import statistics
import sys
import unicodedata


def norm(text):
    text=re.sub(r'\bmr\.', 'mister',unicodedata.normalize('NFKC',text).lower())
    return ''.join(c for c in text if not unicodedata.category(c).startswith(('P','Z')) and not c.isspace())


def words(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?",re.sub(r'\bmr\.', 'mister',unicodedata.normalize('NFKC',text).lower()))


def edits(a,b):
    row=list(range(len(b)+1))
    for i,c in enumerate(a,1):
        nxt=[i]
        for j,d in enumerate(b,1):nxt.append(min(row[j]+1,nxt[-1]+1,row[j-1]+(c!=d)))
        row=nxt
    return row[-1]


def stats(values):
    values=sorted(values)
    return {'n':len(values),'p50':statistics.median(values),'p95':values[math.ceil(.95*len(values))-1],
            'min':values[0],'max':values[-1]}


def thermal(rows):
    snapshots=[r[k].split(', ') for r in rows for k in ['gpu_before','gpu_after']]
    assert all(len(s)==7 for s in snapshots)
    return {'snapshots':len(snapshots),'temperature_c':[min(int(s[0]) for s in snapshots),max(int(s[0]) for s in snapshots)],
            'graphics_mhz':[min(int(s[1]) for s in snapshots),max(int(s[1]) for s in snapshots)],
            'total_gpu_mib':[min(int(s[2]) for s in snapshots),max(int(s[2]) for s in snapshots)],
            'sw_thermal_active':sum(s[5]=='Active' for s in snapshots),
            'hw_thermal_active':sum(s[6]=='Active' for s in snapshots)}


def main():
    root=Path(sys.argv[1]);configs=json.loads((root/'configs.json').read_text());summary={}
    qualities={}
    for config in configs:
        path=root/(config['id']+'.jsonl')
        if not path.exists():continue
        source=[json.loads(line) for line in path.read_text().splitlines()]
        rows=[r for r in source if r['type']=='sample' and not r['warmup']]
        result={'config':config,'warmup_n':sum(r.get('type')=='sample' and r['warmup'] for r in source)}
        for phase in ['latency','quality']:
            rs=[r for r in rows if r['phase']==phase]
            if not rs:continue
            out={'n':len(rs),'failures':sum(r['status']!='ok' for r in rs),'by_sample':{},'thermal':thermal(rs)}
            for sid in sorted({r['id'] for r in rs}):
                sr=[r for r in rs if r['id']==sid];good=[r for r in sr if r['status']=='ok' and math.isfinite(r['wall_s'])]
                by={'n':len(sr),'ok':len(good),'outputs':dict(collections.Counter(r['text'] for r in sr)),
                    'thermal':thermal(sr),'aborts':sum(r['aborted'] for r in sr)}
                for key in ['wall_s','encoder_s','mel_s','decoder_s','frontend_s','backend_s','prefill_s','generate_s','n_generate']:
                    values=[r[key] for r in good if key in r]
                    if values:by[key]=stats(values)
                attempts=[r['decode_attempts'] for r in sr if 'decode_attempts' in r]
                by['extra_decode_attempts_observed']=sum(max(0,n-1) for n in attempts) if attempts else None
                clean=[r['wall_s'] for r in good if all(', Active' not in r[k] for k in ['gpu_before','gpu_after'])]
                if clean:by['unthrottled_endpoint_wall_s']=stats(clean)
                if sr[0].get('reference'):
                    reference=sr[0]['reference'];charref=norm(reference)
                    by['reference']=reference;by['reference_characters']=len(charref)
                    by['cer']=[edits(charref,norm(r['text']))/len(charref) for r in sr]
                    if sid.startswith('libri-'):
                        by['reference_words']=len(words(reference));by['wer']=[edits(words(reference),words(r['text']))/len(words(reference)) for r in sr]
                    qualities[(config['id'],sid)]=sr[0]['text']
                out['by_sample'][sid]=by
            result[phase]=out
        summary[config['id']]=result
    for cid,result in summary.items():
        for sid,values in result.get('latency',{}).get('by_sample',{}).items():
            if 'wall_s' not in values:continue
            for baseline in ['cpu-fallback','cuda-q5']:
                source=summary.get(baseline,{}).get('latency',{}).get('by_sample',{}).get(sid,{}).get('wall_s')
                if source:
                    values['versus_'+baseline]={key:{'speedup':source[key]/values['wall_s'][key],
                        'latency_reduction_pct':100*(source[key]-values['wall_s'][key])/source[key]} for key in ['p50','p95']}
        for sid,values in result.get('quality',{}).get('by_sample',{}).items():
            baseline=qualities.get(('cuda-q5',sid))
            if baseline is not None:
                hypothesis=qualities[(cid,sid)];values['raw_matches_cuda_q5']=hypothesis==baseline
                values['normalized_matches_cuda_q5']=norm(hypothesis)==norm(baseline)
    (root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    for cid,result in summary.items():
        latency=result.get('latency',{});print(cid,'n=',latency.get('n',0),'failures=',latency.get('failures',0),
            [(sid,round(v['wall_s']['p50'],4),round(v['wall_s']['p95'],4)) for sid,v in latency.get('by_sample',{}).items() if 'wall_s' in v])


if __name__=='__main__':
    main()
