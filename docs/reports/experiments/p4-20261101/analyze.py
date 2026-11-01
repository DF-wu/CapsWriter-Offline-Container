import json,math,statistics,re,unicodedata,collections,sys
from pathlib import Path
root=Path(sys.argv[1]);groups=collections.defaultdict(list);metadata=collections.defaultdict(list)
for f in root.glob('*.jsonl'):
 if f.name.startswith('pilot-'):continue
 for line in f.read_text().splitlines():
  r=json.loads(line)
  if r['type']=='metadata':metadata[r['config']].append(r)
  elif not r['warmup']:groups[(r['config'],r['phase'])].append(r)
def dist(a,b):
 row=list(range(len(b)+1))
 for i,c in enumerate(a,1):
  nxt=[i]
  for j,d in enumerate(b,1):nxt.append(min(row[j]+1,nxt[-1]+1,row[j-1]+(c!=d)))
  row=nxt
 return row[-1]
def norm(s):
 s=unicodedata.normalize('NFKC',s).lower();s=re.sub(r'\bmr\.', 'mister',s)
 return ''.join(c for c in s if not unicodedata.category(c).startswith(('P','Z')) and not c.isspace())
def quant(vals,p):
 vals=sorted(vals);return vals[max(0,math.ceil(p*len(vals))-1)]
summary={}
for (config,phase),rows in sorted(groups.items()):
 if phase=='pilot':continue
 d={'n':len(rows),'failures':sum(r['status']!='ok' for r in rows),'by_sample':{}}
 for sid in sorted({r['id'] for r in rows}):
  rs=[r for r in rows if r['id']==sid];good=[r for r in rs if r['status']=='ok' and math.isfinite(r['wall_s']) and r['wall_s']>=0]
  v={'n':len(rs),'ok':len(good),'outputs':dict(collections.Counter(r['text'] for r in rs))}
  for key in ['wall_s','encoder_s','frontend_s','backend_s','prefill_s','generate_s','n_generate']:
   values=[r[key] for r in good if key in r]
   if values:v[key]={'p50':statistics.median(values),'p95':quant(values,.95)}
  ref=rs[0].get('reference')
  if ref:
   v['reference']=ref;v['normalized_reference_chars']=len(norm(ref));v['cer_no_script_or_number_conversion']=[dist(norm(ref),norm(r['text']))/len(norm(ref)) for r in rs]
  clocks=[];thermal=0
  for r in good:
   for key in ['gpu_before','gpu_after']:
    fields=r.get(key,'').split(', ')
    if len(fields)==7:clocks.append(int(fields[1]));thermal+=fields[5]=='Active'
  if clocks:v['graphics_mhz_range']=[min(clocks),max(clocks)];v['thermal_active_snapshots']=thermal;v['thermal_snapshots_n']=len(clocks)
  d['by_sample'][sid]=v
 summary.setdefault(config,{})[phase]=d
(root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
for c,phases in summary.items():
 if 'latency' in phases:print(c,'n=',phases['latency']['n'],[(s,round(v.get('wall_s',{}).get('p50',-1),3),round(v.get('wall_s',{}).get('p95',-1),3)) for s,v in phases['latency']['by_sample'].items()])
