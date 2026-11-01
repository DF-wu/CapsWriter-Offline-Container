"""Same public controls, with deterministic English stress probes (not FFASR)."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request

STATE = Path('/tmp/capswriter-tiro-v2-state.json')
HERE = Path(__file__).resolve().parent
state = json.loads(STATE.read_text()); root = Path(state['scratch'])
(root / 'audio').mkdir(exist_ok=True)
samples = json.loads((HERE.parent / 'p4-exclusive-20261101/samples.json').read_text())
url = samples[0]['dataset_rows_api']
response = json.load(urllib.request.urlopen(url, timeout=45))
for item in response['rows']:
    row = item['row']; sample = next(s for s in samples if s['id'] == 'libri-'+row['id'])
    assert sample['reference'] == row['text']
    with (urllib.request.urlopen(row['audio'][0]['src'], timeout=45) as source,
          (root / sample['path'].removeprefix('/bench/')).open('wb') as target):
        shutil.copyfileobj(source, target)
for sid in ['noise2', 'codeswitch']:
    sample = next(s for s in samples if s['id'] == sid)
    path = root / 'audio' / (sid+'.wav')
    with (urllib.request.urlopen(sample['source'].replace('/blob/', '/resolve/'), timeout=45) as source,
          path.open('wb') as target):
        shutil.copyfileobj(source, target)
subprocess.run(['docker','exec',state['container'],'ffmpeg','-y','-loglevel','error',
                '-i','/bench/audio/codeswitch.wav','-ar','16000','-ac','1','/bench/audio/codeswitch-16k.wav'],check=True)
(root / 'samples-base.json').write_text(json.dumps(samples,ensure_ascii=False,indent=2)+'\n')
code = r'''
import hashlib,json
from pathlib import Path
import numpy as np
import soundfile as sf
r=Path('/bench');samples=json.loads((r/'samples-base.json').read_text())
y,sr=sf.read('/bench/audio/noise2.wav',dtype='float32');assert sr==16000
for n in [2,5,10]:sf.write('/bench/audio/mandarin-'+str(n)+'s.wav',y[:n*16000],16000)
for s in samples:
    p=Path(s['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==s['audio_sha256'],s['id']
    if s['id'].startswith('libri-'):s['condition']='clean English'
for index,s in enumerate(list(samples)):
    if not s['id'].startswith('libri-'):continue
    y,sr=sf.read(s['path'],dtype='float32');assert sr==16000
    power=float(np.mean(y.astype('float64')**2))
    rng=np.random.default_rng(20261101+index)
    noise=rng.standard_normal(len(y));noise/=np.sqrt(np.mean(noise**2))
    for snr in [10,0]:
        noisy=y+noise*np.sqrt(power/(10**(snr/10)))
        # Uniform gain prevents clipping while preserving SNR.
        noisy=noisy/max(1.,float(np.max(np.abs(noisy)))/.95)
        name=s['id']+'-white-'+str(snr)+'db';p=r/'audio'/(name+'.wav')
        sf.write(p,noisy,16000,subtype='FLOAT')
        samples.append(dict(s,id=name,path=str(p),audio_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            condition='white noise '+str(snr)+' dB',transform='seed='+str(20261101+index)+'; white-noise power relative to whole-clip RMS; shared noise across SNR; gain limited to .95 peak'))
    # Fixed sparse echo probe; not a physical room impulse response or FFASR.
    taps=[(0,1.),(80,.4),(150,.25),(230,.15),(310,.08)]
    z=np.zeros(len(y)+int(.310*sr),dtype=np.float64)
    for ms,gain in taps:
        offset=int(ms*.001*sr);z[offset:offset+len(y)]+=gain*y
    z=z/max(1.,float(np.max(np.abs(z)))/.95)
    name=s['id']+'-echo';p=r/'audio'/(name+'.wav');sf.write(p,z,16000,subtype='FLOAT')
    samples.append(dict(s,id=name,path=str(p),audio_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        condition='synthetic echo',duration_s=len(z)/sr,transform='linear delayed copies: [(0ms,1),(80,.4),(150,.25),(230,.15),(310,.08)]; peak limited .95; not measured RIR'))
    samples[-3]['duration_s']=len(y)/sr;samples[-2]['duration_s']=len(y)/sr
(r/'samples.json').write_text(json.dumps(samples,ensure_ascii=False,indent=2)+'\n')
print('Prepared',sum(s.get('phase','quality')=='quality' for s in samples),'quality probes and 3 latency crops')
'''
subprocess.run(['docker','exec',state['container'],'python','-c',code],check=True)
