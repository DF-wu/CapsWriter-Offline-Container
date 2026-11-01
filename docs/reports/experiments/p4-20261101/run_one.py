import subprocess,json,sys,time
from pathlib import Path
state=json.loads(Path('/tmp/capswriter-p4-experiment-state.json').read_text());root=Path(state['scratch'])
config,model,backend,phase,reps=sys.argv[1:]
link=root/'src/core/server/engines/llama/bin';link.unlink();link.symlink_to('/bench/'+('vulkan-bin' if backend in ['vulkan','cpu-fallback'] else 'cuda-build/bin'))
args=['docker','exec']
if backend=='vulkan':args+=['-e','VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json']
if backend=='cpu-fallback':args+=['-e','VK_ICD_FILENAMES=/bench/no-such-icd.json']
args+=[state['container'],'python','/bench/run_engine.py','--model-dir',model,'--samples','/bench/samples.json','--output','/bench/results/'+config+'.jsonl','--config-id',config,'--phase',phase,'--repeats',reps,'--warmup','3']
if backend=='cpu-fallback':args+=['--allow-cpu-decoder']
with open(root/'logs'/f'{config}-{phase}-{time.time_ns()}.log','w') as log:
 try:r=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=900);code=r.returncode
 except subprocess.TimeoutExpired:code=124
print(json.dumps({'config':config,'phase':phase,'exit':code}),flush=True)
raise SystemExit(code)
