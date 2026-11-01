"""Serial exclusive-GPU experiment with exact-container restoration.

Run --pilot first while service remains up; run --matrix only after pilots.
Warmups/pilot/cooldown are recorded separately from measured latency.
"""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from lifecycle import Maintenance, inspect as inspect_service, services, utc

STATE = Path('/tmp/capswriter-p4-exclusive-state.json')
ORIGINAL = '/models-original/Qwen3-ASR/Qwen3-ASR-1.7B'
CONFIGS = [
    {'id': 'cpu-fallback', 'backend': 'cpu', 'model': ORIGINAL},
    {'id': 'vulkan-q5', 'backend': 'vulkan', 'model': ORIGINAL},
    {'id': 'cuda-q5', 'backend': 'cuda', 'model': ORIGINAL},
    {'id': 'cuda-f32-q5', 'backend': 'cuda', 'model': '/bench/models/fp32-promoted'},
    {'id': 'cuda-q4', 'backend': 'cuda', 'model': '/bench/models/q4'},
    {'id': 'cuda-w4-q4', 'backend': 'cuda', 'model': '/bench/models/q4-full'},
    {'id': 'cuda-q6', 'backend': 'cuda', 'model': '/bench/models/q6'},
    {'id': 'cuda-q8', 'backend': 'cuda', 'model': '/bench/models/q8'},
    {'id': 'cuda-iq4-xs', 'backend': 'cuda', 'model': '/bench/models/iq4'},
    *[{'id': 'transcribe-'+quant.lower(), 'backend': 'transcribe',
       'model': '/bench/downloads/Qwen3-ASR-1.7B-'+quant+'.gguf'}
      for quant in ['Q4_K_M', 'Q5_K_M', 'Q6_K', 'Q8_0']]]


def gpu():
    r = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu,clocks.current.graphics,memory.used,utilization.gpu,power.draw,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.hw_thermal_slowdown',
                        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    return r.stdout.strip()


def cooldown(root, config, target=78, budget=60):
    start=time.monotonic();observations=[]
    while True:
        value=gpu();observations.append({'elapsed_s':time.monotonic()-start,'gpu':value})
        temperature=int(value.split(',')[0])
        if temperature <= target or time.monotonic()-start >= budget:break
        time.sleep(3)
    result={'config':config,'target_c':target,'budget_s':budget,'target_reached':temperature<=target,
            'observations':observations,'completed_utc':utc()}
    with open(root/'results/cooldowns.jsonl','a') as stream:stream.write(json.dumps(result)+'\n')
    print('COOLDOWN',config,'reached',result['target_reached'],'last',value,flush=True)


def no_external_gpu(state):
    assert all(not inspect_service(service)['State']['Running'] for service in services(state))
    # An exited docker exec may take a short time to release its CUDA context.
    deadline=time.monotonic()+5
    while True:
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory',
                                      '--format=csv,noheader,nounits'],text=True).strip()
        if not apps:return
        if time.monotonic()>=deadline:raise RuntimeError('Unexpected GPU process before test block: '+apps)
        time.sleep(1)


def launch(state, config, phase, repeats, warmup, pilot=False):
    root=Path(state['scratch']);name=config['id'];backend=config['backend']
    output='/bench/results/'+('pilot-' if pilot else '')+name+'.jsonl'
    args=['docker','exec']
    if backend in ['vulkan','cpu']:
        args += ['-e','VK_ICD_FILENAMES='+('/bench/no-such-icd.json' if backend=='cpu' else '/etc/vulkan/icd.d/nvidia_icd.json')]
    if backend=='transcribe':
        args += [state['container'],'python','/bench/run_transcribe.py','--model',config['model']]
    else:
        link=root/'src/core/server/engines/llama/bin';link.unlink()
        link.symlink_to('/bench/'+('vulkan-bin' if backend in ['vulkan','cpu'] else 'cuda-build/bin'))
        args += [state['container'],'python','/bench/run_engine.py','--model-dir',config['model']]
        if backend=='cpu':args += ['--allow-cpu-decoder']
    args += ['--samples','/bench/'+('pilot-samples.json' if pilot else 'samples.json'),
             '--output',output,'--config-id',name,'--phase',phase,'--repeats',str(repeats),'--warmup',str(warmup)]
    before=gpu();start=time.monotonic()
    with open(root/'logs'/(name+'-'+phase+'-'+str(time.time_ns())+'.log'),'w') as log:
        try:
            result=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=900)
            code=result.returncode
        except subprocess.TimeoutExpired:
            # Kill the contained process too; killing the docker client alone
            # could leave a competing GPU workload alive for the next block.
            subprocess.run(['docker','stop','-t','10',state['container']],capture_output=True)
            raise
    info={'config':name,'phase':phase,'repeats':repeats,'warmup':warmup,'exit':code,
          'elapsed_s':time.monotonic()-start,'gpu_before':before,'gpu_after':gpu(),'completed_utc':utc()}
    print(json.dumps(info),flush=True)
    return info


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pilot',action='store_true');p.add_argument('--matrix',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--only',nargs='*');a=p.parse_args()
    assert a.pilot != a.matrix
    state=json.loads(STATE.read_text());root=Path(state['scratch'])
    lock=open(root/'benchmark.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    shutil.copyfile(Path(__file__).with_name('run_transcribe.py'),root/'run_transcribe.py')
    (root/'results/configs.json').write_text(json.dumps(CONFIGS,indent=2)+'\n')
    if a.pilot:
        schedule_path=root/'results/pilot-schedule.json'
        outcomes=json.loads(schedule_path.read_text()) if schedule_path.exists() else []
        for config in CONFIGS:
            if a.only and config['id'] not in a.only:continue
            print('PILOT',config['id'],flush=True)
            info=launch(state,config,'pilot',1,0,pilot=True)
            outcomes=[r for r in outcomes if r['config']!=config['id']]+[info]
            schedule_path.write_text(json.dumps(outcomes,indent=2)+'\n')
        return
    outcomes=json.loads((root/'results/pilot-schedule.json').read_text())
    passed={r['config'] for r in outcomes if r['exit']==0}
    # Nonzero exits and failed/blank/aborted pilot calls remain visible but
    # are not silently promoted to the performance candidate list.
    for config in list(passed):
        rows=[json.loads(line) for line in (root/'results'/('pilot-'+config+'.jsonl')).read_text().splitlines()]
        if not any(r['type']=='sample' and r['status']=='ok' for r in rows):passed.remove(config)
    configs=[c for c in CONFIGS if c['id'] in passed]
    assert any(c['id']=='cuda-q5' for c in configs) and any(c['id']=='cuda-q6' or c['id']=='transcribe-q6_k' for c in configs)
    schedule_path=root/'results/schedule.json'
    schedule=json.loads(schedule_path.read_text()) if a.resume and schedule_path.exists() else []
    completed={(r['config'],r['block']) for r in schedule if r['exit']==0}
    gpu_configs=[c for c in configs if c['backend']!='cpu']
    telemetry_file=open(root/'results/gpu-telemetry.csv','a' if a.resume else 'w')
    telemetry=None
    with Maintenance() as maintenance:
        try:
            telemetry=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,temperature.gpu,clocks.current.graphics,memory.used,utilization.gpu,power.draw,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.hw_thermal_slowdown',
                                        '--format=csv','-lms','200'],stdout=telemetry_file,stderr=subprocess.DEVNULL)
            no_external_gpu(state);cooldown(root,'initial-empty-GPU',target=78,budget=120)
            cpu=next((c for c in configs if c['backend']=='cpu'),None)
            if cpu and (cpu['id'],'cpu-baseline') not in completed:
                info=launch(state,cpu,'latency',10,3);info['block']='cpu-baseline';schedule.append(info)
                (root/'results/schedule.json').write_text(json.dumps(schedule,indent=2)+'\n')
            for block in range(2):
                pivot=0 if block==0 else len(gpu_configs)//2
                order=gpu_configs[pivot:]+gpu_configs[:pivot]
                for config in order:
                    if (config['id'],block) in completed:continue
                    no_external_gpu(state);cooldown(root,config['id'])
                    print('MATRIX',block,config['id'],flush=True)
                    info=launch(state,config,'latency',5,3);info['block']=block;schedule.append(info)
                    (root/'results/schedule.json').write_text(json.dumps(schedule,indent=2)+'\n')
                    if info['exit']!=0:raise RuntimeError('Matrix process failed: '+config['id'])
            for config in configs:
                if (config['id'],'quality') in completed:continue
                no_external_gpu(state);cooldown(root,config['id']+'-quality')
                info=launch(state,config,'quality',1,1);info['block']='quality';schedule.append(info)
                (root/'results/schedule.json').write_text(json.dumps(schedule,indent=2)+'\n')
                if info['exit']!=0:raise RuntimeError('Quality process failed: '+config['id'])
        finally:
            if telemetry:
                telemetry.terminate()
                try:telemetry.wait(timeout=5)
                except subprocess.TimeoutExpired:telemetry.kill();telemetry.wait()
            telemetry_file.close()


if __name__=='__main__':
    main()
