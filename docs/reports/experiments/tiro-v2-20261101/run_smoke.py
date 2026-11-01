"""Small same-runtime model experiment; restores both original services."""
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time

from lifecycle import Maintenance, STATE, inspect, services, utc

HERE = Path(__file__).resolve().parent
CONFIGS = [
    {'id':'original-q5','model':'/models-original/Qwen3-ASR/Qwen3-ASR-1.7B'},
    {'id':'tiro-f16','model':'/bench/models/tiro-f16'},
    {'id':'tiro-q5','model':'/bench/models/tiro-q5'},
    {'id':'tiro-q4','model':'/bench/models/tiro-q4'}]


def gpu():
    return subprocess.check_output(['nvidia-smi',
        '--query-gpu=temperature.gpu,clocks.current.graphics,memory.used,utilization.gpu,power.draw,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.hw_thermal_slowdown',
        '--format=csv,noheader,nounits'],text=True).strip()


def check_empty(state):
    assert all(not inspect(s)['State']['Running'] for s in services(state))
    for attempt in range(6):
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory',
            '--format=csv,noheader,nounits'],text=True).strip()
        if not apps:
            return
        if attempt==5:
            raise RuntimeError('Unexpected GPU process: '+apps)
        time.sleep(1)


def main():
    state=json.loads(STATE.read_text());root=Path(state['scratch'])
    lock=open(root/'benchmark.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    shutil.copy2(HERE.parent/'p4-exclusive-20261101/run_engine.py',root/'run_engine.py')
    link=root/'src/core/server/engines/llama/bin'
    if link.is_symlink():link.unlink()
    elif link.exists():shutil.rmtree(link)
    link.symlink_to('/bench/vulkan-bin')
    (root/'results/configs.json').write_text(json.dumps(CONFIGS,indent=2)+'\n')
    schedule=[]
    with Maintenance(max_seconds=1200):
        telemetry_file=open(root/'results/gpu-telemetry.csv','w')
        telemetry=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,temperature.gpu,clocks.current.graphics,memory.used,utilization.gpu,power.draw,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.hw_thermal_slowdown',
            '--format=csv','-lms','200'],stdout=telemetry_file,stderr=subprocess.DEVNULL)
        try:
            for phase,repeats,warmup in [('quality',1,1),('latency',3,3)]:
                for config in CONFIGS:
                    check_empty(state);start=time.monotonic()
                    while int(gpu().split(',')[0])>78 and time.monotonic()-start<90:
                        time.sleep(3)
                    before=gpu();print('Running',phase,config['id'],before,flush=True)
                    args=['docker','exec','-e','VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json',state['container'],
                        'python','/bench/run_engine.py','--model-dir',config['model'],
                        '--samples','/bench/samples.json','--output','/bench/results/'+config['id']+'.jsonl',
                        '--config-id',config['id'],'--phase',phase,'--repeats',str(repeats),'--warmup',str(warmup)]
                    t=time.monotonic()
                    with open(root/'logs'/(config['id']+'-'+phase+'.log'),'w') as log:
                        try:
                            result=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=240)
                        except subprocess.TimeoutExpired:
                            subprocess.run(['docker','stop','-t','10',state['container']],capture_output=True)
                            raise
                    item={'config':config['id'],'phase':phase,'exit':result.returncode,'elapsed_s':time.monotonic()-t,
                          'gpu_before':before,'gpu_after':gpu(),'completed_utc':utc()}
                    schedule.append(item)
                    (root/'results/schedule.json').write_text(json.dumps(schedule,indent=2)+'\n')
                    print(json.dumps(item),flush=True)
                    assert result.returncode==0,config['id']
                    rows=[json.loads(l) for l in (root/'results'/(config['id']+'.jsonl')).read_text().splitlines()]
                    assert all(x['status']=='ok' for x in rows if x.get('type')=='sample')
                    check_empty(state)
                    # Persist acceptance only after output and exclusivity checks.
                    run_ids={x['run_id'] for x in rows if x.get('type')=='sample' and x['phase']==phase}
                    assert len(run_ids)==1
                    item.update(accepted=True,run_id=next(iter(run_ids)),accepted_utc=utc())
                    (root/'results/schedule.json').write_text(json.dumps(schedule,indent=2)+'\n')
        finally:
            telemetry.terminate()
            try:telemetry.wait(timeout=5)
            except subprocess.TimeoutExpired:telemetry.kill();telemetry.wait()
            telemetry_file.close()


if __name__=='__main__':main()
