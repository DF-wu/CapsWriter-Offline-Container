import json,subprocess,time
from pathlib import Path
s=json.loads(Path('/tmp/capswriter-p4-experiment-state.json').read_text());root=Path(s['scratch']);one=root/'run_one.py'
configs=[('vulkan-f16-q5','/models-original/Qwen3-ASR/Qwen3-ASR-1.7B','vulkan'),('cuda-f16-q5','/models-original/Qwen3-ASR/Qwen3-ASR-1.7B','cuda'),('cuda-f32-q5','/bench/models/fp32-promoted','cuda'),('cuda-f16-q4','/bench/models/q4','cuda'),('cuda-int4-q4','/bench/models/q4-full','cuda')]
telemetry_file=open(root/'results/gpu-telemetry.csv','w')
telemetry=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,temperature.gpu,clocks.current.graphics,memory.used,utilization.gpu,power.draw,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.hw_thermal_slowdown','--format=csv','-lms','200'],stdout=telemetry_file,stderr=subprocess.DEVNULL)
try:
 schedule=[]
 for block,reps in enumerate([4,3,3]):
  order=configs[block:]+configs[:block]
  for config,model,backend in order:
   print('START',block,config,flush=True);t=time.monotonic();r=subprocess.run(['python3',str(one),config,model,backend,'latency',str(reps)],timeout=1000);schedule.append({'block':block,'config':config,'repeats':reps,'exit':r.returncode,'elapsed_s':time.monotonic()-t});(root/'results/schedule.json').write_text(json.dumps(schedule,indent=2))
   if r.returncode:raise SystemExit(r.returncode)
 for config,model,backend in configs+[('shipped-cpu-fallback','/models-original/Qwen3-ASR/Qwen3-ASR-1.7B','cpu-fallback')]:
  print('QUALITY',config,flush=True);r=subprocess.run(['python3',str(one),config,model,backend,'quality','1'],timeout=1000)
  if r.returncode:raise SystemExit(r.returncode)
finally:
 telemetry.terminate()
 try:telemetry.wait(timeout=5)
 except subprocess.TimeoutExpired:telemetry.kill();telemetry.wait()
 telemetry_file.close()
