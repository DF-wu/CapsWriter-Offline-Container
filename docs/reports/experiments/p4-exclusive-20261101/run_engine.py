import argparse,json,time,sys,hashlib,functools,os,subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
p=argparse.ArgumentParser()
p.add_argument('--allow-cpu-decoder',action='store_true');p.add_argument('--model-dir',required=True);p.add_argument('--samples',required=True);p.add_argument('--output',required=True);p.add_argument('--config-id',required=True);p.add_argument('--repeats',type=int,default=1);p.add_argument('--temperature',type=float,default=0);p.add_argument('--phase',default='latency');p.add_argument('--warmup',type=int,default=1)
a=p.parse_args()
import logging,ctypes
from core.server.engines import logger
logger.addHandler(logging.StreamHandler(sys.stderr));logger.setLevel(logging.INFO)
from core.server.engines.qwen_asr_gguf.inference.schema import ASREngineConfig
from core.server.engines.qwen_asr_gguf.asr_engine import QwenASREngine
cfg=ASREngineConfig(model_dir=a.model_dir,encoder_frontend_fn='qwen3_asr_encoder_frontend.onnx',encoder_backend_fn='qwen3_asr_encoder_backend.onnx',llm_fn='qwen3_asr_llm.gguf',onnx_provider='CUDA',llm_use_gpu=True,n_ctx=2048,n_batch=4096,n_ubatch=512,n_threads=4,n_threads_batch=4,flash_attn=True,verbose=False,chunk_size=80)
t=time.perf_counter();adapter=QwenASREngine(cfg);load_s=time.perf_counter()-t;engine=adapter.engine
state={}
def timed(obj,name,key):
 orig=getattr(obj,name)
 @functools.wraps(orig)
 def wrapped(*args,**kwargs):
  t=time.perf_counter();res=orig(*args,**kwargs);state[key]=state.get(key,0)+time.perf_counter()-t
  if key=='decode_call':
   state['prefill_s']=state.get('prefill_s',0)+res.t_prefill;state['generate_s']=state.get('generate_s',0)+res.t_generate;state['n_prefill']=res.n_prefill;state['n_generate']=state.get('n_generate',0)+res.n_generate;state['decode_attempts']=state.get('decode_attempts',0)+1;state['aborted']=res.is_aborted
  return res
 setattr(obj,name,wrapped)
for method,key in [('_run_frontend','frontend_s'),('_run_backend','backend_s'),('encode','encoder_s')]:timed(engine.encoder,method,key)
timed(engine,'_decode','decode_call')
run_id=str(time.time_ns())
meta={'type':'metadata','run_id':run_id,'config':a.config_id,'load_s':load_s,'onnx_providers':{'frontend':engine.encoder.sess_fe.get_providers(),'backend':engine.encoder.sess_be.get_providers()},'input_dtype':str(engine.encoder.input_dtype),'parameters':vars(a),'library_path':str(Path('/bench/src/core/server/engines/llama/bin').resolve())}
from core.server.engines.llama import llama as l
g=l.ggml;g.ggml_backend_dev_count.restype=ctypes.c_size_t;g.ggml_backend_dev_get.argtypes=[ctypes.c_size_t];g.ggml_backend_dev_get.restype=ctypes.c_void_p;g.ggml_backend_dev_name.argtypes=[ctypes.c_void_p];g.ggml_backend_dev_name.restype=ctypes.c_char_p
meta['ggml_devices']=[g.ggml_backend_dev_name(g.ggml_backend_dev_get(i)).decode() for i in range(g.ggml_backend_dev_count())]
if not a.allow_cpu_decoder and not any('Vulkan' in d or 'CUDA' in d for d in meta['ggml_devices']):raise RuntimeError('GPU decoder backend not available')
samples=[x for x in json.loads(Path(a.samples).read_text()) if x.get('phase','quality')==a.phase]
if not samples:raise ValueError('no samples')
loaded={}
for s in samples:
 y,sr=sf.read(s['path'],dtype='float32');assert sr==16000 and y.ndim==1;loaded[s['id']]=y
def gpu_snapshot():
 try:
  r=subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu,clocks.current.graphics,memory.used,utilization.gpu,power.draw,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.hw_thermal_slowdown','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
  return r.stdout.strip()
 except Exception as e:return str(e)
with open(a.output,'a',buffering=1) as out:
 out.write(json.dumps(meta,ensure_ascii=False)+'\n')
 seq=[(True,i,samples[i%len(samples)]) for i in range(a.warmup)]+[(False,r,s) for r in range(a.repeats) for s in samples]
 for warm,r,s in seq:
  state.clear();before=gpu_snapshot();np.random.seed(1729+r);stream=adapter.create_stream();stream.accept_waveform(16000,loaded[s['id']]);wall_start_unix_ns=time.time_ns();start=time.perf_counter()
  try:
   adapter.decode_stream(stream,language=s.get('language'),context=None,temperature=a.temperature);elapsed=time.perf_counter()-start;text=stream.result.text;status='ok' if text.strip() and not state.get('aborted') else 'invalid_output';error=None
  except Exception as e:
   elapsed=time.perf_counter()-start;text='';status='error';error=type(e).__name__+': '+str(e)
  row={'type':'sample','run_id':run_id,'config':a.config_id,'phase':a.phase,'id':s['id'],'repeat':r,'warmup':warm,'wall_s':elapsed,'wall_start_unix_ns':wall_start_unix_ns,'wall_end_unix_ns':time.time_ns(),'audio_s':len(loaded[s['id']])/16000,'text':text,'reference':s.get('reference'),'status':status,'error':error,'gpu_before':before,'gpu_after':gpu_snapshot(),**state}
  out.write(json.dumps(row,ensure_ascii=False)+'\n');print(json.dumps({k:row[k] for k in ['config','id','warmup','wall_s','status']}) ,flush=True)
