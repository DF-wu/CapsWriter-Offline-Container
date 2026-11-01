"""Read-only post-benchmark hashes and conversion/runtime checks, in test container."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0,'/bench/src/core/server/engines/qwen_asr_gguf/export')
from gguf import GGUFReader

root=Path('/bench');original=Path('/models-original/Qwen3-ASR/Qwen3-ASR-1.7B')
expected={'qwen3_asr_encoder_frontend.onnx':'fc288ba9a0ca732251b33628c803cd5188d3dd77a2cd7336f0f0fd37969ed4cd',
          'qwen3_asr_encoder_backend.onnx':'4b0b1e34e8393f29222006aae0719283b0ff28b61f289b6ff8377b3ddbda27e3',
          'qwen3_asr_llm.gguf':'89e198470f41d487ab5b6b8a36a35400b3b1534451c4499236983050fe9193e5'}


def artifact(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest()}


result={'packages':{p:importlib.metadata.version(p) for p in ['numpy','onnx','onnxruntime-gpu','soundfile']},
        'artifacts':[],'gguf':{},'encoder_cpu_finite':{}}
for name,sha in expected.items():
    item=artifact(original/name);assert item['sha256']==sha
    result['artifacts'].append(dict(item,original_unchanged=True))
for path in [root/'hf/model.safetensors',root/'models/tiro-decoder-f16.gguf',
             root/'models/tiro-q5/qwen3_asr_llm.gguf',root/'models/tiro-q4/qwen3_asr_llm.gguf',
             root/'models/tiro-q5/qwen3_asr_encoder_frontend.onnx',
             root/'models/tiro-q5/qwen3_asr_encoder_backend.onnx']:
    result['artifacts'].append(artifact(path))
for path in sorted((root/'vulkan-bin').glob('*.so*')):
    if path.is_file() and not path.is_symlink():result['artifacts'].append(artifact(path))
for cid,path in [('original-q5',original/'qwen3_asr_llm.gguf'),('tiro-f16',root/'models/tiro-decoder-f16.gguf'),
                 ('tiro-q5',root/'models/tiro-q5/qwen3_asr_llm.gguf'),('tiro-q4',root/'models/tiro-q4/qwen3_asr_llm.gguf')]:
    reader=GGUFReader(str(path))
    result['gguf'][cid]={'architecture':reader.fields['general.architecture'].contents(),
        'tensor_count':len(reader.tensors),'file_type':reader.fields['general.file_type'].contents(),
        'tensor_types':{t.name:str(t.tensor_type) for t in reader.tensors}}
    assert len(reader.tensors)==311
result['q5_type_recipe_matches_original']=(result['gguf']['original-q5']['tensor_types']==result['gguf']['tiro-q5']['tensor_types'])
# Independent CPU EP load: valid graph and finite arbitrary-shape outputs. This
# is a structural/numerical smoke, not HF/PyTorch full-model parity validation.
opts=ort.SessionOptions();opts.intra_op_num_threads=4
fe=ort.InferenceSession(str(root/'models/tiro-q5/qwen3_asr_encoder_frontend.onnx'),opts,providers=['CPUExecutionProvider'])
be=ort.InferenceSession(str(root/'models/tiro-q5/qwen3_asr_encoder_backend.onnx'),opts,providers=['CPUExecutionProvider'])
for session,label in [(fe,'frontend'),(be,'backend')]:
    result['encoder_cpu_finite'][label]={'inputs':[(x.name,x.shape,x.type) for x in session.get_inputs()],
                                       'outputs':[(x.name,x.shape,x.type) for x in session.get_outputs()]}
fe_type=np.float16 if 'float16' in fe.get_inputs()[0].type else np.float32
hidden=fe.run(None,{fe.get_inputs()[0].name:np.zeros((1,128,100),dtype=fe_type)})[0]
assert hidden.shape==(1,13,1024) and np.isfinite(hidden).all()
feeds={}
for inp in be.get_inputs():
    dtype=np.float16 if 'float16' in inp.type else np.float32
    feeds[inp.name]=(np.zeros((1,1,13,13),dtype=dtype) if len(inp.shape)==4 else hidden.astype(dtype))
embedding=be.run(None,feeds)[0]
assert embedding.shape==(1,13,2048) and np.isfinite(embedding).all()
result['encoder_cpu_finite']['smoke']={'frontend_shape':list(hidden.shape),
    'backend_shape':list(embedding.shape),'all_finite':True,'input':'zeros; structural smoke only'}
(root/'results/environment.json').write_text(json.dumps(result,indent=2)+'\n')
print('Verified original hashes, converted files, GGUF recipes and independent CPU ONNX load.')
