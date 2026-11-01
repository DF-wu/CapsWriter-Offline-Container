"""Convert pinned Tiro BF16 to the existing split ONNX / qwen3vl contract.

Run only in the disposable container. Graphs/tokenizer metadata come from the
read-only existing model, but EVERY learned weight comes from the source BF16.
This is a compatibility conversion, not a claim of optimal P4 quantization.
"""
import hashlib
import json
import mmap
from pathlib import Path
import re
import struct
import sys

import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0, '/bench/src/core/server/engines/qwen_asr_gguf/export')
from gguf import GGUFReader, GGUFWriter, GGUFValueType

ROOT = Path('/bench')
ORIGINAL = Path('/models-original/Qwen3-ASR/Qwen3-ASR-1.7B')
DEST = ROOT / 'models/tiro-q5'
DEST.mkdir(parents=True, exist_ok=True)
file = (ROOT / 'hf/model.safetensors').open('rb')
data = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
header_size = struct.unpack('<Q', data[:8])[0]
header = json.loads(data[8:8+header_size])
used = set()
records = []


def tensor(name):
    info = header[name]
    assert info['dtype'] == 'BF16', (name, info['dtype'])
    start, end = info['data_offsets']
    raw = np.frombuffer(data, dtype='<u2', count=(end-start)//2,
                        offset=8+header_size+start)
    result = (raw.astype(np.uint32) << 16).view(np.float32).reshape(info['shape'])
    assert np.isfinite(result).all(), name
    used.add(name)
    return result


def audio_name(name):
    if name.startswith('proj1.'):
        return name.replace('proj1.', 'model.multi_modal_projector.linear_1.', 1)
    if name.startswith('proj2.'):
        return name.replace('proj2.', 'model.multi_modal_projector.linear_2.', 1)
    return 'model.audio_tower.' + name


for part in ['frontend', 'backend']:
    model = onnx.load(ORIGINAL / ('qwen3_asr_encoder_'+part+'.onnx'))
    initializers = {x.name: x for x in model.graph.initializer}
    mapping = {}
    for name in initializers:
        source = audio_name(name)
        if source in header:
            mapping[name] = (source, False)
    for node in model.graph.node:
        if node.op_type != 'MatMul' or node.input[1] not in initializers:
            continue
        if part == 'frontend':
            source = 'model.audio_tower.conv_out.weight'
        else:
            consumers = [n for n in model.graph.node if node.output[0] in n.input]
            biases = {v for n in consumers for v in n.input
                      if v in initializers and v.endswith('.bias')}
            assert len(biases) == 1, (node.name, biases)
            source = audio_name(next(iter(biases)).removesuffix('.bias')+'.weight')
        assert source in header, source
        mapping[node.input[1]] = (source, True)
    for name, (source, transpose) in mapping.items():
        old = numpy_helper.to_array(initializers[name])
        value = tensor(source)
        if transpose:
            value = value.T
        assert value.shape == old.shape, (name, value.shape, old.shape)
        value = value.astype(old.dtype)
        initializers[name].CopyFrom(numpy_helper.from_array(value, name))
        records.append({'part': part, 'initializer': name, 'source': source,
                        'transpose': transpose, 'shape': list(value.shape),
                        'dtype': str(value.dtype)})
    # Original ORT graph contains Cast nodes after consumers. Sort it without
    # changing operators to allow an independent ONNX checker pass.
    available = set(initializers) | {x.name for x in model.graph.input}
    remaining = list(model.graph.node); ordered = []
    while remaining:
        ready = [n for n in remaining if all(not x or x in available for x in n.input)]
        assert ready, 'Graph has an unresolved dependency'
        for node in ready:
            ordered.append(node); available.update(node.output); remaining.remove(node)
    del model.graph.node[:]; model.graph.node.extend(ordered)
    onnx.checker.check_model(model)
    onnx.save(model, DEST / ('qwen3_asr_encoder_'+part+'.onnx'))
    print('Mapped encoder', part, len(mapping), flush=True)

reader = GGUFReader(str(ORIGINAL / 'qwen3_asr_llm.gguf'))
tokens = reader.fields['tokenizer.ggml.tokens'].contents()
tokenizer = json.loads((ROOT / 'hf/tokenizer.json').read_text())
vocab = dict(tokenizer['model']['vocab'])
vocab.update({x['content']: x['id'] for x in tokenizer['added_tokens']})
for text, index in vocab.items():
    assert tokens[index] == text, ('tokenizer mismatch', index, text, tokens[index])
assert reader.fields['tokenizer.ggml.merges'].contents() == [
    ' '.join(x) if isinstance(x, list) else x for x in tokenizer['model']['merges']]
config = json.loads((ROOT / 'hf/config.json').read_text())
assert config['tie_word_embeddings'] and config['text_config']['tie_word_embeddings']
tc = config['text_config']
for key, value in [('block_count',tc['num_hidden_layers']),('embedding_length',tc['hidden_size']),
                   ('feed_forward_length',tc['intermediate_size']),
                   ('attention.head_count',tc['num_attention_heads']),
                   ('attention.head_count_kv',tc['num_key_value_heads'])]:
    assert reader.fields['qwen3vl.'+key].contents() == value

target = ROOT / 'models/tiro-decoder-f16.gguf'
writer = GGUFWriter(str(target), 'qwen3vl')
for key, field in reader.fields.items():
    if key.startswith('GGUF.') or key in ['general.architecture','general.quantization_version','general.file_type']:
        continue
    writer.add_key_value(key, field.contents(), field.types[0],
                         field.types[-1] if field.types[0] == GGUFValueType.ARRAY else None)
writer.add_string('general.name', 'Tiro v2 decoder compatibility conversion')
writer.add_uint32('general.file_type', 1)
writer.add_string('general.source.url', 'https://huggingface.co/rohansheth/tiro-qwen3-asr-1.7b-v2/tree/5dd5f33e0dbaa46254fc552a658d90152f1eaf75')
parts = {'attn_q':'self_attn.q_proj','attn_k':'self_attn.k_proj','attn_v':'self_attn.v_proj',
         'attn_output':'self_attn.o_proj','attn_q_norm':'self_attn.q_norm','attn_k_norm':'self_attn.k_norm',
         'attn_norm':'input_layernorm','ffn_norm':'post_attention_layernorm',
         'ffn_gate':'mlp.gate_proj','ffn_up':'mlp.up_proj','ffn_down':'mlp.down_proj'}
decoder_records = []
for item in reader.tensors:
    name = item.name
    if name in ['token_embd.weight', 'output.weight']:
        source = 'model.language_model.embed_tokens.weight'
    elif name == 'output_norm.weight':
        source = 'model.language_model.norm.weight'
    else:
        match = re.fullmatch(r'blk\.(\d+)\.(.+)\.weight', name)
        assert match and match[2] in parts, name
        source = 'model.language_model.layers.'+match[1]+'.'+parts[match[2]]+'.weight'
    value = tensor(source)
    assert tuple(value.shape) == tuple(reversed(item.shape.tolist())), (name, value.shape, item.shape)
    value = value.astype(np.float32 if value.ndim == 1 else np.float16)
    writer.add_tensor(name, value)
    decoder_records.append({'tensor': name, 'source': source, 'shape': list(value.shape), 'dtype': str(value.dtype)})
writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
all_names = set(header)-{'__metadata__'}
assert used == all_names, ('unmapped trained tensors', sorted(all_names-used))
result = {'source_revision':'5dd5f33e0dbaa46254fc552a658d90152f1eaf75',
          'source_tensor_count':len(all_names), 'mapped_source_tensor_count':len(used),
          'encoder_mappings':records, 'decoder_mappings':decoder_records,
          'declared_token_ids_checked':len(vocab), 'merges_equal':True,
          'tied_output_uses_embedding_weight':True,
          'position_table':'unchanged deterministic sinusoidal constant from existing graph',
          'onnx_checker':'both passed after topological sorting',
          'semantic_limit':'Existing encoder graph attends over the full clip; source n_window_infer=800 implies 8s inference windows. Some probes exceed 8s. Same-runtime comparison only, not HF numerical parity.',
          'note':'Model conversion; not a new P4 kernel or proved optimal format.'}
(ROOT / 'results/conversion.json').write_text(json.dumps(result,indent=2)+'\n')
print('Converted every source tensor; validated token IDs / shapes / ONNX graph.',flush=True)
