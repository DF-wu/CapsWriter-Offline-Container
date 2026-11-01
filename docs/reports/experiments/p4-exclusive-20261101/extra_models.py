"""Download pinned decoder-only Q6/Q8/IQ4 candidates, validating whole files."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

STATE = Path('/tmp/capswriter-p4-exclusive-state.json')
MODELS = [
    {'variant': 'q6', 'repo': 'mradermacher/Qwen3-ASR-1.7B-GGUF',
     'revision': '664b960e358348943aefcdc26b3cf09eff1fb0ac', 'filename': 'Qwen3-ASR-1.7B.Q6_K.gguf',
     'bytes': 1673003488, 'sha256': '534b653ee0074264c80533df9637e9eb556c04457f30cd1ba08fe48dcb608263'},
    {'variant': 'q8', 'repo': 'ggml-org/Qwen3-ASR-1.7B-GGUF',
     'revision': '36a678687ba7d07a74ca70ccb0e36902e005fb80', 'filename': 'Qwen3-ASR-1.7B-Q8_0.gguf',
     'bytes': 2165034944, 'sha256': '58e22d0532d4eacaf034cfac17a6fed159f37c41390c710186783be439d1fc57'},
    {'variant': 'iq4', 'repo': 'mradermacher/Qwen3-ASR-1.7B-GGUF',
     'revision': '664b960e358348943aefcdc26b3cf09eff1fb0ac', 'filename': 'Qwen3-ASR-1.7B.IQ4_XS.gguf',
     'bytes': 1181583840, 'sha256': '33bb28421a3bf976eff8f65b5871946e88c0bcad479a46be9539b8fae521f51a'}]


def main():
    state = json.loads(STATE.read_text());root = Path(state['scratch'])

    def fetch(model):
        url = 'https://huggingface.co/'+model['repo']+'/resolve/'+model['revision']+'/'+model['filename']
        relative = Path(model['repo'].replace('/', '--')) / model['revision'] / model['filename']
        target = root / 'downloads' / relative
        target.parent.mkdir(parents=True,exist_ok=True)
        # Some publishers use the exact same basename for different model
        # contracts. Namespace by source+revision; never overwrite another.
        legacy = root / 'downloads' / model['filename']
        if not target.exists() and legacy.exists() and legacy.stat().st_size == model['bytes']:
            check=hashlib.sha256()
            with legacy.open('rb') as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b''):check.update(block)
            if check.hexdigest()==model['sha256']:legacy.rename(target)
        if not target.exists() or target.stat().st_size != model['bytes']:
            with urllib.request.urlopen(url, timeout=120) as response, target.open('wb') as stream:
                shutil.copyfileobj(response, stream, 1024*1024)
        h = hashlib.sha256()
        with target.open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''):h.update(block)
        assert target.stat().st_size == model['bytes'] and h.hexdigest() == model['sha256'], model['variant']
        directory = root / 'models' / model['variant'];directory.mkdir(exist_ok=True)
        for part in ['frontend', 'backend']:
            name = 'qwen3_asr_encoder_'+part+'.onnx'
            (directory / name).unlink(missing_ok=True)
            (directory / name).symlink_to('/models-original/Qwen3-ASR/Qwen3-ASR-1.7B/'+name)
        (directory / 'qwen3_asr_llm.gguf').unlink(missing_ok=True)
        (directory / 'qwen3_asr_llm.gguf').symlink_to('/bench/downloads/'+str(relative))
        print('VERIFIED',model['variant'],model['sha256'],flush=True)
        return dict(model, url=url,download_relative_path=str(relative))

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        result = list(pool.map(fetch, MODELS))
    (root / 'results/extra-models.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
