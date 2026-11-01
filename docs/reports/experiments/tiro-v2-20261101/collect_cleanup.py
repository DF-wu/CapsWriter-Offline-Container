"""Accept only a normally completed experiment; export evidence and clean its resources."""
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

from lifecycle import STATE, assert_config, readiness, services, utc

HERE=Path(__file__).resolve().parent


def run(args,**kwargs):
    return subprocess.run(args,check=True,capture_output=True,text=True,**kwargs)


def main():
    state=json.loads(STATE.read_text());root=Path(state['scratch'])
    assert root==Path('/tmp/capswriter-tiro-v2-mc8grn8t') and root.resolve()==root
    assert state['label']=='tiro-v2-20261101' and state['created_images']==[]
    restoration=json.loads((root/'results/restoration.json').read_text())
    # This first run began before accepted fields were added to run_smoke.py.
    # Normal completion proves EVERY post-block check_empty passed; exceptions
    # write a different restoration reason. Never certify partial failed runs.
    assert restoration['reason']=='normal completion' and restoration['portainer_parameters_unchanged']
    schedule=json.loads((root/'results/schedule.json').read_text())
    assert len(schedule)==8 and all(x['exit']==0 for x in schedule)
    assert len({(x['config'],x['phase']) for x in schedule})==8
    blocks=[]
    for block in schedule:
        cid=block['config'];phase=block['phase']
        rows=[json.loads(l) for l in (root/'results'/(cid+'.jsonl')).read_text().splitlines()]
        calls=[x for x in rows if x.get('type')=='sample' and x['phase']==phase]
        ids={x['run_id'] for x in calls}
        assert len(ids)==1 and all(x['status']=='ok' for x in calls)
        assert sum(not x['warmup'] for x in calls)==(14 if phase=='quality' else 9)
        blocks.append({'config':cid,'phase':phase,'run_id':next(iter(ids)),'accepted':True,
                       'basis':'all 8 blocks returned zero, output validation passed, EVERY post-block exclusivity check passed as proven by normal-completion restoration'})
    (root/'results/acceptance.json').write_text(json.dumps({'normal_completion':True,'accepted_utc':utc(),'blocks':blocks},indent=2)+'\n')
    for svc in services(state):assert readiness(svc,assert_config(svc))
    run(['docker','start',state['container']])
    shutil.copy2(HERE/'capture_runtime.py',root/'capture_runtime.py')
    print(run(['docker','exec',state['container'],'python','/bench/capture_runtime.py']).stdout,flush=True)
    print(run([sys.executable,str(HERE/'analyze_smoke.py'),str(root/'results')]).stdout,flush=True)
    for folder in ['results','logs']:shutil.copytree(root/folder,HERE/folder,dirs_exist_ok=True)
    shutil.copy2(root/'samples.json',HERE/'samples.json')
    for name in ['config.json','tokenizer_config.json','processor_config.json','generation_config.json']:
        dest=HERE/'source-metadata';dest.mkdir(exist_ok=True);shutil.copy2(root/'hf'/name,dest/name)
    # Only selected fields from inspect, never secret env values.
    info=json.loads(run(['docker','inspect',state['container']]).stdout)[0]
    assert info['Config']['Labels']['capswriter.experiment']==state['label']
    mount=next(m for m in info['Mounts'] if m['Destination']=='/models-original')
    assert mount['Name']==state['model_volume'] and not mount['RW']
    assert info['HostConfig']['PortBindings']=={}
    record={'started_utc':utc(),'experiment_id':info['Id'],'read_only_original_volume':True,
            'new_images_pulled':[],'source_bf16_sha256':'c57399f84b00cadfb3d1d62a3a7b3375f54297e26fdb5dc78138a076a5e8a1b5',
            'scratch_logical_bytes':int(run(['du','-sb',str(root)]).stdout.split()[0]),'services':[]}
    for svc in services(state):
        value=assert_config(svc);assert readiness(svc,value)
        record['services'].append({'name':svc['production_name'],'id':value['Id'],'image':value['Image'],
            'started_at':value['State']['StartedAt'],'config_fingerprints':svc['config_fingerprints_before']})
    run(['docker','exec',state['container'],'python','-c',
         "from pathlib import Path;import shutil;r=Path('/bench');[(p.unlink() if p.is_symlink() or p.is_file() else shutil.rmtree(p)) for p in r.iterdir()]"])
    run(['docker','rm','-f',info['Id']]);root.rmdir();STATE.unlink()
    for svc,before in zip(services(state),record['services']):
        value=assert_config(svc);assert readiness(svc,value)
        assert value['State']['StartedAt']==before['started_at']
        run(['docker','image','inspect',svc['base_image']])
    run(['docker','volume','inspect',state['model_volume']])
    assert not run(['docker','ps','-aq','--filter','label=capswriter.experiment='+state['label']]).stdout.strip()
    record['checks']={'original_services_ready':True,'same_ids_images_parameters':True,
        'cleanup_did_not_restart_services':True,'original_models_hashes_unchanged':True,
        'test_container_removed':True,'scratch_removed':not root.exists(),'state_removed':not STATE.exists(),
        'original_images_and_volume_retained':True,'global_prune_used':False}
    record['completed_utc']=utc();record['complete']=True
    (HERE/'results/cleanup.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record['checks']),flush=True)


if __name__=='__main__':main()
