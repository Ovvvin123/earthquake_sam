"""Portable cloud configuration and preflight; only 'train --execute' launches."""
import argparse
from datetime import datetime,timezone
import gc
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from verify_vendor import verify,sha256


def absolute(value):
    path=Path(value).expanduser()
    return (ROOT/path).resolve() if not path.is_absolute() else path.resolve()


def overlaps(a,b):
    return a==b or a in b.parents or b in a.parents


def settings(env=None):
    from run_experiment import read_profile
    env=os.environ if env is None else env
    args=read_profile(ROOT/'experiments/california_v9_30m_1024_cloud.json')
    name=env.get('V9_RUN_NAME','v9_30m_1024_cloud_r01')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}',name) or '..' in name:
        raise ValueError('V9_RUN_NAME must be a plain experiment name, not a path')
    data=absolute(env.get('V9_DATA_DIR','data/Earth_dataset_v7'))
    runs=absolute(env.get('V9_RUNS_DIR','runs'))
    checks=absolute(env.get('V9_CHECKS_DIR','cloud_checks'))
    args.update(dataset=str(data/'key_train.lst'),dataFile=str(data/'faults.h5'),resDir=str(runs/name)+os.sep)
    backup=env.get('V9_BACKUP_DIR','').strip()
    args['backupDir']=str(absolute(backup)/name) if backup else ''
    outputs=[Path(args['resDir']),checks]
    if args['backupDir']:outputs.append(Path(args['backupDir']))
    for i,path in enumerate(outputs):
        if overlaps(path,data):raise ValueError('Output/check/backup paths must be separate from the dataset')
        for other in outputs[:i]:
            if overlaps(path,other):raise ValueError('Run, check and backup directories must be separate')
    return args,checks


def write_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    tmp.replace(path)


def environment_check():
    profile=os.environ.get('V9_ENV_PROFILE','baseline')
    if profile=='autodl5090':
        from environment_autodl5090 import environment_versions
        versions=environment_versions()
    elif profile=='baseline':
        if sys.platform!='linux' or sys.version_info[:2]!=(3,10):
            raise RuntimeError('The locked cloud environment requires Linux / Python 3.10')
        baseline=json.loads((ROOT/'cloud/environment_baseline.json').read_text())
        versions={}
        for name,wanted in baseline['versions'].items():
            actual=metadata.version(name)
            if actual!=wanted:raise RuntimeError(f'Environment mismatch: {name}={actual}, expected {wanted}; use cloud/setup.sh')
            versions[name]=actual
    else:
        raise RuntimeError(f'Unknown V9_ENV_PROFILE: {profile}')
    vendor=verify()
    import sam2
    package=Path(sam2.__file__).parent
    for name,wanted in vendor['runtime_files'].items():
        if not (package/name).is_file() or sha256(package/name)!=wanted:
            raise RuntimeError(f'SAM2 source differs from verified bundle: {name}')
    return dict(profile=profile,python=sys.version,packages=versions,sam2_commit=vendor['source_commit'],sam2_source_match=True)


def assets_check(args):
    manifest=json.loads((ROOT/'cloud/assets_manifest.json').read_text())
    data=Path(args['dataFile']).parent
    for category,directory in [('data',data),('weights',ROOT)]:
        for name,wanted in manifest[category].items():
            path=directory/name
            if not path.is_file():raise FileNotFoundError(f'Upload required file: {path}')
            if path.stat().st_size!=wanted['bytes'] or sha256(path)!=wanted['sha256']:
                raise RuntimeError(f'Uploaded file does not match the verified V7 experiment: {path}')
    keys={s:(data/f'key_{s}.lst').read_text().split() for s in ('train','val','test')}
    for s,items in keys.items():
        if not items or len(items)!=len(set(items)):raise ValueError(f'Empty/duplicate {s} split')
    if any(set(keys[a])&set(keys[b]) for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('Dataset split keys overlap')
    return dict(hash_verified=True,split_sizes={s:len(k) for s,k in keys.items()},resolution_m=30,tile_size=1024)


def writable_storage(parent):
    parent.mkdir(parents=True,exist_ok=True)
    # Test the rename/fsync operations used by backup without creating a run.
    with tempfile.TemporaryDirectory(prefix='.v9-write-check-',dir=parent) as folder:
        root=Path(folder);stage=root/'partial';stage.mkdir()
        with (stage/'test.bin').open('wb') as f:
            f.write(b'v9-storage-check');f.flush();os.fsync(f.fileno())
        stage.rename(root/'complete')
        assert (root/'complete/test.bin').read_bytes()==b'v9-storage-check'
    return dict(path=str(parent),free_bytes=shutil.disk_usage(parent).free,atomic_rename_test='PASS')


def gpu_check(args):
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from v7_dataset import FullTileGenerator
    from sam2_lora import build_model
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU is unavailable')
    device=torch.device(f'cuda:{args["gpuIndex"]}')
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported(including_emulation=False):raise RuntimeError('Native BF16 is unavailable')
    if not torch.backends.cuda.is_flash_attention_available():raise RuntimeError('PyTorch was built without FlashAttention')
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        q=torch.randn(1,2,256,16,device=device,dtype=torch.bfloat16)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            z=torch.nn.functional.scaled_dot_product_attention(q,q,q,dropout_p=0.)
        assert torch.isfinite(z).all()
        key=Path(args['dataset']).read_text().split()[0]
        x,_=FullTileGenerator(args['dataFile'],[key])[0]
        model=build_model(args).to(device).eval()
        result=model(torch.from_numpy(x[None]).to(device),return_logits=True)
        torch.cuda.synchronize(device)
        if tuple(result.shape)!=(1,3,1024,1024) or not torch.isfinite(result).all():
            raise RuntimeError('1024px cloud inference failed')
        report=dict(name=torch.cuda.get_device_name(device),capability=list(torch.cuda.get_device_capability(device)),
                    total_bytes=torch.cuda.get_device_properties(device).total_memory,
                    peak_inference_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    native_bf16=True,strict_flash_test='PASS',full_tile_inference='PASS',
                    backward_tested=False,optimizer_steps=0)
    del q,z,x,model,result
    gc.collect();torch.cuda.empty_cache()
    return report


def launcher_command(profile,report_dir,execute=False):
    cmd=[sys.executable,str(ROOT/'run_experiment.py'),'--sequence',str(profile),'--reportDir',str(report_dir)]
    if execute:cmd.append('--execute')
    return cmd


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('check','train'))
    parser.add_argument('--execute',action='store_true')
    cli=parser.parse_args(argv)
    if (cli.mode=='train')!=cli.execute:
        parser.error('Training requires train --execute; check never accepts --execute')
    from run_experiment import check_outputs,command,validate_parser
    from training_sources import training_source_snapshot
    args,checks=settings()
    check_outputs([args])
    if args['backupDir'] and Path(args['backupDir']).exists():
        raise FileExistsError('Backup experiment directory already exists; choose a new V9_RUN_NAME')
    directory=checks/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    directory.mkdir(parents=True,exist_ok=False)
    report=dict(status='CHECKING',training_requested=cli.execute,check_directory=str(directory),arguments=args)
    try:
        print('Environment / SAM2 verification',flush=True)
        report['environment']=environment_check()
        print('Dataset / checkpoint SHA-256 verification',flush=True)
        report['assets']=assets_check(args)
        resolved=validate_parser(command(args))
        report['resolved_config']=resolved
        report['storage']=[writable_storage(Path(args['resDir']).parent)]
        if args['backupDir']:report['storage'].append(writable_storage(Path(args['backupDir']).parent))
        print('Strict FlashAttention / full 1024px inference (no backward)',flush=True)
        report['gpu']=gpu_check(resolved)
        report['source_sha256']=training_source_snapshot(ROOT)
        profile=directory/'resolved_cloud.json'
        write_json(profile,{'arguments':args})
        report.update(status='ENVIRONMENT_AND_GPU_PASS',training_started=False)
        write_json(directory/'cloud_preflight.json',report)
        # Existing launcher does the complete train/val loader check before any
        # optional launch. In check mode its command never contains --execute.
        cmd=launcher_command(profile,directory/'loader',cli.execute)
        report['launcher_command']=cmd
        if cli.execute:
            # A launcher failure can occur before or after child startup. The run's
            # launch.json is authoritative; do not falsely report 'not started'.
            report['training_started']=None
            write_json(directory/'cloud_preflight.json',report)
        subprocess.run(cmd,cwd=ROOT,check=True)
        report.update(status='TRAINING_COMPLETED' if cli.execute else 'PREFLIGHT_PASS',
                      training_started=cli.execute)
    except Exception as error:
        report.update(status='FAILED',error=str(error))
        write_json(directory/'cloud_preflight.json',report)
        raise
    write_json(directory/'cloud_preflight.json',report)
    print('Report:',directory/'cloud_preflight.json',flush=True)
    if not cli.execute:print('CHECK ONLY: no backward, optimizer update, or formal training started.',flush=True)


if __name__=='__main__':main()
