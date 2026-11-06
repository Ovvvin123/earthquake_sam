"""Authorized one-shot recovery: preserve A, run a fresh B from original e500.

Never changes the failed queue's records or results. No automatic retries.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from run_experiment import execute_sequence, verify_finished
from run_v5_feedback_ablation import HERE, SOURCE, CONTROL as ORIGINAL_CONTROL, preflight, paired_audit
from training_sources import training_source_snapshot, verify_training_sources

CONTROL=HERE/'experiment_controls/v5_feedback_recovery_20261028'
B_OUTPUT=HERE/'runs/v5_e500_to600_ema_only_20261028_recovery1'


def prepare():
    import torch
    from sam2_lora import sha256_file, checkpoint_identity
    original_reports=json.loads((ORIGINAL_CONTROL/'production/preflight.json').read_text())
    a_folder=HERE/'runs/v5_e500_to600_keep_feedback_20261028'
    a_launch=json.loads((a_folder/'launch.json').read_text())
    old_b=HERE/'runs/v5_e500_to600_ema_only_20261028'
    b_launch=json.loads((old_b/'launch.json').read_text())
    a_args=a_launch['resolved_config'];b_args=dict(b_launch['resolved_config'])
    if a_launch['status']!='complete':raise RuntimeError('A is not verified complete')
    if b_launch.get('pid') or set(p.name for p in old_b.iterdir())!={'launch.json','resolved_config.json'}:
        raise RuntimeError('Original B has unexpected training artifacts; inspect before recovery')
    b_args['resDir']=str(B_OUTPUT)+os.sep
    if Path(b_args['emaResume']).resolve()!=(SOURCE/'ema_checkpoint.pth').resolve():
        raise RuntimeError('B must restore original epoch500, never A epoch600')
    if b_args.get('emaFeedbackFork')!='disable' or b_args.get('onlineCorrection') or b_args.get('unimatch'):
        raise RuntimeError('Wrong B feedback configuration')
    snapshot=training_source_snapshot(HERE)
    expected=original_reports[0]['source_sha256']
    differences=sorted(k for k,v in snapshot.items() if expected.get(k)!=v)
    if differences:raise RuntimeError('Training dependencies differ from A: '+', '.join(differences))
    if sha256_file(SOURCE/'ema_checkpoint.pth')!=original_reports[0]['source_sha256_checkpoint']:
        raise RuntimeError('Original epoch500 checkpoint changed')
    verify_finished(a_args,original_reports[0]['identity'])
    rows=[json.loads(s) for s in (a_folder/'adaptation_history.jsonl').read_text().splitlines() if s.strip()]
    if [r['epoch'] for r in rows]!=list(range(501,601)) or rows[-1]['global_step']!=6000:
        raise RuntimeError('A history is incomplete')
    state=torch.load(a_folder/'ema_checkpoint.pth',map_location='cpu',weights_only=True)
    if state['teacher']['num_updates']!=6000:raise RuntimeError('A EMA update count mismatch')
    del state
    if checkpoint_identity(b_args)!=original_reports[0]['identity']:
        raise RuntimeError('Data/base identity differs from A')
    protected=[SOURCE/'ema_checkpoint.pth']
    protected += [a_folder/n for n in ('ema_checkpoint.pth','adaptation_history.jsonl','fixed_val.jsonl',
                                     'launch.json','resolved_config.json','known_crop_inventory.json')]
    protected += list(old_b.iterdir())
    preservation={str(p):sha256_file(p) for p in protected}
    reports=preflight([b_args])
    return [a_args,b_args],reports,dict(protected_sha256=preservation,
        unchanged_training_dependencies=list(snapshot),source_scope='trainV10_local_imports_v1',
        original_a=str(a_folder),preserved_failed_b=str(old_b),new_b=str(B_OUTPUT),
        export_source_sha256=sha256_file(HERE/'export_v5_feedback_comparison.py'))


def verify_preserved(report):
    from sam2_lora import sha256_file
    changed=[p for p,h in report['protected_sha256'].items() if sha256_file(p)!=h]
    if changed:raise RuntimeError('Protected original artifacts changed: '+', '.join(changed))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    cli=parser.parse_args()
    if (CONTROL/'launch_once.json').exists():raise FileExistsError('Recovery already attempted; never auto-retry')
    if (HERE/'v5_feedback_comparison_20261028').exists():raise FileExistsError('Comparison output exists')
    profiles,reports,audit=prepare()
    print('PREFLIGHT PASS: A verified e600; original e500 unchanged; training dependencies match A',flush=True)
    print('NEW B:',profiles[1]['resDir'],flush=True)
    if not cli.execute:return
    CONTROL.mkdir(parents=True,exist_ok=True)
    with (CONTROL/'launch_once.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(),profiles=profiles,created_utc=datetime.now(timezone.utc).isoformat()),stream,indent=2)
    with (CONTROL/'preflight.json').open('x') as stream:json.dump(dict(reports=reports,audit=audit),stream,indent=2)
    state=dict(status='checking',pid=os.getpid(),outputs=[p['resDir'] for p in profiles])
    def status(**updates):
        state.update(updates,updated_utc=datetime.now(timezone.utc).isoformat())
        tmp=CONTROL/'status.json.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(CONTROL/'status.json')
    try:
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try:argv=(proc/'cmdline').read_bytes().decode().split('\0')
            except (OSError,UnicodeError):continue
            if any(Path(a).name=='trainV10.py' for a in argv[:3]):raise RuntimeError(f'Another training process exists: {proc.name}')
        status(status='training_b')
        execute_sequence([profiles[1]],reports)
        verify_preserved(audit)
        paired=paired_audit(profiles,100)
        with (CONTROL/'paired_audit.json').open('x') as stream:json.dump(paired,stream,indent=2)
        verify_training_sources(HERE,reports[0]['source_sha256'])
        from sam2_lora import sha256_file
        if sha256_file(HERE/'export_v5_feedback_comparison.py')!=audit['export_source_sha256']:
            raise RuntimeError('Gallery exporter changed; review before export')
        status(status='exporting')
        subprocess.run([sys.executable,'-u',str(HERE/'export_v5_feedback_comparison.py'),
                        '--b-run',profiles[1]['resDir']],cwd=HERE,check=True)
        verify_preserved(audit)
        gallery=HERE/'v5_feedback_comparison_20261028'
        manifest=json.loads((gallery/'manifest.json').read_text())
        if manifest['status']!='complete' or manifest['verified_panels']!=216 or not (gallery/'index.html').is_file():
            raise RuntimeError('Incomplete comparison gallery')
        status(status='complete',exit_code=0,gallery=str(gallery/'index.html'),original_results_unchanged=True)
    except BaseException as error:
        status(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise


if __name__=='__main__':main()
