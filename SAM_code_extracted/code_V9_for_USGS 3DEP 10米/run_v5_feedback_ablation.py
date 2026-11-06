"""Fork the completed V5 epoch-500 checkpoint, run A then B, then export comparisons.

No automatic retries. --smoke runs one epoch per arm in separate smoke outputs.
Default is preflight only. --execute is required for training.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from run_experiment import (HERE, profile_arguments, command, validate_parser,
                            inspect_data, check_outputs, execute_sequence)
from training_sources import training_source_snapshot

SOURCE = HERE/'runs/california_ds5_bceft_lora8_ema_unimatch_seed7_r01'
CONTROL = HERE/'experiment_controls/v5_feedback_ablation_20261028'
NAMES = {'keep': 'v5_e500_to600_keep_feedback_20261028',
         'disable': 'v5_e500_to600_ema_only_20261028'}


def configurations(smoke=False):
    result = []
    for mode, name in NAMES.items():
        args = profile_arguments(HERE/'experiments/california_ds5.json')
        args.update(epochs=501 if smoke else 600,
                    resDir=str(HERE/'runs'/(name + ('_smoke' if smoke else '')))+os.sep,
                    emaResume=str(SOURCE/'ema_checkpoint.pth'), emaFeedbackFork=mode)
        if mode == 'disable':
            args.pop('onlineCorrection'); args.pop('unimatch')
        result.append(args)
    return result


def paired_audit(profiles, expected_epochs):
    rows = [[json.loads(s) for s in (Path(p['resDir'])/'adaptation_history.jsonl').read_text().splitlines() if s.strip()] for p in profiles]
    expected = list(range(501, 501+expected_epochs))
    for group in rows:
        if [r['epoch'] for r in group] != expected:
            raise RuntimeError('Not exactly the requested continuation epochs')
    for a, b in zip(*rows):
        if not a.get('input_sha256') or a['input_sha256'] != b.get('input_sha256'):
            raise RuntimeError(f'Paired input mismatch at epoch {a["epoch"]}')
        if a['global_step'] != b['global_step'] or a['learning_rates'] != b['learning_rates']:
            raise RuntimeError('Paired update count / learning rate mismatch')
    baselines = [json.loads((Path(p['resDir'])/'fixed_val.jsonl').read_text().splitlines()[0]) for p in profiles]
    if baselines[0] != baselines[1] or baselines[0]['epoch'] != 500:
        raise RuntimeError('Restored baseline predictions do not match')
    b_dir = Path(profiles[1]['resDir'])
    if (b_dir/'online').exists() or (b_dir/'unimatch').exists():
        raise RuntimeError('EMA-only arm unexpectedly generated feedback diagnostics')
    return dict(epochs=expected, identical_inputs=True, identical_baseline=True,
                identical_learning_rates=True, updates_per_arm=expected_epochs*10,
                final_metrics={Path(p['resDir']).name:r[-1] for p,r in zip(profiles,rows)})


def preflight(profiles):
    import torch
    from sam2_lora import normalized_config, checkpoint_identity, sha256_file
    from aio2_ema import feedback_fork_overrides
    check_outputs(profiles)
    checkpoint = SOURCE/'ema_checkpoint.pth'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['next_epoch'] != 500 or saved['config']['epochs'] != 500 or saved['teacher']['num_updates'] != 5000:
        raise ValueError('Expected completed V5 epoch 500 / 5000 EMA updates')
    if saved['gradients']:
        raise ValueError('Unexpected unfinished gradient window')
    source_sha = sha256_file(checkpoint)
    reports = []
    for args in profiles:
        resolved = validate_parser(command(args))
        old, new = normalized_config(saved['config']), normalized_config(resolved)
        allowed = feedback_fork_overrides(old,new) | {'epochs','resDir','emaResume'}
        differences = {k:[old.get(k),new.get(k)] for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}
        illegal = set(differences)-allowed
        if illegal: raise ValueError(f'Unexpected config changes: {illegal}')
        identity = checkpoint_identity(args)
        if identity != saved['identity']: raise ValueError('Dataset / SAM base identity changed')
        data = inspect_data(args) if not reports else reports[0]['data']
        if (data['train']['selected_samples'],data['val']['selected_samples'],data['expected_updates_per_epoch']) != (80,10,10):
            raise ValueError('V5 sample/update inventory changed')
        report = dict(command=command(args),resolved_config=resolved,data=data,identity=identity,
                      source_checkpoint=str(checkpoint),source_sha256_checkpoint=source_sha,
                      config_changes=differences,
                      source_scope='trainV10_local_imports_v1',
                      source_sha256=training_source_snapshot(HERE))
        reports.append(report)
    del saved
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    cli = parser.parse_args()
    profiles = configurations(cli.smoke)
    reports = preflight(profiles)
    control = CONTROL/('smoke' if cli.smoke else 'production')
    control.mkdir(parents=True,exist_ok=True)
    (control/'preflight.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    print('PREFLIGHT PASS: same epoch 500 source; 80 train / 10 val; 10 updates per epoch',flush=True)
    if not cli.execute: return
    # Exclusive persistent guard remains even after failure.
    with (control/'launch_once.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(),created_utc=datetime.now(timezone.utc).isoformat(),profiles=profiles),stream,indent=2)
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try: argv=(proc/'cmdline').read_bytes().decode().split('\0')
        except (OSError, UnicodeError): continue
        if any(Path(a).name=='trainV10.py' for a in argv[:3]):
            raise RuntimeError(f'Another training process exists: {proc.name}')
    state = dict(status='training',pid=os.getpid(),outputs=[p['resDir'] for p in profiles])
    def status(**updates):
        state.update(updates,updated_utc=datetime.now(timezone.utc).isoformat())
        tmp=control/'status.json.tmp'; tmp.write_text(json.dumps(state,indent=2)); tmp.replace(control/'status.json')
    status()
    try:
        execute_sequence(profiles,reports)
        audit = paired_audit(profiles,1 if cli.smoke else 100)
        from sam2_lora import sha256_file
        if sha256_file(SOURCE/'ema_checkpoint.pth') != reports[0]['source_sha256_checkpoint']:
            raise RuntimeError('Source checkpoint was modified')
        (control/'paired_audit.json').write_text(json.dumps(audit,indent=2))
        if not cli.smoke:
            status(status='exporting')
            subprocess.run([sys.executable,'-u',str(HERE/'export_v5_feedback_comparison.py')],cwd=HERE,check=True)
        status(status='complete',exit_code=0)
    except BaseException as error:
        status(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise


if __name__=='__main__':
    main()
