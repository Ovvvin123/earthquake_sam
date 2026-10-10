"""Step-budget trainer for the isolated non-overlapping-grid experiment."""
import hashlib
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
from grid_core import local_path
from grid_data import GridDataset, target_weights
from grid_validation import validate
from step_schedule import online_weight, consistency_weight, lr_factor
from aio2_ema import EMATeacher, _rng_state, _restore_rng
from aio2_online import temporary_targets
from unimatch_fault import backward_strong_views
from training_numerics import require_finite, clip_finite_gradients


def read_splits(data):
    keys = {}
    for split in ('train', 'val'):
        path = data/f'key_{split}.lst'
        keys[split] = path.read_text(encoding='utf-8').split()
        if not keys[split] or len(keys[split]) != len(set(keys[split])):
            raise ValueError(f'Empty/duplicate {split} keys')
    if set(keys['train']) & set(keys['val']):
        raise ValueError('Train/val overlap')
    import h5py
    with h5py.File(data/'faults.h5', 'r') as f:
        if set(f) != set(keys['train']) | set(keys['val']):
            raise ValueError('Train/val must cover all saved tiles')
    digest = hashlib.sha256()
    for name in ('faults.h5', 'manifest.json', 'key_train.lst', 'key_val.lst'):
        with (data/name).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                digest.update(chunk)
    return keys, digest.hexdigest()


def save_atomic(value, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def train(args, model_factory=None):
    if args.modelType != 'mars_sam' or not args.k0Only or args.lossFunc != 'FocalTversky' or args.isalign:
        raise ValueError('Grid experiment requires mars_sam, k0Only, FocalTversky, no isalign')
    if args.datasetAdd or args.isOther or args.emaResume or args.emaStageTransition:
        raise ValueError('Use plain k0 experiment and --resume grid_checkpoint.pth; legacy extra-data/stage options unsupported')
    if int(args.mn) != 256:
        raise ValueError('SAM2 configuration expects mn=256')
    data = local_path(args.dataset).parent
    out = local_path(args.resDir)
    if Path(args.dataset).name != 'key_train.lst':
        raise ValueError('--dataset must name key_train.lst')
    keys, signature = read_splits(data)
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume and any((out/n).exists() for n in ('grid_checkpoint.pth', 'model.pth', 'history.jsonl')):
        raise FileExistsError('Use a new run directory or explicit --resume')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    from torchUnet import conv_sam, FocalTversky
    model = (model_factory() if model_factory else conv_sam(32, dropout=args.dropout, isvit=False)).to(device)
    teacher = EMATeacher(model, args.emaDecay) if args.emaTeacher else None
    optimizer = torch.optim.AdamW([
        {'params': model.not_sam.parameters(), 'lr': args.lr0},
        {'params': model.in_sam.parameters(), 'lr': args.lr0/8}])
    training = GridDataset(data/'faults.h5', keys['train'], training=True)
    validation = GridDataset(data/'faults.h5', keys['val'], full=True)
    config = dict(vars(args))
    step, epoch, cursor, order = 0, 1, 0, []
    best, teacher_best = float('inf'), float('inf')
    if args.resume:
        ck = torch.load(local_path(args.resume), map_location='cpu', weights_only=False)
        if ck.get('format') != 'v5_grid_steps_v1' or ck['data_sha256'] != signature:
            raise ValueError('Checkpoint format or dataset fingerprint mismatch')
        ignored = {'maxSteps', 'epochs', 'resume', 'device', 'resDir', 'evalEverySteps',
                   'previewEvery', 'emaPreview', 'checkpointEvery'}
        mismatches = [k for k, v in config.items() if k not in ignored and ck['config'].get(k) != v]
        if mismatches:
            raise ValueError(f'Resume configuration changed: {mismatches}')
        model.load_state_dict(ck['student']); optimizer.load_state_dict(ck['optimizer'])
        if teacher: teacher.load_state_dict(ck['teacher'])
        step, epoch, cursor, order = ck['global_step'], ck['epoch'], ck['cursor'], ck['order']
        best, teacher_best = ck['best_loss'], ck['teacher_best_loss']
        if out != Path(ck['output_dir']):
            best = teacher_best = float('inf')
        _restore_rng(ck['rng'])
    (out/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    (out/'data_signature.json').write_text(json.dumps(dict(sha256=signature, keys=keys), indent=2), encoding='utf-8')
    print(f'Train={len(training)}, val={len(validation)}, updates/epoch={math.ceil(len(training)/(args.batchsize*args.stepPerMiniBatch))}, '
          f'steps={step}..{args.maxSteps}, device={device}', flush=True)
    while step < args.maxSteps:
        if cursor >= len(order):
            if order: epoch += 1
            order, cursor = torch.randperm(len(training)).tolist(), 0
        stop = min(cursor+args.batchsize*args.stepPerMiniBatch, len(order))
        selected = order[cursor:stop]
        optimizer.zero_grad(set_to_none=True)
        for idx, group in enumerate(optimizer.param_groups):
            group['lr'] = args.lr0*lr_factor(step, args.warmupSteps)/(8 if idx else 1)
        ow, uw = online_weight(step, args), consistency_weight(step, args)
        pu = args.pu*min(1., max(0., (step-args.puWarmupSteps+1)/args.puRampSteps))
        group_loss, strong_loss, candidates = 0., 0., 0
        model.train(); model.in_sam.eval()
        for start in range(0, len(selected), args.batchsize):
            ids = selected[start:start+args.batchsize]
            pairs = [training[k] for k in ids]
            image = torch.from_numpy(np.stack([p[0] for p in pairs])).to(device)
            distances = torch.from_numpy(np.stack([p[1] for p in pairs])).to(device)
            require_finite(image, 'training input', out, step=step)
            target, sw = target_weights(image, distances, args.d0, args.dw, args.alpha)
            corrected, weights = target, sw
            mask = torch.zeros_like(target, dtype=torch.bool)
            if ow > 0:
                probabilities = teacher.predict(image)
                require_finite(probabilities, 'teacher prediction', out, step=step)
                corrected, weights, mask, stats = temporary_targets(target, sw, probabilities, image, ow, args)
                candidates += stats['candidate_pixels']
                del probabilities
            prediction = model(image)
            require_finite(prediction, 'student prediction', out, step=step)
            loss = FocalTversky(corrected, prediction, sw=weights, pu=pu)
            require_finite(loss, 'training loss', out, step=step)
            fraction = len(ids)/len(selected)
            (loss*fraction).backward()
            group_loss += float(loss.detach())*fraction
            del prediction, loss
            if uw > 0:
                stats, _ = backward_strong_views(model, image, target, mask, ow, args, uw,
                    1/fraction, out, epoch, start)
                strong_loss += stats['weighted_strong_loss']*fraction
        clip_finite_gradients(model, .1, out, step=step)
        optimizer.step()
        if teacher: teacher.update(model)
        optimizer.zero_grad(set_to_none=True)
        step += 1; cursor = stop
        row = dict(global_step=step, epoch=epoch, samples=len(selected), train_loss=group_loss,
            strong_loss=strong_loss, online_weight=ow, consistency_weight=uw,
            pseudo_pixels=candidates, lr=optimizer.param_groups[0]['lr'])
        if step % args.evalEverySteps == 0 or step == args.maxSteps:
            # Validation must not perturb the next training sample/augmentation RNG sequence.
            from aio2_preview import preserve_rng
            with preserve_rng():
                row['validation'] = validate(model, validation, device, args, FocalTversky,
                    out/'validation_preview.png' if args.emaPreview else None)
                if teacher:
                    row['teacher_validation'] = validate(teacher.model, validation, device, args, FocalTversky)
            if row['validation']['loss'] < best:
                best = row['validation']['loss']; save_atomic(model.state_dict(), out/'model.pth')
                (out/'best_metrics.json').write_text(json.dumps(dict(global_step=step, **row['validation']), indent=2))
            if teacher and row['teacher_validation']['loss'] < teacher_best:
                teacher_best = row['teacher_validation']['loss']
                (out/'teacher').mkdir(exist_ok=True)
                save_atomic(teacher.model.state_dict(), out/'teacher/model.pth')
                (out/'teacher/best_metrics.json').write_text(json.dumps(dict(global_step=step, **row['teacher_validation']), indent=2))
            save_atomic(dict(format='v5_grid_steps_v1', student=model.state_dict(), optimizer=optimizer.state_dict(),
                teacher=teacher.state_dict() if teacher else None, global_step=step, epoch=epoch,
                cursor=cursor, order=order, rng=_rng_state(), config=config, data_sha256=signature,
                best_loss=best, teacher_best_loss=teacher_best, output_dir=str(out)), out/'grid_checkpoint.pth')
            print(json.dumps(row), flush=True)
        with (out/'history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row)+'\n')
    return dict(global_step=step, best_loss=best)
