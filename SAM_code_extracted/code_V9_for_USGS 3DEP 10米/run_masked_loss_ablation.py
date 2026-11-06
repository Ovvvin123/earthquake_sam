"""Fixed-crop, supervised-only LoRA loss experiment; no source data are modified.

Run from the SAM code directory. Each update uses all eight selected training
crops in four microbatches, with the same seeded schedule in every arm.
"""
import argparse
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from fault_losses import forward_for_loss, supervised_loss, supervision
from fault_metrics import FaultMetrics
from sam2_lora import (add_adaptation_arguments, build_model, optimizer_groups,
                       training_mode, gradient_norms, load_run_model,
                       write_run_metadata, sha256_file)
from smoke_anticollapse import ARMS, batch
from tool import Generator_new


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def position(key):
    return np.array([int(v) for v in key.split('_')[-2:]], dtype=np.float64)


def select_samples(args, split, count):
    keys = (args.dataDir / f'key_{split}.lst').read_text().split()
    reader = Generator_new(str(args.dataDir/'faults.h5'), keys, training=False,
                           fixed_crop=True, supervision_key=args.supervisionKey)
    candidates, statistics = [], []
    for i, key in enumerate(keys):
        x, d = reader[i]
        reader.dataD.clear()  # A one-pass inventory does not need the full-tile cache.
        y, w, _ = supervision(torch.from_numpy(x[None]), torch.from_numpy(d[None]),
                              args, args.d0, args.dw, args.alpha)
        known = int((w > 0).sum())
        positive = int(((y > .5) & (w > 0)).sum())
        row = dict(key=key, split=split, known=known, positive=positive,
                   background=known-positive)
        statistics.append(row)
        # Exclude tiny mask slivers; keep genuinely empty known crops if present.
        if known >= 1000 and known-positive >= 256:
            candidates.append((key, x, d, row))
    if len(candidates) < count:
        raise ValueError(f'{split}: only {len(candidates)} substantial known crops')
    selected = []
    empty = [s for s in candidates if s[3]['positive'] == 0]
    if empty:
        selected.append(max(empty, key=lambda s: s[3]['known']))
    positive = [s for s in candidates if s[3]['positive'] >= 64]
    if not positive:
        raise ValueError(f'{split}: no substantial positive supervision')
    if not selected:
        selected.append(max(positive, key=lambda s: s[3]['known']))
    while len(selected) < count:
        used = {s[0] for s in selected}
        remaining = [s for s in positive if s[0] not in used]
        if not remaining:
            raise ValueError('Insufficient distinct positive crops')
        def score(s):
            distance = min(np.linalg.norm(position(s[0])-position(t[0])) for t in selected)
            return distance, s[3]['known'], s[0]
        selected.append(max(remaining, key=score))
    return [(k, x, d) for k, x, d, _ in selected], statistics


def state_hash(net):
    h = hashlib.sha256()
    for name, value in net.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def evaluate(net, items, args):
    from torchUnet import FocalTversky
    net.eval()
    meter, losses, per_image, predictions = FaultMetrics(), [], [], []
    for item in items:
        x, d = batch([item], args.device)
        y, w, valid = supervision(x, d, args, args.d0, args.dw, args.alpha)
        p, z = forward_for_loss(net, x, args)
        value = supervised_loss(y, p, w, args, FocalTversky, z)
        if not torch.isfinite(value) or not torch.isfinite(p).all():
            raise AssertionError('Nonfinite evaluation')
        meter.observe(p, y, w)
        local = FaultMetrics(); local.observe(p, y, w)
        unknown = valid & ~(w > 0)
        row = dict(key=item[0], loss=float(value), **local.report())
        row['unknown_prediction_positive_fraction'] = (
            float((p[:, :1][unknown] >= .5).float().mean()) if unknown.any() else None)
        per_image.append(row)
        losses.append(float(value))
        predictions.append(p[0, 0].cpu().numpy())
    result = dict(loss=float(np.mean(losses)), **meter.report(), per_image=per_image)
    pos = result['label_positive_fraction']
    result['all_positive_f1_baseline'] = 2*pos/(1+pos)
    result['all_negative_f1_baseline'] = 0.
    result['probability_separation'] = (
        result['positive_probability_mean']-result['background_probability_mean'])
    return result, np.stack(predictions)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataDir', type=Path, default=Path('earth_dataset_v5'))
    ap.add_argument('--outDir', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=100)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--lr0', type=float, default=1e-4)
    ap.add_argument('--tverskyWeight', type=float, default=.5)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--prepare-only', action='store_true')
    add_adaptation_arguments(ap)
    ap.set_defaults(samAdaptation='lora')
    args = ap.parse_args()
    if not 1 <= args.steps <= 1000 or args.samAdaptation != 'lora':
        ap.error('Requires LoRA and 1..1000 optimizer updates')
    if args.outDir.exists():
        ap.error('Output directory must not exist; existing results are never overwritten')
    args.supervisionKey='label_known'; args.k0Only=True; args.swMode='single'
    args.isOther=False; args.d0=.01; args.dw=.08; args.alpha=.1
    args.modelType='mars_sam'; args.accumulationVersion='window_v2'
    args.supervisionVersion='known_mask_v2'; args.collapseGuard=True
    args.consistencyLoss='probability'; args.dataset=str(args.dataDir/'key_train.lst')
    args.trainingRecipe='supervised'; args.ema=False; args.onlineCorrection=False
    args.consistencyWeight=0.; args.dropout=0.
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    train, train_stats = select_samples(args, 'train', 8)
    val, val_stats = select_samples(args, 'val', 4)
    if {k for k,_,_ in train} & {k for k,_,_ in val}:
        raise AssertionError('Identical train and validation keys')
    items = train+val
    schedule = [np.random.default_rng(args.seed+i).permutation(8).tolist()
                for i in range(args.steps)]
    args.outDir.mkdir(parents=True)
    dump(args.outDir/'schedule.json', schedule)
    dump(args.outDir/'candidate_statistics.json', train_stats+val_stats)
    overlap = []
    for a,_,_ in train:
        for b,_,_ in val:
            delta = np.abs(position(a)-position(b))
            area = int(np.maximum(0, 256-delta).prod())
            if area: overlap.append(dict(train=a, val=b, crop_overlap_pixels=area))
    selection = dict(train_keys=[s[0] for s in train], val_keys=[s[0] for s in val],
                     train_val_overlap=overlap, selection='Known >=1000, background >=256; '
                     'include an empty known crop if available, otherwise positive >=64; '
                     'greedy spatial spread on original tile grid. No test data accessed.')
    dump(args.outDir/'selection.json', selection)
    x, d = batch(items, 'cpu')
    y, w, valid = supervision(x, d, args, args.d0, args.dw, args.alpha)
    np.savez_compressed(args.outDir/'samples.npz', keys=np.array([s[0] for s in items]),
                        split=np.array(['train']*8+['val']*4), terrain=x[:,1].numpy(),
                        target=y[:,0].numpy(), known=(w[:,0]>0).numpy(), valid=valid[:,0].numpy())
    provenance = {name: sha256_file(args.dataDir/name)
                  for name in ('faults.h5','key_train.lst','key_val.lst','manifest.json')}
    report = dict(seed=args.seed, requested_updates=args.steps, microbatch_size=2,
                  accumulation=4, effective_batch_size=8, training_samples=8,
                  validation_samples=4, ema=False, pseudo_labels=False, consistency=False,
                  augmentation=False, threshold=.5, d0=args.d0, tversky_weight=args.tverskyWeight,
                  supervision='User-approved geometric coastal envelope; not verified survey completeness',
                  dataset_sha256=provenance, schedule_sha256=sha256_file(args.outDir/'schedule.json'),
                  source_sha256=sha256_file(Path(__file__)), selection=selection, arms={},
                  caveat='Small fixed-crop optimization diagnostic. Validation crops can spatially '
                         'overlap training. Unknown areas have no accuracy ground truth. Test split untouched.')
    dump(args.outDir/'report.json', report)
    for source in ('run_masked_loss_ablation.py','build_loss_ablation_gallery.py'):
        (args.outDir/source).write_bytes((Path(__file__).parent/source).read_bytes())
    print('SELECTION', json.dumps(selection), flush=True)
    if args.prepare_only: return
    from torchUnet import FocalTversky
    base_lr, base_lora_lr = args.lr0, args.loraLr
    initial_hash = initial_prediction = None
    snapshots = sorted(set([0, min(20,args.steps), min(50,args.steps), args.steps]))
    for arm, (loss_name, factor) in ARMS.items():
        start = time.perf_counter()
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        args.lossFunc=loss_name; args.lr0=base_lr*factor; args.loraLr=base_lora_lr*factor
        folder=args.outDir/arm; folder.mkdir(); args.resDir=str(folder)+'/'
        net=build_model(vars(args)).to(args.device)
        identity=state_hash(net)
        if initial_hash is None: initial_hash=identity
        assert identity == initial_hash, 'Different initialization across arms'
        frozen={n:p.detach().cpu().clone() for n,p in net.sam.named_parameters() if not p.requires_grad}
        adapters={n:p.detach().cpu().clone() for n,p in net.sam.named_parameters() if p.requires_grad}
        opt=torch.optim.AdamW(optimizer_groups(net,args))
        write_run_metadata(net,args,folder)
        trajectory=[]; saved={}

        def measure(step):
            tr, pt = evaluate(net,train,args); va, pv = evaluate(net,val,args)
            all_p=np.concatenate([pt,pv])
            row=dict(update=step,train=tr,val=va)
            trajectory.append(row)
            with (folder/'evaluation.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(row,allow_nan=False)+'\n')
            if step in snapshots: saved[f'step_{step}']=all_p
            print(f'EVAL {arm} update={step} train_F1={tr["f1"]:.4f} '
                  f'val_F1={va["f1"]:.4f} val_pred={va["prediction_positive_fraction"]:.4f} '
                  f'val_sep={va["probability_separation"]:.4f} collapse={va["collapse"]}',flush=True)
            return all_p

        initial=measure(0)
        if initial_prediction is None: initial_prediction=initial.copy()
        np.testing.assert_allclose(initial,initial_prediction,rtol=0,atol=1e-7)
        for update, order in enumerate(schedule,1):
            training_mode(net); opt.zero_grad(set_to_none=True)
            loss_total=0.
            for offset in range(0,8,2):
                xb,db=batch([train[i] for i in order[offset:offset+2]],args.device)
                target,weights,_=supervision(xb,db,args,args.d0,args.dw,args.alpha)
                assert (weights.flatten(1).sum(1)>0).all()
                p,z=forward_for_loss(net,xb,args)
                loss=supervised_loss(target,p,weights,args,FocalTversky,z)
                if not torch.isfinite(loss): raise AssertionError(f'{arm}: nonfinite loss')
                (loss/4).backward()
                loss_total+=float(loss.detach())/4
            norms=gradient_norms(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(),.1,error_if_nonfinite=True)
            opt.step(); opt.zero_grad(set_to_none=True)
            with (folder/'steps.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(dict(update=update,loss=loss_total,gradient_norms=norms),allow_nan=False)+'\n')
            if update%10 == 0 or update==args.steps: measure(update)
        for n,p in net.sam.named_parameters():
            if n in frozen:
                torch.testing.assert_close(p.detach().cpu(),frozen[n],rtol=0,atol=0)
                assert p.grad is None
        changed=sum(not torch.equal(p.detach().cpu(),adapters[n])
                    for n,p in net.sam.named_parameters() if n in adapters)
        assert changed>0, 'No adapter changes'
        np.savez_compressed(folder/'predictions.npz',**saved)
        torch.save(net.state_dict(),folder/'model.pth')
        expected=saved[f'step_{args.steps}']
        del opt,net,frozen,adapters,p,z,loss,xb,db
        gc.collect(); torch.cuda.empty_cache()
        reloaded=load_run_model(folder/'model.pth',args.device)
        _,actual=evaluate(reloaded,items,args)
        np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
        del reloaded
        gc.collect(); torch.cuda.empty_cache()
        result=dict(loss=loss_name,head_lr=args.lr0,lora_lr=args.loraLr,
                    optimizer_updates=len(schedule),initial_state_sha256=identity,
                    initial=trajectory[0],final=trajectory[-1],
                    frozen_sam_unchanged=True,changed_adapter_tensors=changed,
                    reload_predictions_match=True,elapsed_seconds=time.perf_counter()-start)
        report['arms'][arm]=result
        dump(args.outDir/'report.json',report)
        print('FINISHED',arm,'seconds',result['elapsed_seconds'],flush=True)
    from build_loss_ablation_gallery import build_gallery
    build_gallery(args.outDir)
    print('COMPLETE',args.outDir/'index.html',flush=True)


if __name__=='__main__':
    main()
