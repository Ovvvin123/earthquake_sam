"""Short real-data SAM-2 LoRA verification; never starts a full training run."""
import argparse
import json
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np
import torch
from aio2_ema import EMATeacher
from aio2_online import temporary_targets
from sam2_lora import (add_adaptation_arguments, configure_sam, optimizer_groups,
                       training_mode, gradient_norms, write_run_metadata, load_run_model)
from training_numerics import clip_finite_gradients
from unimatch_fault import backward_strong_views


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_adaptation_arguments(parser)
    parser.set_defaults(samAdaptation='lora')
    parser.add_argument('--dataDir', type=Path, required=True)
    parser.add_argument('--outDir', type=Path, required=True)
    parser.add_argument('--grid', action='store_true')
    parser.add_argument('--d0', type=float, default=.01)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--lr0', type=float, default=1e-4)
    args = parser.parse_args()
    if args.samAdaptation != 'lora':
        parser.error('This smoke verifies LoRA; use training commands for other modes')
    if args.outDir.exists() and any(args.outDir.iterdir()):
        parser.error('Choose an empty smoke output directory')
    args.outDir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(2026); np.random.seed(2026); random.seed(2026)
    from torchUnet import conv_sam, FocalTversky
    keys = (args.dataDir/'key_train.lst').read_text().split()
    if args.grid:
        sys.path.insert(0, str(Path(__file__).parent/'V5_different_split'))
        from grid_data import GridDataset, target_weights
        dataset = GridDataset(args.dataDir/'faults.h5', keys, training=False)
    else:
        from tool import Generator_new
        dataset = Generator_new(str(args.dataDir/'faults.h5'), keys,
                                n=256, m=256, training=False, fixed_crop=True)
    # Greece/NZ: one positive crop per region. California: positive + background if available.
    buckets = {}
    for i, key in enumerate(keys):
        x, y = dataset[i]
        positive = bool(((y[0] < args.d0) & (x[2:4].min(0) > .5)).any())
        bucket = key.split('__')[0] if '__' in key else ('positive' if positive else 'background')
        if '__' in key and not positive:
            continue
        buckets.setdefault(bucket, (key, x, y))
        if len(buckets) >= 2:
            break
    if not buckets:
        raise ValueError('No usable training crops')
    selected = list(buckets.values())
    image = torch.from_numpy(np.stack([a[1] for a in selected])).to(args.device)
    distance = torch.from_numpy(np.stack([a[2] for a in selected])).to(args.device)
    if args.grid:
        original, weights = target_weights(image, distance, args.d0, .08, .1)
    else:
        distance = distance[:, :1]
        original = (distance < args.d0).float()
        weights = torch.exp(-distance.square()/.08**2)*.9+.1
        weights = torch.where(image[:,2:4].amin(1,keepdim=True)<1e-9,1.,weights)
        weights = weights/weights.mean()
    student = conv_sam(32, dropout=0, isvit=False).to(args.device).eval()
    with torch.no_grad():
        initial = student(image).clone()
    configure_sam(student, vars(args))
    with torch.no_grad():
        torch.testing.assert_close(initial, student(image), rtol=0, atol=0)
    report = write_run_metadata(student, args, args.outDir)
    frozen = {n:p.detach().cpu().clone() for n,p in student.sam.named_parameters() if not p.requires_grad}
    adapter_before = {n:p.detach().cpu().clone() for n,p in student.sam.named_parameters() if p.requires_grad}
    input_before = student.input_sam[0].weight.detach().clone()
    teacher = EMATeacher(student, .98)
    optimizer = torch.optim.AdamW(optimizer_groups(student, args))
    params = argparse.Namespace(onlineSupport=.5, onlineThreshold=.9, onlineBuffer=3,
        onlineMinPixels=16, onlineMaxFraction=.05, unimatchNoDropout=False,
        unimatchContrast=.2, unimatchBrightness=.1, unimatchNoise=.03)
    if image.is_cuda:
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    started = time.perf_counter()
    steps = []
    for step in range(2):
        training_mode(student); optimizer.zero_grad(set_to_none=True)
        target, sw, mask, candidates = temporary_targets(original, weights,
            teacher.predict(image), image, .2, params)
        prediction = student(image)
        loss = FocalTversky(target, prediction, sw=sw, pu=0.)
        if not torch.isfinite(loss): raise AssertionError('Nonfinite loss')
        loss.backward()
        del prediction
        stats, _ = backward_strong_views(student, image, original, mask, .2,
            params, .1, 1, args.outDir, step+1, 0)
        if not stats['active']: raise AssertionError('Strong views inactive')
        norms = gradient_norms(optimizer)
        if not all(np.isfinite(v) and v > 0 for v in norms.values()):
            raise AssertionError(f'Missing/invalid gradients: {norms}')
        if student.input_sam[0].weight.grad is None:
            raise AssertionError('Input adapter gradient was cut')
        clip_finite_gradients(student, .1, args.outDir, step=step)
        optimizer.step(); teacher.update(student)
        steps.append(dict(loss=float(loss.detach()), gradient_norms=norms,
                          candidates=candidates, strong=stats))
    for name, p in student.sam.named_parameters():
        if name in frozen:
            torch.testing.assert_close(frozen[name], p.cpu(), rtol=0, atol=0)
            if p.grad is not None: raise AssertionError('Frozen base has gradient')
    changed = [n for n,p in student.sam.named_parameters()
               if n in adapter_before and not torch.equal(adapter_before[n],p.detach().cpu())]
    if not any('.lora_A' in n for n in changed) or not any('.lora_B' in n for n in changed):
        raise AssertionError('Adapters did not update')
    if torch.equal(input_before, student.input_sam[0].weight):
        raise AssertionError('Input adapter did not update')
    student.eval()
    with torch.no_grad(): reference = student(image)
    with tempfile.TemporaryDirectory(dir=args.outDir) as tmp:
        path = Path(tmp)/'model.pth'
        torch.save(student.state_dict(), path)
        loaded = load_run_model(path, args.device, vars(args))
        with torch.no_grad(): torch.testing.assert_close(reference,loaded(image),rtol=1e-5,atol=1e-6)
        del loaded
    if image.is_cuda: torch.cuda.synchronize()
    report.update(keys=[a[0] for a in selected], steps=steps, ema_updates=teacher.num_updates,
        zero_increment_exact=True, frozen_base_unchanged=True, input_adapter_updated=True,
        changed_adapter_tensors=len(changed), strict_reload_passed=True,
        elapsed_seconds=time.perf_counter()-started,
        peak_cuda_gib=torch.cuda.max_memory_allocated()/1024**3 if image.is_cuda else None,
        purpose='Two real-data updates including dual strong views; not an accuracy experiment')
    (args.outDir/'smoke_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
