"""One real SAM-2 batch with both strong branches; does not start long training."""
import argparse
import ast
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from aio2_ema import EMATeacher
from aio2_online import temporary_targets
from training_numerics import require_finite, clip_finite_gradients
from unimatch_fault import backward_strong_views, ConsistencyDiagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outDir', default='train_v5_unimatch_smoke/')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batchsize', type=int, default=2)
    parser.add_argument('--dataDir', type=Path, default=Path('earth_dataset_v3'))
    parser.add_argument('--d0', type=float, default=.02, help='k0 distance threshold')
    parser.add_argument('--checkpoint', default='', help='optional EMA checkpoint; load model weights only for this smoke test')
    args = parser.parse_args()
    if not 0 < args.d0 < .35:
        parser.error('Invalid d0.')
    out = Path(args.outDir)
    if out.exists() and any(out.iterdir()):
        parser.error('Smoke output must be empty; choose a new --outDir.')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    np.random.seed(2026)
    random.seed(2026)
    from tool import Generator_new
    from torchUnet import conv_sam, FocalTversky
    import torchUnet

    keys = (args.dataDir / 'key_train.lst').read_text().split()
    dataset = Generator_new(str(args.dataDir / 'faults.h5'), keys,
                            n=256, m=256, training=False, fixed_crop=True)
    items, selected_keys = [], []
    for index, key in enumerate(keys):
        image, distances = dataset[index]
        valid = (image[2:4] > .5).all(axis=0)
        if ((distances[0] < args.d0) & valid).any():
            items.append((image, distances))
            selected_keys.append(key)
        if len(items) == args.batchsize:
            break
    if len(items) != args.batchsize:
        raise ValueError('Not enough training crops with positive labels for smoke test.')
    image = torch.from_numpy(np.stack([item[0] for item in items])).float().to(args.device)
    distance = torch.from_numpy(np.stack([item[1][:1] for item in items])).float().to(args.device)
    original = (distance < args.d0).float()
    saved_image, saved_original = image.clone(), original.clone()
    student = conv_sam(32, dropout=0, isvit=False, isalign=False).to(args.device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        student.load_state_dict(state['student'])
        del state
    student.train()
    student.in_sam.eval()
    # Compare default forward with the exact pre-change source, when available.
    backups = sorted(Path('.').glob('_backup_before_unimatch_*/torchUnet.py'))
    legacy_equal = None
    if backups:
        tree = ast.parse(backups[0].read_text(encoding='utf-8'))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'conv_sam')
        forward = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'forward')
        scope = dict(vars(torchUnet))
        exec(compile(ast.Module(body=[forward], type_ignores=[]), str(backups[0]), 'exec'), scope)
        with torch.no_grad():
            legacy_equal = torch.equal(student(image), scope['forward'](student, image))
        if not legacy_equal:
            raise AssertionError('Normal forward differs from pre-change model.')
    teacher = EMATeacher(student, decay=.98)
    opt = torch.optim.AdamW([
        {'params': student.not_sam.parameters(), 'lr': 1e-4},
        {'params': student.in_sam.parameters(), 'lr': 1e-4 / 8},
    ])
    before = student.tail[-2].weight.detach().clone()
    parameters = argparse.Namespace(
        onlineSupport=.5, onlineThreshold=.9, onlineBuffer=3, onlineMinPixels=16,
        onlineMaxFraction=.05, unimatchNoDropout=False, unimatchContrast=.2,
        unimatchBrightness=.1, unimatchNoise=.03)
    if image.is_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    teacher_p = teacher.predict(image)
    sw = torch.exp(-distance.square() / .08**2) * .9 + .1
    sw = torch.where(image[:, 2:4].amin(1, keepdim=True) < 1e-9, 1., sw)
    sw = sw / sw.mean()
    target, weights, pseudo, candidates = temporary_targets(
        original, sw, teacher_p, image, .2, parameters)
    prediction = student(image)
    base_loss = FocalTversky(target, prediction, sw=weights, pu=0.)
    require_finite(base_loss, 'smoke base loss', out)
    (base_loss / 4).backward()
    stats, sample = backward_strong_views(student, image, original, pseudo, .2,
        parameters, .1, 4, out, 1, 0, capture=True)
    if not stats['active']:
        raise AssertionError('Smoke did not exercise strong branches.')
    clip_finite_gradients(student, .1, out, epoch=1, batch=0)
    opt.step()
    teacher.update(student)
    opt.zero_grad()
    assert teacher.num_updates == 1
    assert all(p.grad is None and not p.requires_grad for p in teacher.model.parameters())
    assert torch.equal(image, saved_image) and torch.equal(original, saved_original)
    assert not torch.equal(before, student.tail[-2].weight)
    student.eval()
    with torch.no_grad():
        require_finite(student(image), 'post-update inference', out)
        require_finite(teacher.predict(image), 'post-update teacher', out)
    if image.is_cuda:
        torch.cuda.synchronize()
    logger = ConsistencyDiagnostics(1, .1, out, capture=True)
    logger.observe(float(base_loss.detach()), stats, sample)
    report = logger.save()
    report.update(
        device=args.device, batchsize=args.batchsize, keys=selected_keys,
        data_dir=str(args.dataDir.resolve()), d0=args.d0,
        legacy_forward_exact=legacy_equal, ema_updates=teacher.num_updates,
        elapsed_seconds=time.perf_counter() - started,
        peak_cuda_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3 if image.is_cuda else None,
        candidate_filter=candidates, purpose='One-batch code smoke test; not an accuracy experiment.')
    (out / 'smoke_report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
