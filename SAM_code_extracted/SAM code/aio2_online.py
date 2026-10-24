"""Fixed-trigger online correction experiment; not a full ACT/O2C reproduction."""
import json
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation, label

from aio2_preview import preserve_rng
from training_numerics import require_finite
from fault_losses import forward_for_loss, supervised_loss, supervision
from fault_metrics import FaultMetrics


def pseudo_weight(epoch, args):
    """Epochs are one-based. Hold 41..50; linear ramp 51..80; plateau thereafter."""
    if epoch < args.onlineStart:
        return 0.
    if epoch <= args.onlineHoldEnd:
        return args.onlineWeightStart
    progress = min(1., (epoch - args.onlineHoldEnd) / (args.onlineRampEnd - args.onlineHoldEnd))
    return args.onlineWeightStart + progress * (args.onlineWeightMax - args.onlineWeightStart)


def select_candidates(probability, original, valid, args):
    """Reject WHOLE support components touching labels before any subtraction.

    Lower support threshold detects connections around high-confidence seeds.
    Reject area overflow, never truncate a blob to meet the area budget.
    """
    support = np.isfinite(probability) & valid & (probability >= args.onlineSupport)
    components, count = label(support, structure=np.ones((3, 3), dtype=bool))
    exclusion = binary_dilation(original, iterations=args.onlineBuffer) if args.onlineBuffer else original
    chosen = np.zeros_like(original, dtype=bool)
    limit = int(valid.sum() * args.onlineMaxFraction)
    stats = dict(components=int(count), rejected_touching=0, rejected_size=0,
                 rejected_confidence=0, rejected_budget=0, accepted_components=0)
    if getattr(args, 'collapseGuard', False) and valid.any() and support.sum() / valid.sum() > .99:
        stats['rejected_budget'] = int(count)
        return chosen, stats
    for component in range(1, count + 1):
        region = components == component
        if (region & exclusion).any():
            stats['rejected_touching'] += 1
        elif not (region & (probability >= args.onlineThreshold)).any():
            stats['rejected_confidence'] += 1
        elif region.sum() < args.onlineMinPixels or region.sum() > limit:
            stats['rejected_size'] += 1
        else:
            chosen |= region
            stats['accepted_components'] += 1
    if chosen.sum() > limit:
        stats['rejected_budget'] = stats['accepted_components']
        stats['accepted_components'] = 0
        chosen[:] = False
    return chosen, stats


@torch.no_grad()
def temporary_targets(original, sw, probabilities, image, weight, args):
    """Candidates replace their background term within the SAME weighted loss.

    Candidate weight = lambda * sample maximum spatial weight: avoid attenuating
    distant pseudo positives again with the original annotation distance map.
    Original positives/other weights are unchanged. No candidates is identity.
    """
    mask = torch.zeros_like(original, dtype=torch.bool)
    totals = dict(components=0, rejected_touching=0, rejected_size=0,
                  rejected_confidence=0, rejected_budget=0, accepted_components=0)
    if weight > 0:
        for index in range(len(original)):
            valid = ((image[index, 2:4] > .5).all(dim=0) &
                     torch.isfinite(image[index]).all(dim=0))
            if getattr(args, 'supervisionKey', ''):
                # Reviewed labels cannot be replaced by teacher predictions.
                valid = valid & (sw[index, 0] <= 0)
            selected, stats = select_candidates(
                probabilities[index, 0].detach().cpu().numpy(),
                original[index, 0].detach().cpu().numpy() > .5,
                valid.cpu().numpy(), args)
            mask[index, 0] = torch.from_numpy(selected).to(mask.device)
            for key, value in stats.items():
                totals[key] += value
    totals['candidate_pixels'] = int(mask.sum())
    totals['samples_with_candidates'] = int(mask.flatten(1).any(1).sum())
    if not mask.any():
        return original, sw, mask, totals
    corrected = torch.where(mask, torch.ones_like(original), original)
    weights = torch.where(mask, weight * sw.amax(dim=(-2, -1), keepdim=True).clamp_min(1.), sw)
    return corrected, weights, mask, totals


class OnlineDiagnostics:
    def __init__(self, epoch, weight, output_dir, capture=False):
        self.epoch, self.weight = epoch, weight
        self.output_dir, self.capture = Path(output_dir), capture
        self.rows, self.examples = [], []

    def observe(self, batch, original, mask, probabilities, inputs, stats, loss):
        self.rows.append(dict(batch=batch, loss=float(loss), **stats))
        if self.capture and len(self.examples) < 2:
            indices = mask.flatten(1).any(1).nonzero().flatten().tolist()
            if not self.examples and not indices:
                indices = [0]
            for index in indices[:2 - len(self.examples)]:
                self.examples.append(dict(
                    terrain=inputs[index, 0].detach().cpu().numpy().copy(),
                    original_label=original[index, 0].detach().cpu().numpy().copy(),
                    teacher_probability=probabilities[index, 0].detach().cpu().numpy().copy(),
                    temporary_label=(original[index, 0].bool() | mask[index, 0]).cpu().numpy(),
                    accepted_candidates=mask[index, 0].cpu().numpy(), batch=batch))

    def write(self):
        path = self.output_dir / 'online' / f'epoch_{self.epoch:04d}'
        path.mkdir(parents=True, exist_ok=True)
        keys = ('components', 'rejected_touching', 'rejected_size', 'rejected_confidence',
                'rejected_budget', 'accepted_components', 'candidate_pixels', 'samples_with_candidates')
        totals = {key: sum(row[key] for row in self.rows) for key in keys}
        report = dict(epoch=self.epoch, pseudo_weight=self.weight, totals=totals, batches=self.rows,
                      note='Counts include repeated augmented crops, not unique faults. Labels are temporary.')
        (path / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        if self.examples:
            import matplotlib.pyplot as plt
            with preserve_rng():
                for index, example in enumerate(self.examples):
                    np.savez_compressed(path / f'batch_sample_{index:02d}.npz', **example)
                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    terrain = example['terrain']
                    overlay = np.zeros((*terrain.shape, 4))
                    overlay[example['original_label'] > .5] = (0, 1, 0, .8)
                    axes[0].imshow(terrain, cmap='gray')
                    axes[0].imshow(overlay)
                    axes[0].set_title('Original labels (green)')
                    axes[1].imshow(example['teacher_probability'], vmin=0, vmax=1, cmap='magma')
                    axes[1].set_title('Teacher probability')
                    overlay[example['accepted_candidates']] = (1, 0, 1, .8)
                    axes[2].imshow(terrain, cmap='gray')
                    axes[2].imshow(overlay)
                    axes[2].set_title('Used temporary positives (magenta)')
                    for axis in axes:
                        axis.axis('off')
                    fig.suptitle(f'Epoch {self.epoch}, batch {example["batch"]}, weight={self.weight:.3f}')
                    fig.tight_layout()
                    fig.savefig(path / f'batch_sample_{index:02d}.png', dpi=120)
                    plt.close(fig)
        print(f'Online epoch={self.epoch}, weight={self.weight:.4f}, '
              f'candidate_pixels={totals["candidate_pixels"]}, '
              f'rejected_touching={totals["rejected_touching"]}')


@torch.no_grad()
def fixed_validation(student, teacher, loader, device, args, loss_func, d0, dw, alpha, epoch=None, return_metrics=False):
    """Per-sample fixed-crop mean, original labels, valid pixels only, PU=0."""
    modes = [(module, module.training) for module in student.modules()]
    student_losses, teacher_losses = [], []
    meters = {'student': FaultMetrics(), 'teacher': FaultMetrics()}
    baselines = {'all_negative': [], 'all_positive': []}
    unknown_samples = 0
    keys = list(getattr(getattr(loader, 'dataset', None), 'list', []))
    sample_number = 0
    output_dir = getattr(args, 'resDir', '.')
    try:
        with preserve_rng():
            student.eval()
            for images, distances in loader:
                images = torch.as_tensor(images, device=device)
                distances = torch.as_tensor(distances, device=device)
                for index in range(len(images)):
                    image = images[index:index+1]
                    context = dict(epoch=epoch, validation_sample=sample_number,
                                   key=keys[sample_number] if sample_number < len(keys) else None)
                    sample_number += 1
                    require_finite(image, 'validation input', output_dir, **context)
                    require_finite(distances[index:index+1], 'validation distances', output_dir, **context)
                    target, sw, valid = supervision(image, distances[index:index+1], args, d0, dw, alpha)
                    if not valid.any():
                        raise ValueError('Fixed validation crop has no valid pixels; revise validation windows.')
                    if not (sw > 0).any():
                        unknown_samples += 1
                        continue
                    require_finite(sw, 'validation weights', output_dir, **context)
                    prediction, logits = forward_for_loss(student, image, args)
                    require_finite(prediction, 'student validation prediction', output_dir, **context)
                    value = supervised_loss(target, prediction, sw, args, loss_func, logits)
                    require_finite(value, 'student validation loss', output_dir, **context)
                    student_losses.append(float(value))
                    meters['student'].observe(prediction, target, sw)
                    for name, probability in [('all_negative', 0.), ('all_positive', 1.)]:
                        constant = torch.full_like(prediction, probability)
                        # Finite saturated logits for BCE; exact constants for legacy probability losses.
                        constant_logits = torch.zeros_like(prediction)
                        constant_logits[:, :1] = (20. if probability else -20.) + math.log(prediction.shape[1]-1)
                        baselines[name].append(float(supervised_loss(
                            target, constant, sw, args, loss_func, constant_logits)))
                    if teacher is not None:
                        prediction, logits = forward_for_loss(teacher, image, args, teacher=True)
                        require_finite(prediction, 'teacher validation prediction', output_dir, **context)
                        value = supervised_loss(target, prediction, sw, args, loss_func, logits)
                        require_finite(value, 'teacher validation loss', output_dir, **context)
                        teacher_losses.append(float(value))
                        meters['teacher'].observe(prediction, target, sw)
    finally:
        for module, mode in modes:
            module.training = mode
    if not student_losses or not all(math.isfinite(v) for v in student_losses + teacher_losses):
        raise ValueError('Empty or non-finite fixed validation results.')
    health = dict(student=meters['student'].report(),
                  teacher=meters['teacher'].report() if teacher is not None else None,
                  baseline_losses={k: sum(v)/len(v) for k, v in baselines.items()},
                  baseline_note='BCE constants use binary logits +/-20; probability losses use exact 0/1',
                  unknown_samples=unknown_samples,
                  supervision='mask' if getattr(args, 'supervisionKey', '') else 'catalog_background_assumption')
    if return_metrics:
        return student_losses, teacher_losses, health
    return student_losses, teacher_losses


def validation_manifest(loader, output_dir):
    """Fingerprint the exact fixed inputs and labels, guarding against data drift."""
    digest = hashlib.sha256()
    count = 0
    with preserve_rng():
        for images, targets in loader:
            images = torch.as_tensor(images).detach().cpu().numpy()
            targets = torch.as_tensor(targets).detach().cpu().numpy()
            for image, target in zip(images, targets):
                for array in (image, target):
                    digest.update(str((array.shape, array.dtype.str)).encode())
                    digest.update(np.ascontiguousarray(array).tobytes())
                count += 1
    protocol = ('fixed_max_known_per_sample_loss_v1' if
                getattr(getattr(loader,'dataset',None),'known_crop',False) else
                'fixed_center_per_sample_loss_v1')
    report = dict(protocol=protocol, samples=count,
                  sha256=digest.hexdigest(),
                  keys=list(getattr(getattr(loader, 'dataset', None), 'list', [])))
    path = Path(output_dir) / 'validation_manifest.json'
    if path.exists():
        previous = json.loads(path.read_text(encoding='utf-8'))
        if previous != report:
            raise ValueError('Fixed validation samples changed in this output directory.')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report['sha256']


class BestCheckpoints:
    """Selection state travels in full checkpoints, independent of output dir."""
    def __init__(self, output_dir, config, state=None, signature=None):
        self.output_dir, self.config = Path(output_dir), dict(config)
        self.state = state or {'student': None, 'teacher': None,
                               'metric': ('fixed_max_known_per_sample_loss_v1'
                                          if config.get('knownCrop') else 'fixed_center_per_sample_loss_v1')}
        if signature is not None:
            if self.state.get('validation_signature', signature) != signature:
                raise ValueError('Fixed validation data differs from the resumed checkpoint.')
            self.state['validation_signature'] = signature

    def update(self, epoch, student_loss, teacher_loss, student, optimizer, teacher, baseline=False, health=None):
        from aio2_ema import save_ema_checkpoint
        improved = []
        for name, value in {'student': student_loss, 'teacher': teacher_loss}.items():
            if value is None:
                continue
            if not math.isfinite(value):
                raise ValueError('Non-finite validation loss; refusing best checkpoint selection.')
            if self.config.get('collapseGuard', False):
                if health is None or health.get(name) is None:
                    raise ValueError('Collapse guard requires validation health metrics')
                if not health[name]['eligible']:
                    continue
            previous = self.state[name]
            if previous is None or value < previous['loss']:
                self.state[name] = dict(epoch=epoch, loss=value)
                improved.append(name)
        for name in improved:
            directory = self.output_dir / f'best_{name}'
            directory.mkdir(parents=True, exist_ok=True)
            weights = student.state_dict() if name == 'student' else teacher.model.state_dict()
            temporary = directory / 'model.pth.tmp'
            torch.save(weights, temporary)
            temporary.replace(directory / 'model.pth')
            if teacher is not None:
                save_ema_checkpoint(directory / 'ema_checkpoint.pth', student, optimizer, teacher,
                                    epoch, self.config, training_state=self.state)
            (directory / 'metrics.json').write_text(json.dumps(self.state[name], indent=2), encoding='utf-8')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'best_metrics.json').write_text(json.dumps(self.state, indent=2), encoding='utf-8')
        with (self.output_dir / 'fixed_val.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(epoch=epoch, baseline=baseline, student_loss=student_loss,
                                         teacher_loss=teacher_loss, health=health)) + '\n')
        return self.state
