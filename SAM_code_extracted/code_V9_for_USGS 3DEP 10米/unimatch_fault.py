"""UniMatch V2-inspired, positive-only consistency for incomplete fault labels.

Keeps SAM-2 and the original supervised branch. This is an adaptation, not a
reproduction: no DINOv2, CutMix, pseudo-background or separate unlabeled loader.
"""
import json
from pathlib import Path

import numpy as np
import torch

from training_numerics import require_finite


def consistency_weight(epoch, args):
    if not args.unimatch or epoch < args.unimatchStart:
        return 0.0
    length = args.unimatchRampEnd - args.unimatchStart + 1
    return args.unimatchWeight * min(1.0, (epoch - args.unimatchStart + 1) / length)


@torch.no_grad()
def strong_view(image, args):
    """Pointwise display perturbations; depth, masks and coordinates are exact.

    CTX (0) and hillshades (4:7) share per-image contrast/brightness and spatial
    noise, preserving channel order. Only jointly valid pixels are changed.
    The loader's existing geometric augmentation is shared by ALL views.
    """
    if image.ndim != 4 or image.shape[1] != 7:
        raise ValueError('Strong views require N x 7 x H x W input.')
    output = image.detach().clone()
    contrast, brightness, noise = (args.unimatchContrast, args.unimatchBrightness,
                                  args.unimatchNoise)
    if contrast == brightness == noise == 0:
        return output
    n, _, h, w = image.shape
    gain = 1 + (torch.rand(n, 1, 1, 1, device=image.device) * 2 - 1) * contrast
    shift = (torch.rand(n, 1, 1, 1, device=image.device) * 2 - 1) * brightness
    disturbance = torch.randn(n, 1, h, w, device=image.device) * noise
    channels = [0, 4, 5, 6]
    visual = image[:, channels]
    valid = (image[:, 2:4] > .5).all(1, keepdim=True)
    changed = (visual * gain + shift + disturbance).clamp(-1, 1).to(image.dtype)
    output[:, channels] = torch.where(valid, changed, visual)
    return output


@torch.no_grad()
def complementary_masks(batch_size, channels, device, dtype):
    """Paired masks share channel positions at every feature scale."""
    first = tuple(torch.empty(batch_size, c, 1, 1, device=device, dtype=dtype)
                  .bernoulli_(.5).mul_(2) for c in channels)
    second = tuple(2 - mask for mask in first)
    return first, second


@torch.no_grad()
def reliable_weights(original, pseudo_mask, image, pseudo_weight):
    """No pseudo negatives; reject invalid pixels even when originally positive."""
    valid = ((image[:, 2:4] > .5).all(1, keepdim=True)
             & torch.isfinite(image).all(1, keepdim=True))
    positive = (original > .5) & valid
    candidate = pseudo_mask.bool() & ~positive & valid
    weights = torch.zeros_like(original)
    weights[positive] = 1.
    weights[candidate] = float(pseudo_weight)
    return weights


def positive_consistency(prediction, weights):
    """Binary k0 positive CE; never argmax the model's three output channels.

    Divide by UNWEIGHTED selected-pixel count: candidate-only crops must retain
    their reduced weight. A weighted denominator would cancel that reduction.
    """
    p = prediction[:, :1].float()
    w = weights.detach().float()
    selected = w > 0
    # Index first, so ignored pixels have exactly zero gradient.
    p = p[selected]
    w = w[selected]
    return -(w * p.clamp_min(torch.finfo(p.dtype).eps).log()).sum() / max(p.numel(), 1)


def positive_consistency_logits(logits, weights):
    from fault_losses import binary_fault_logit
    z = binary_fault_logit(logits)
    w = weights.detach().float()
    selected = w > 0
    return (w[selected] * torch.nn.functional.softplus(-z[selected])).sum() / max(int(selected.sum()), 1)


def backward_strong_views(student, image, original, pseudo_mask, pseudo_weight,
                          args, weight, grad_divisor, output_dir, epoch, batch,
                          capture=False):
    """Two independent forward/backward calls; no optimizer/EMA updates here."""
    stats = dict(strong_loss=0., weighted_strong_loss=0., active=0,
                 reliable_pixels=0, original_pixels=0, candidate_pixels=0,
                 view_delta=0., reliable_positive_rate=0.)
    if weight <= 0:
        return stats, None
    weights = reliable_weights(original, pseudo_mask, image, pseudo_weight)
    selected = weights > 0
    stats['reliable_pixels'] = int(selected.sum())
    stats['original_pixels'] = int(((original > .5) & selected).sum())
    stats['candidate_pixels'] = stats['reliable_pixels'] - stats['original_pixels']
    if not stats['reliable_pixels']:
        return stats, None  # no meaningless forwards, no extra RNG consumption
    if grad_divisor <= 0:
        raise ValueError('Gradient accumulation divisor must be positive.')
    pair = (None, None) if args.unimatchNoDropout else complementary_masks(
        image.shape[0], student.complementary_feature_channels(),
        image.device, image.dtype)
    sample = None
    if capture:
        index = int(selected.flatten(1).any(1).nonzero()[0, 0])
        sample = dict(weak=image[index].detach().cpu().numpy(),
                      original=original[index, 0].detach().cpu().numpy(),
                      pseudo=pseudo_mask[index, 0].detach().cpu().numpy(),
                      weights=weights[index, 0].detach().cpu().numpy())
    for view_index, masks in enumerate(pair, start=1):
        strong = strong_view(image, args)
        require_finite(strong, 'strong view', output_dir,
                       epoch=epoch, batch=batch, view=view_index)
        if getattr(args, 'consistencyLoss', 'probability') == 'logits':
            logits = student(strong, feature_masks=masks, return_logits=True)
            prediction = logits.softmax(1)
            loss = positive_consistency_logits(logits, weights)
        else:
            prediction = student(strong) if masks is None else student(strong, feature_masks=masks)
            loss = positive_consistency(prediction, weights)
        require_finite(prediction, 'strong prediction', output_dir,
                       epoch=epoch, batch=batch, view=view_index)
        require_finite(loss, 'strong consistency loss', output_dir,
                       epoch=epoch, batch=batch, view=view_index)
        (loss * (weight / (2 * grad_divisor))).backward()
        stats['strong_loss'] += float(loss.detach()) / 2
        stats['view_delta'] += float((strong - image).abs().mean()) / 2
        stats['reliable_positive_rate'] += float(
            (prediction[:, :1].detach()[selected] >= .5).float().mean()) / 2
        if sample is not None:
            sample[f'strong{view_index}'] = strong[index].detach().cpu().numpy()
            sample[f'prediction{view_index}'] = prediction[index, 0].detach().cpu().numpy()
            if masks is not None:
                for level, mask in enumerate(masks):
                    sample[f'mask{view_index}_level{level}'] = mask[index].detach().cpu().numpy()
        del prediction, loss, strong
        if getattr(args, 'consistencyLoss', 'probability') == 'logits':
            del logits
    stats['weighted_strong_loss'] = stats['strong_loss'] * weight
    stats['active'] = 1
    return stats, sample


class ConsistencyDiagnostics:
    def __init__(self, epoch, weight, output_dir, capture=False):
        self.epoch, self.weight = epoch, weight
        self.directory = Path(output_dir) / 'unimatch'
        self.capture = capture
        self.rows = []
        self.sample = None

    def observe(self, base_loss, stats, sample):
        self.rows.append(dict(base_loss=float(base_loss), **stats))
        if self.sample is None and sample is not None:
            self.sample = sample

    def save(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        count = len(self.rows)
        average = lambda key: sum(row[key] for row in self.rows) / max(count, 1)
        report = dict(
            epoch=self.epoch, consistency_weight=self.weight, batches=count,
            active_batches=sum(row['active'] for row in self.rows),
            base_loss=average('base_loss'), strong_loss=average('strong_loss'),
            weighted_strong_loss=average('weighted_strong_loss'),
            total_loss=average('base_loss') + average('weighted_strong_loss'),
            reliable_pixels=sum(row['reliable_pixels'] for row in self.rows),
            original_pixels=sum(row['original_pixels'] for row in self.rows),
            candidate_pixels=sum(row['candidate_pixels'] for row in self.rows),
            view_delta=average('view_delta'),
            reliable_positive_rate=average('reliable_positive_rate'),
            pixel_counts_note='Repeated training crops; not unique geographic pixels.',
        )
        path = self.directory / f'epoch_{self.epoch:04d}.json'
        path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        if self.sample is not None:
            np.savez_compressed(self.directory / f'epoch_{self.epoch:04d}.npz', **self.sample)
            self._plot()
        print(f"UniMatch epoch={self.epoch}: lambda={self.weight:.4f}, "
              f"base={report['base_loss']:.6f}, strong={report['strong_loss']:.6f}, "
              f"total={report['total_loss']:.6f}, active={report['active_batches']}/{count}")
        return report

    def _plot(self):
        from aio2_preview import preserve_rng
        with preserve_rng():
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            s = self.sample
            panels = [
                (s['weak'][0], 'Shared geometry / weak CTX', 'gray', -1, 1),
                (s['strong1'][0], 'Strong view 1 / CTX', 'gray', -1, 1),
                (s['strong2'][0], 'Strong view 2 / CTX', 'gray', -1, 1),
                (s['original'], 'Original fault labels', 'gray', 0, 1),
                (s['pseudo'], 'Accepted temporary labels', 'magma', 0, 1),
                (s['weights'], 'Positive consistency weights', 'magma', 0, 1),
                (s['prediction1'], 'Strong student 1 / k0', 'viridis', 0, 1),
                (s['prediction2'], 'Strong student 2 / k0', 'viridis', 0, 1),
                (abs(s['prediction1'] - s['prediction2']), 'Prediction disagreement', 'magma', 0, 1),
            ]
            fig, axes = plt.subplots(3, 3, figsize=(11, 10))
            for ax, (value, title, cmap, low, high) in zip(axes.flat, panels):
                im = ax.imshow(value, cmap=cmap, vmin=low, vmax=high)
                ax.set_title(title, fontsize=10)
                ax.axis('off')
                fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
            fig.suptitle(f'UniMatch-inspired training / epoch {self.epoch} (not validation)')
            fig.tight_layout()
            fig.savefig(self.directory / f'epoch_{self.epoch:04d}.png', dpi=130)
            plt.close(fig)
