"""Shared fault supervision. Zero weights mean unknown/ignored, never background."""
import math
import torch
import torch.nn.functional as F

LOGIT_LOSSES = {'BCEWithLogits', 'BCEFocalTversky'}


def binary_fault_logit(logits):
    if logits.ndim != 4 or logits.shape[1] < 2:
        raise ValueError('Expected B,C,H,W logits with C >= 2')
    return logits[:, :1].float() - torch.logsumexp(logits[:, 1:].float(), 1, keepdim=True)


def fault_loss(logits, target, weights, name, tversky_weight=0.5):
    if name not in LOGIT_LOSSES:
        raise ValueError(f'Unsupported logits loss: {name}')
    if not math.isfinite(tversky_weight) or tversky_weight < 0:
        raise ValueError('tversky_weight must be finite and nonnegative')
    z = binary_fault_logit(logits)
    target = target.float()
    weights = torch.broadcast_to(torch.as_tensor(weights, device=z.device).float(), target.shape)
    if target.shape != z.shape:
        raise ValueError('Logits losses require a single k0 target channel')
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Supervision weights must be finite and nonnegative')
    selected = weights > 0
    if not selected.any():
        # A wholly unknown crop has no supervised gradient. Do not invent negatives.
        return z[selected].sum()
    if not torch.isfinite(target[selected]).all() or ((target[selected] < 0) | (target[selected] > 1)).any():
        raise ValueError('Supervised targets must be finite probabilities')
    bce = (F.binary_cross_entropy_with_logits(z[selected], target[selected], reduction='none')
           * weights[selected]).sum() / weights[selected].sum()
    if name == 'BCEWithLogits' or tversky_weight == 0:
        return bce
    # Lazy import keeps loss-only tests independent of SAM model construction.
    from torchUnet import FocalTversky
    ft = FocalTversky(target[selected].reshape(1, 1, -1),
                     z[selected].sigmoid().reshape(1, 1, -1),
                     sw=weights[selected].reshape(1, 1, -1), pu=0.)
    return bce + tversky_weight * ft


def forward_for_loss(model, images, args, teacher=False):
    call = model.predict if teacher else model
    if args.lossFunc in LOGIT_LOSSES:
        logits = call(images, return_logits=True)
        return logits.softmax(1), logits
    return call(images), None


def supervised_loss(target, probabilities, weights, args, legacy_loss=None, logits=None, pu=0.):
    if args.lossFunc in LOGIT_LOSSES:
        return fault_loss(logits, target, weights, args.lossFunc, getattr(args, 'tverskyWeight', .5))
    if not (weights > 0).any():
        return probabilities.sum() * 0.
    kwargs = dict(sw=weights)
    if args.lossFunc == 'FocalTversky':
        kwargs['pu'] = pu
    return legacy_loss(target, probabilities, **kwargs)


def supervision(images, distances, args, d0, dw, alpha):
    """Optional fourth distance channel is a known-label mask (1 known, 0 unknown)."""
    target = distances[:, :1] if args.k0Only else distances[:, :2]
    nearest = target if args.swMode == 'single' else target.amin(1, keepdim=True)
    weights = torch.exp(-nearest.square() / dw**2) * (1-alpha) + alpha
    if args.isOther:
        other = torch.exp(-distances[:, 2:3].square() / dw**2) * (1-alpha) + alpha
        weights = torch.maximum(weights, other)
    valid = (images[:, 2:4] > .5).all(1, keepdim=True) & torch.isfinite(images).all(1, keepdim=True)
    known = valid
    if getattr(args, 'supervisionKey', ''):
        if distances.shape[1] != 4:
            raise ValueError('supervisionKey requires a fourth known-label mask channel')
        known = known & (distances[:, 3:4] > .5)
    target = ((target < d0) & known).float()
    weights = torch.where(known, weights, 0.)
    weights = weights / weights.mean().clamp_min(1e-12)
    return target, weights, valid


def accumulation_window(batch, count, size, version='window_v2'):
    if size < 1 or count < 1 or not 0 <= batch < count:
        raise ValueError('Invalid accumulation window')
    if version == 'legacy_v1':
        return size, batch % size == 0
    if version != 'window_v2':
        raise ValueError('Unknown accumulation version')
    start = batch // size * size
    return min(size, count-start), (batch+1) % size == 0 or batch+1 == count
