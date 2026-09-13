"""EMA teacher for the first AIO2 integration step; no label correction."""
from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import torch


class EMATeacher:
    """Independent eval-only model, updated once per student optimizer step.

    Trainable parameters and floating buffers use EMA. Frozen parameters and
    integer buffers are copied exactly. Named iterators deduplicate aliases in
    conv_sam.not_sam/in_sam; iterating state_dict would update them twice.
    """

    def __init__(self, student, decay=0.99):
        if not 0.0 <= decay < 1.0:
            raise ValueError('EMA decay must be in [0, 1).')
        self.decay = float(decay)
        self.num_updates = 0
        self.model = deepcopy(student).requires_grad_(False).eval()
        for parameter in self.model.parameters():
            parameter.grad = None

    @torch.no_grad()
    def update(self, student):
        sources = dict(student.named_parameters())
        for name, target in self.model.named_parameters():
            source = sources[name]
            if source.requires_grad:
                target.lerp_(source.detach(), 1.0 - self.decay)
            else:
                target.copy_(source.detach())
        sources = dict(student.named_buffers())
        for name, target in self.model.named_buffers():
            source = sources[name].detach()
            if target.is_floating_point() or target.is_complex():
                target.lerp_(source, 1.0 - self.decay)
            else:
                target.copy_(source)
        self.num_updates += 1
        self.model.eval()

    @torch.no_grad()
    def predict(self, inputs):
        self.model.eval()
        return self.model(inputs)

    def state_dict(self):
        return {'model': self.model.state_dict(), 'decay': self.decay,
                'num_updates': self.num_updates}

    def load_state_dict(self, state):
        decay = float(state['decay'])
        if not 0.0 <= decay < 1.0:
            raise ValueError('Invalid EMA decay in checkpoint.')
        self.model.load_state_dict(state['model'], strict=True)
        self.decay = decay
        self.num_updates = int(state['num_updates'])
        self.model.requires_grad_(False).eval()
        for parameter in self.model.parameters():
            parameter.grad = None


def _rng_state():
    np_state = np.random.get_state()
    state = {
        'python': random.getstate(), 'torch': torch.get_rng_state(),
        'numpy': (np_state[0], np_state[1].tolist(), *np_state[2:]),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        state['mps'] = torch.mps.get_rng_state()
    return state


def _restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    np_state = state['numpy']
    np.random.set_state((np_state[0], np.asarray(np_state[1], dtype=np.uint32),
                         *np_state[2:]))
    if 'cuda' in state and torch.cuda.is_available():
        if len(state['cuda']) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(state['cuda'])
    if 'mps' in state and hasattr(torch, 'mps') and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state['mps'])


def _plain_scalars(value):
    """The legacy LR warmup assigns numpy.float64 into optimizer groups."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain_scalars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_scalars(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_plain_scalars(item) for item in value)
    return value


def save_ema_checkpoint(path, student, optimizer, teacher, next_epoch, config, training_state=None):
    """Preserve pending grads: the existing trainer can carry them across epochs.

    This deliberately preserves the existing gradient accumulation schedule.
    Configuration and RNG are stored using weights_only-compatible types.
    """
    checkpoint = {
        'format': 'aio2_ema_v1', 'student': student.state_dict(),
        'optimizer': _plain_scalars(optimizer.state_dict()), 'teacher': teacher.state_dict(),
        'next_epoch': int(next_epoch), 'config': dict(config), 'rng': _rng_state(),
        'training_state': training_state,
        'gradients': {name: p.grad.detach().cpu().clone()
                      for name, p in student.named_parameters() if p.grad is not None},
    }
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_ema_checkpoint(path, student, optimizer, teacher, config):
    """Restore an EMA run, not a legacy student-only model.pth."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format') != 'aio2_ema_v1':
        raise ValueError('Use an ema_checkpoint.pth, not a student-only model.pth.')
    # Diagnostic-only options may be introduced/changed on old checkpoints.
    ignored = {'epochs', 'resDir', 'emaResume', 'emaPreview', 'previewCount',
               'previewEvery', 'previewThreshold', 'previewBuffer', 'previewMinPixels'}
    ignored.update({'emaStageTransition', 'checkpointEvery'})
    saved_config = checkpoint['config']
    transition = config.get('emaStageTransition', False)
    online_keys = {key for key in config.keys() | saved_config.keys() if key.startswith('online')}
    if transition:
        if saved_config.get('onlineCorrection', False) or not config.get('fixedVal', False):
            raise ValueError('Stage transition requires a warm-up source and fixed validation.')
        if Path(config['resDir']).resolve() == Path(saved_config['resDir']).resolve():
            raise ValueError('Stage transition requires a NEW output directory.')
        if any((Path(config['resDir']) / name).exists()
               for name in ('ema_checkpoint.pth', 'model.pth', 'best_metrics.json')):
            raise ValueError('Stage transition output already contains model results; use a new directory or normal resume.')
        if config.get('onlineCorrection', False) and checkpoint['next_epoch'] >= config['onlineStart']:
            raise ValueError('Online start is already in the past; choose a checkpoint before onlineStart.')
        ignored.update(online_keys | {'emaDecay', 'fixedVal'})
    elif not saved_config.get('onlineCorrection', False) and not config.get('onlineCorrection', False):
        ignored.update(online_keys)
    for key in (saved_config.keys() | config.keys()) - ignored:
        default = False if key in {'fixedVal', 'onlineCorrection'} else None
        if saved_config.get(key, default) != config.get(key, default):
            raise ValueError(f'Resume configuration differs for {key!r}.')
    student.load_state_dict(checkpoint['student'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    teacher.load_state_dict(checkpoint['teacher'])
    if transition:
        teacher.decay = float(config['emaDecay'])
        print(f'Stage transition: EMA decay -> {teacher.decay}; original checkpoint preserved.')
    teacher.resume_training_state = None if transition else checkpoint.get('training_state')
    if saved_config.get('resDir') != config.get('resDir'):
        teacher.resume_training_state = None  # establish best models within the new output directory

    for name, parameter in student.named_parameters():
        grad = checkpoint['gradients'].get(name)
        parameter.grad = None if grad is None else grad.to(parameter).clone()
    _restore_rng(checkpoint['rng'])
    return int(checkpoint['next_epoch'])
