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
    def predict(self, inputs, return_logits=False):
        self.model.eval()
        if return_logits:
            return self.model(inputs, return_logits=True)
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


def activate_ema(student, teacher, epoch, args, global_step=0):
    """Create once at the one-based boundary; never allocate a warm-up teacher."""
    if teacher is None and args.emaTeacher and epoch >= getattr(args, 'emaStartEpoch', 1):
        teacher = EMATeacher(student, args.emaDecay)
        teacher.global_step = int(global_step)
    return teacher


def restore_ema_run(path, student, optimizer, config):
    """Restore both pre-activation and active checkpoints without an early teacher."""
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('format') != 'aio2_ema_v1':
        raise ValueError('Use an ema_checkpoint.pth')
    active = state.get('teacher') is not None
    global_step = int(state.get('global_step', state['teacher']['num_updates'] if active else 0))
    best = state.get('training_state') if state['config'].get('resDir') == config.get('resDir') else None
    del state
    teacher = EMATeacher(student, config['emaDecay']) if active else None
    epoch = load_ema_checkpoint(path, student, optimizer, teacher, config)
    if teacher is not None:
        best = teacher.resume_training_state
    return teacher, epoch, global_step, best


def save_ema_checkpoint(path, student, optimizer, teacher, next_epoch, config, training_state=None, global_step=None):
    """Preserve pending grads: the existing trainer can carry them across epochs.

    This deliberately preserves the existing gradient accumulation schedule.
    Configuration and RNG are stored using weights_only-compatible types.
    """
    from sam2_lora import checkpoint_identity
    if teacher is None and not (config.get('emaTeacher') and next_epoch < config.get('emaStartEpoch', 1)):
        raise ValueError('Missing teacher outside delayed EMA warm-up')
    checkpoint = {
        'identity': checkpoint_identity(config),
        'format': 'aio2_ema_v1', 'student': student.state_dict(),
        'optimizer': _plain_scalars(optimizer.state_dict()), 'teacher': teacher.state_dict() if teacher is not None else None,
        'global_step': int(global_step if global_step is not None else getattr(teacher, 'global_step', getattr(teacher, 'num_updates', 0))),
        'next_epoch': int(next_epoch), 'config': dict(config), 'rng': _rng_state(),
        'training_state': training_state,
        'gradients': {name: p.grad.detach().cpu().clone()
                      for name, p in student.named_parameters() if p.grad is not None},
    }
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def feedback_fork_overrides(saved_config, config):
    """Allow only the declared feedback ablation, never arbitrary resume drift."""
    mode = config.get('emaFeedbackFork', '')
    saved_mode = saved_config.get('emaFeedbackFork', '')
    if not mode or mode == saved_mode:
        return set()
    if mode not in ('keep', 'disable') or saved_mode:
        raise ValueError('Cannot change an existing feedback fork mode')
    if config.get('emaStageTransition') or not config.get('emaTeacher') or not config.get('fixedVal'):
        raise ValueError('Feedback fork requires EMA and fixed validation without stage transition')
    if not all(saved_config.get(k) for k in ('emaTeacher', 'onlineCorrection', 'unimatch')):
        raise ValueError('Feedback fork source must have both feedback branches enabled')
    output = Path(config['resDir']).resolve()
    source = Path(saved_config['resDir']).resolve()
    if output == source or output in source.parents or source in output.parents:
        raise ValueError('Feedback fork requires a separate NEW output directory')
    if any((output / n).exists() for n in ('ema_checkpoint.pth', 'model.pth', 'best_metrics.json', 'adaptation_history.jsonl')):
        raise ValueError('Feedback fork output already contains results')
    enabled = mode == 'keep'
    if any(bool(config.get(k, False)) != enabled for k in ('onlineCorrection', 'unimatch')):
        raise ValueError('Feedback fork must keep or disable both branches')
    return {'emaFeedbackFork', 'onlineCorrection', 'unimatch'}


def load_ema_checkpoint(path, student, optimizer, teacher, config):
    """Restore an EMA run, not a legacy student-only model.pth."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format') != 'aio2_ema_v1':
        raise ValueError('Use an ema_checkpoint.pth, not a student-only model.pth.')
    # Diagnostic-only options may be introduced/changed on old checkpoints.
    ignored = {'epochs', 'resDir', 'emaResume', 'emaPreview', 'previewCount',
               'previewEvery', 'previewThreshold', 'previewBuffer', 'previewMinPixels'}
    ignored.update({'emaStageTransition', 'checkpointEvery'})
    ignored.update({'backupDir', 'backupEvery', 'backupMilestones'})
    from sam2_lora import normalized_config
    saved_config = normalized_config(checkpoint['config'])
    config = normalized_config(config)
    ignored.update(feedback_fork_overrides(saved_config, config))
    for key, value in {'seed': 0, 'resume': ''}.items():
        saved_config.setdefault(key, value)
        config.setdefault(key, value)
    transition = config.get('emaStageTransition', False)
    online_keys = {key for key in config.keys() | saved_config.keys() if key.startswith('online')}
    unimatch_keys = {key for key in config.keys() | saved_config.keys()
                     if key.startswith('unimatch')}
    if not config.get('unimatch', False) and not saved_config.get('unimatch', False):
        ignored.update(unimatch_keys)  # old checkpoints remain compatible
    if transition:
        if saved_config.get('unimatch', False):
            raise ValueError('Stage transition requires a source without UniMatch training.')
        ignored.update(unimatch_keys)
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
    from sam2_lora import check_checkpoint_identity
    check_checkpoint_identity(checkpoint, config)
    active = checkpoint.get('teacher') is not None
    if active != (teacher is not None):
        raise ValueError('Teacher activation state does not match checkpoint')
    if not active and not (config.get('emaTeacher') and checkpoint['next_epoch'] < config.get('emaStartEpoch', 1)):
        raise ValueError('Missing teacher outside delayed EMA warm-up')
    student.load_state_dict(checkpoint['student'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    if teacher is not None:
        teacher.load_state_dict(checkpoint['teacher'])
        teacher.global_step = int(checkpoint.get('global_step', teacher.num_updates))
    if transition:
        if teacher is None:
            raise ValueError('Stage transition requires an active teacher checkpoint')
        teacher.decay = float(config['emaDecay'])
        print(f'Stage transition: EMA decay -> {teacher.decay}; original checkpoint preserved.')
    if teacher is not None:
        teacher.resume_training_state = None if transition else checkpoint.get('training_state')
        if saved_config.get('resDir') != config.get('resDir'):
            teacher.resume_training_state = None  # establish best models within the new output directory

    for name, parameter in student.named_parameters():
        grad = checkpoint['gradients'].get(name)
        parameter.grad = None if grad is None else grad.to(parameter).clone()
    _restore_rng(checkpoint['rng'])
    return int(checkpoint['next_epoch'])
