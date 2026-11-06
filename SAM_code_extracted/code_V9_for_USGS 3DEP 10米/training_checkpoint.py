"""Full supervised continuation for the legacy epoch trainer (including pending grads)."""
from pathlib import Path
import torch
from aio2_ema import _rng_state, _restore_rng, _plain_scalars
from sam2_lora import normalized_config, checkpoint_identity, check_checkpoint_identity


def save_supervised_checkpoint(path, student, optimizer, next_epoch, global_step,
                               config, training_state=None):
    state = dict(format='sam_supervised_v1', student=student.state_dict(),
                 optimizer=_plain_scalars(optimizer.state_dict()),
                 next_epoch=int(next_epoch), global_step=int(global_step),
                 config=normalized_config(config), rng=_rng_state(),
                 training_state=training_state, identity=checkpoint_identity(config),
                 gradients={n: p.grad.detach().cpu().clone() for n, p in student.named_parameters()
                            if p.grad is not None})
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def load_supervised_checkpoint(path, student, optimizer, config):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('format') != 'sam_supervised_v1':
        raise ValueError('Use a supervised training_checkpoint.pth')
    current, saved = normalized_config(config), normalized_config(state['config'])
    ignored = {'epochs', 'resDir', 'resume', 'checkpointEvery'}
    ignored.update({'backupDir', 'backupEvery', 'backupMilestones'})
    changed = [k for k in current.keys() | saved.keys()
               if k not in ignored and current.get(k) != saved.get(k)]
    if changed:
        raise ValueError(f'Resume configuration changed: {changed}')
    check_checkpoint_identity(state, config)
    student.load_state_dict(state['student'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    for name, parameter in student.named_parameters():
        grad = state['gradients'].get(name)
        parameter.grad = None if grad is None else grad.to(parameter).clone()
    _restore_rng(state['rng'])
    best = state.get('training_state') if config.get('resDir') == saved.get('resDir') else None
    return int(state['next_epoch']), int(state['global_step']), best
