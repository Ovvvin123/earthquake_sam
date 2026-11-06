"""Stateless epoch-based LR schedules; legacy behavior remains the default."""
import math


DEFAULTS = dict(lrSchedule='legacy', lrWarmupEpochs=10, lrHoldEnd=100,
                lrDecayEnd=500, lrMinFactor=0.125)


def schedule_config(config=None):
    source = {} if config is None else (config if isinstance(config, dict) else vars(config))
    cfg = {k: source.get(k, v) for k, v in DEFAULTS.items()}
    if cfg['lrSchedule'] not in ('legacy', 'warmup_hold_cosine'):
        raise ValueError('Unknown lrSchedule')
    for key in ('lrWarmupEpochs', 'lrHoldEnd', 'lrDecayEnd'):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int):
            raise ValueError(f'{key} must be an integer')
    if not 1 <= cfg['lrWarmupEpochs'] <= cfg['lrHoldEnd'] < cfg['lrDecayEnd']:
        raise ValueError('Require 1 <= lrWarmupEpochs <= lrHoldEnd < lrDecayEnd')
    if not math.isfinite(cfg['lrMinFactor']) or not 0 < cfg['lrMinFactor'] <= 1:
        raise ValueError('lrMinFactor must be finite and in (0, 1]')
    return cfg


def add_schedule_arguments(parser):
    parser.add_argument('--lrSchedule', choices=('legacy', 'warmup_hold_cosine'), default='legacy')
    parser.add_argument('--lrWarmupEpochs', type=int, default=10)
    parser.add_argument('--lrHoldEnd', type=int, default=100)
    parser.add_argument('--lrDecayEnd', type=int, default=500)
    parser.add_argument('--lrMinFactor', type=float, default=0.125)


def learning_rate_factors(config, epoch_index, batch_index, num_batches):
    """Zero-based epoch/batch indices. No mutable scheduler state on resume.

    New schedule: warm up within batches, hold through lrHoldEnd, then cosine
    decay at epoch boundaries. lrDecayEnd is independent of total run epochs;
    extending a completed run therefore holds the floor, never restarts decay.
    """
    cfg = schedule_config(config)
    if epoch_index < 0 or num_batches < 1 or not 0 <= batch_index < num_batches:
        raise ValueError('Invalid epoch/batch position')
    if cfg['lrSchedule'] == 'legacy':
        head = 1. if epoch_index < 2 else .5 if epoch_index < 4 else .25 if epoch_index < 8 else .125
        sam = 1. if epoch_index < 2 else .5 if epoch_index < 4 else .25
        if epoch_index == 0:
            # Keep NumPy's original floating-point evaluation for old runs.
            import numpy as np
            head = sam = 1 - np.exp(-batch_index / num_batches * 3)
        return head, sam
    epoch = epoch_index + 1
    if epoch <= cfg['lrWarmupEpochs']:
        factor = (epoch_index + (batch_index + 1) / num_batches) / cfg['lrWarmupEpochs']
    elif epoch <= cfg['lrHoldEnd']:
        factor = 1.
    else:
        progress = min(1., (epoch - cfg['lrHoldEnd']) / (cfg['lrDecayEnd'] - cfg['lrHoldEnd']))
        floor = cfg['lrMinFactor']
        factor = floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2
    return factor, factor
