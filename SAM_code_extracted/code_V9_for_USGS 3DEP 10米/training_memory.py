"""Opt-in V9 execution settings. FP32 parameters/EMA; BF16 forward only."""
from contextlib import nullcontext
import torch
from torch.utils.checkpoint import checkpoint

DEFAULTS = dict(computePrecision='fp32', attentionBackend='legacy',
                poolFirst=False, activationCheckpoint='none')


def memory_config(config=None):
    config = {} if config is None else config if isinstance(config, dict) else vars(config)
    return {key:config.get(key,value) for key,value in DEFAULTS.items()}


def precision_context(tensor, precision):
    if precision == 'fp32':
        return nullcontext()
    if precision != 'bf16' or tensor.device.type != 'cuda':
        raise ValueError('V9 BF16 execution requires CUDA')
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError('Native CUDA BF16 is unavailable')
    return torch.autocast(device_type='cuda',dtype=torch.bfloat16)


def recompute(function, *args, enabled=False):
    if enabled and torch.is_grad_enabled():
        return checkpoint(function,*args,use_reentrant=False,preserve_rng_state=True)
    return function(*args)
