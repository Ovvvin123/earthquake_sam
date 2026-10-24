"""SAM-2 Q/V adapters and shared training/inference configuration.

No automatic weight merging: eval() must never mutate the frozen SAM base.
This file is mirrored in V5_different_split to keep that experiment standalone.
"""
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn


DEFAULTS = dict(samAdaptation='full', loraRank=8, loraAlpha=16.,
                loraDropout=0., loraStages='all', loraLr=5e-5,
                loraWeightDecay=0.)


def adaptation_config(config=None):
    source = {} if config is None else (config if isinstance(config, dict) else vars(config))
    result = {key: source.get(key, value) for key, value in DEFAULTS.items()}
    if result['samAdaptation'] not in ('full', 'frozen', 'lora'):
        raise ValueError('samAdaptation must be full, frozen or lora')
    if result['loraStages'] not in ('all', 'last2'):
        raise ValueError('loraStages must be all or last2')
    rank = result['loraRank']
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError('loraRank must be a positive integer')
    for key in ('loraAlpha', 'loraLr'):
        if not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f'{key} must be positive and finite')
    if not math.isfinite(result['loraWeightDecay']) or result['loraWeightDecay'] < 0:
        raise ValueError('loraWeightDecay must be finite and nonnegative')
    if not 0 <= result['loraDropout'] < 1:
        raise ValueError('loraDropout must be in [0, 1)')
    return result


def add_adaptation_arguments(parser):
    parser.add_argument('--samAdaptation', choices=('full', 'frozen', 'lora'), default='full')
    parser.add_argument('--loraRank', type=int, default=8)
    parser.add_argument('--loraAlpha', type=float, default=16.)
    parser.add_argument('--loraDropout', type=float, default=0.)
    parser.add_argument('--loraStages', choices=('all', 'last2'), default='all')
    parser.add_argument('--loraLr', type=float, default=5e-5)
    parser.add_argument('--loraWeightDecay', type=float, default=0.)


class QVLoRA(nn.Module):
    """Wrap fused Linear(dim_in, 3*dim_out), including Hiera stage transitions."""
    def __init__(self, base, rank, alpha, dropout=0.):
        super().__init__()
        if not isinstance(base, nn.Linear) or base.out_features % 3:
            raise ValueError('Expected a fused SAM-2 QKV linear layer')
        self.base = base.requires_grad_(False)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        width = base.out_features // 3
        options = dict(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_Aq = nn.Linear(base.in_features, rank, bias=False, **options)
        self.lora_Bq = nn.Linear(rank, width, bias=False, **options)
        self.lora_Av = nn.Linear(base.in_features, rank, bias=False, **options)
        self.lora_Bv = nn.Linear(rank, width, bias=False, **options)
        nn.init.zeros_(self.lora_Bq.weight)
        nn.init.zeros_(self.lora_Bv.weight)

    def forward(self, x):
        q, k, v = self.base(x).chunk(3, dim=-1)
        z = self.dropout(x)
        return torch.cat((q + self.scale*self.lora_Bq(self.lora_Aq(z)), k,
                          v + self.scale*self.lora_Bv(self.lora_Av(z))), dim=-1)


def configure_sam(model, config=None):
    cfg = adaptation_config(config)
    model.sam_adaptation = cfg
    model.lora_layers = []
    if cfg['samAdaptation'] == 'full':
        return
    if getattr(model, 'isvit', False) or getattr(model, 'isalign', False):
        raise ValueError('Frozen/LoRA experiments support mars_sam without isalign only')
    model.sam.requires_grad_(False)
    if cfg['samAdaptation'] == 'frozen':
        return
    trunk = model.sam.image_encoder.trunk
    first = 0 if cfg['loraStages'] == 'all' else trunk.stage_ends[-3] + 1
    # Adapter initialization must not change subsequent data/augmentation RNG.
    devices = sorted({p.device.index for p in trunk.parameters() if p.is_cuda})
    with torch.random.fork_rng(devices=devices):
        for index, block in enumerate(trunk.blocks):
            if index < first:
                continue
            if isinstance(block.attn.qkv, QVLoRA):
                raise ValueError('LoRA already injected')
            block.attn.qkv = QVLoRA(block.attn.qkv, cfg['loraRank'],
                                    cfg['loraAlpha'], cfg['loraDropout'])
            model.lora_layers.append(f'sam.image_encoder.trunk.blocks.{index}.attn.qkv')
    if not model.lora_layers:
        raise ValueError('No SAM-2 QKV layers matched')


def training_mode(model):
    model.train()
    if hasattr(model, 'in_sam'):
        model.in_sam.eval()
    # Keep SAM stochastic depth disabled, while honoring adapter dropout.
    for module in model.modules():
        if isinstance(module, QVLoRA):
            module.train()


def optimizer_groups(model, args):
    cfg = adaptation_config(args)
    head = [p for p in model.not_sam.parameters() if p.requires_grad]
    sam = [p for p in model.in_sam.parameters() if p.requires_grad]
    groups = [dict(name='head', params=head, lr=args.lr0, base_lr=args.lr0)]
    if cfg['samAdaptation'] == 'full':
        groups.append(dict(name='sam', params=sam, lr=args.lr0/8, base_lr=args.lr0/8))
    elif cfg['samAdaptation'] == 'lora':
        expected = {id(p) for n, p in model.sam.named_parameters() if '.lora_' in n}
        if {id(p) for p in sam} != expected or not expected:
            raise ValueError('LoRA optimizer contains missing adapters or unfrozen SAM parameters')
        groups.append(dict(name='lora', params=sam, lr=cfg['loraLr'],
                           base_lr=cfg['loraLr'], weight_decay=cfg['loraWeightDecay']))
    elif sam:
        raise ValueError('Frozen SAM still has trainable parameters')
    ids = [id(p) for g in groups for p in g['params']]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError('Optimizer parameters duplicated or incomplete')
    return groups


def set_learning_rates(optimizer, args, head_factor, sam_factor=None):
    """Legacy checkpoints may lack group names; restore metadata by mode/order."""
    mode = adaptation_config(args)['samAdaptation']
    for index, group in enumerate(optimizer.param_groups):
        name = group.get('name', 'head' if index == 0 else ('lora' if mode == 'lora' else 'sam'))
        base = args.loraLr if name == 'lora' else args.lr0/8 if name == 'sam' else args.lr0
        factor = sam_factor if name == 'sam' and sam_factor is not None else head_factor
        group.update(name=name, base_lr=base, lr=float(base*factor))


def gradient_norms(optimizer):
    return {g.get('name', str(i)): float(torch.stack([
        p.grad.detach().float().norm().square() for p in g['params'] if p.grad is not None
    ]).sum().sqrt()) if any(p.grad is not None for p in g['params']) else 0.
            for i, g in enumerate(optimizer.param_groups)}


def normalized_config(config):
    result = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    result.update(adaptation_config(config))
    # Missing fields describe historical behavior, never silently opt old resumes in.
    for key, value in dict(tverskyWeight=.5, accumulationVersion='legacy_v1',
                           supervisionKey='', supervisionVersion='legacy_v1',
                           collapseGuard=False, consistencyLoss='probability',
                           dataFile='', knownCrop=False, minKnownPixels=1,
                           knownCropStride=32).items():
        result.setdefault(key, value)
    return result


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=32)
def _cached_digest(path, size, modified):
    return sha256_file(path)


def checkpoint_identity(config):
    """Content fingerprints, cached only while file size/mtime stay unchanged."""
    paths = {'sam_base': Path(__file__).parent/'sam2_hiera_small.pt'}
    if config.get('dataset'):
        data = Path(config['dataset']).parent
        paths.update({name: data/name for name in
                      ('faults.h5', 'key_train.lst', 'key_val.lst', 'manifest.json')})
    if config.get('dataFile'):
        paths['faults.h5'] = Path(config['dataFile'])
    result = {}
    for name, path in paths.items():
        if path.is_file():
            stat = path.stat()
            result[name] = _cached_digest(str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    return result


def check_checkpoint_identity(checkpoint, config):
    saved = checkpoint.get('identity')
    if saved is not None and saved != checkpoint_identity(config):
        raise ValueError('Checkpoint SAM base or dataset fingerprint mismatch')
    if saved is None and adaptation_config(config)['samAdaptation'] != 'full':
        raise ValueError('Adapted checkpoints require SAM base/data fingerprints')


def write_run_metadata(model, args, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = normalized_config(vars(args))
    (directory/'model_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sam = getattr(model, 'sam', nn.Identity())
    base_trainable = sum(p.numel() for n, p in sam.named_parameters()
                         if p.requires_grad and '.lora_' not in n)
    lora = sum(p.numel() for n, p in sam.named_parameters() if '.lora_' in n)
    report = dict(config=adaptation_config(args), total_parameters=total,
                  trainable_parameters=trainable, sam_base_trainable=base_trainable,
                  lora_parameters=lora, custom_trainable=trainable-base_trainable-lora,
                  injected_layers=getattr(model, 'lora_layers', []))
    report['source_sha256'] = {name: sha256_file(Path(__file__).parent/name)
        for name in ('sam2_lora.py', 'torchUnet.py', 'trainV10.py', 'train_grid.py', 'config.py',
                     'fault_losses.py', 'fault_metrics.py', 'tool.py', 'known_crops.py', 'aio2_online.py', 'unimatch_fault.py')
        if (Path(__file__).parent/name).is_file()}
    base = Path(__file__).parent/'sam2_hiera_small.pt'
    if base.exists():
        report['base_sha256'] = checkpoint_identity({})['sam_base']
    (directory/'adaptation_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)
    return report


def read_model_config(path):
    path = Path(path)
    directory = path if path.is_dir() else path.parent
    # best_student, teacher, best_teacher and checkpoints are run subdirectories.
    for parent in (directory, directory.parent):
        for name in ('model_config.json', 'config.json', 'ema_config.json'):
            candidate = parent/name
            if candidate.exists():
                return normalized_config(json.loads(candidate.read_text(encoding='utf-8')))
    return adaptation_config()


def build_model(config):
    from torchUnet import conv_sam
    return conv_sam(32, dropout=config.get('dropout', 0.),
                    isvit=config.get('modelType', 'mars_sam') == 'mars_sam_vit',
                    isalign=config.get('isalign', False), adaptation=config)


def load_run_model(path, device='cpu', config=None):
    """Strict loading; missing LoRA metadata is an error, never a silent fallback."""
    path = Path(path)
    if path.is_dir():
        path = path/'model.pth'
    state = torch.load(path, map_location='cpu', weights_only=True)
    if 'student' in state and 'config' in state:
        config = state['config'] if config is None else config
        state = state['student']
    cfg = read_model_config(path) if config is None else normalized_config(config)
    if any('.lora_' in key for key in state) and cfg['samAdaptation'] != 'lora':
        raise ValueError('LoRA weights require the matching model_config.json')
    model = build_model(cfg)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
