"""Whole-tile k0 validation for the random-grid experiment (not a held-out test)."""
import argparse
import json
import torch
from grid_core import local_path
from grid_data import GridDataset
from grid_validation import validate
from train_grid import read_splits
from torchUnet import conv_sam, FocalTversky

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resDir', default='runs/supervised')
    p.add_argument('--dataDir', default='datasets/ca_stride384')
    p.add_argument('--keyFile', default='key_val.lst')
    p.add_argument('--d0', type=float, default=.02)
    p.add_argument('--dw', type=float, default=.08)
    p.add_argument('--alpha', type=float, default=.1)
    p.add_argument('--device', default='auto')
    a = p.parse_args()
    out, data = local_path(a.resDir), local_path(a.dataDir)
    if a.keyFile != 'key_val.lst':
        raise ValueError('This experiment evaluates the fixed validation split only')
    config_file = out/'config.json'
    if not config_file.exists(): config_file = out.parent/'config.json'
    if config_file.exists():
        config = json.loads(config_file.read_text())
        for name in ('d0', 'dw', 'alpha'):
            if getattr(a, name) != config[name]:
                raise ValueError(f'{name} differs from training config; pass the trained value explicitly')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device
    keys, signature = read_splits(data)
    signature_file = out/'data_signature.json'
    if not signature_file.exists(): signature_file = out.parent/'data_signature.json'
    if not signature_file.exists() or json.loads(signature_file.read_text())['sha256'] != signature:
        raise ValueError('Evaluation data differ from training dataset fingerprint')
    from sam2_lora import load_run_model
    net = load_run_model(out, device)
    dataset = GridDataset(data/'faults.h5', (data/a.keyFile).read_text().split(), full=True)
    report = validate(net, dataset, device, a, FocalTversky, out/'val_qualitative.png')
    report['scope'] = 'within-region random validation; used for model selection; not independent test'
    (out/'val_metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
