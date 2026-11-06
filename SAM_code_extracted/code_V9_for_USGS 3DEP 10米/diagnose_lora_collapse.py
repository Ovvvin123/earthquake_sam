"""Read-only diagnosis of the completed V5/V6 LoRA runs.

Writes only diagnostics/lora_collapse.json. No optimizer steps or checkpoint edits.
Run from the project directory with the earthquake-sam WSL Python environment.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torchUnet import FocalTversky
from tool import Generator_new
from sam2_lora import load_run_model
from diagnose_greece_nz_collapse import log_stats, gradient_probe

HERE = Path(__file__).resolve().parent
RUNS = {
    'V5': ('train_v5_unimatch_lora_r8_d001_seed7', 'earth_dataset_v3'),
    'V6': ('train_greece_nz_unimatch_lora_r8_d001_seed7', 'earth_dataset_greece_nz_stride64_v2'),
}


def read_items(folder, split):
    keys = (folder / f'key_{split}.lst').read_text().split()
    reader = Generator_new(str(folder / 'faults.h5'), keys, n=256, m=256,
                           training=False, fixed_crop=True)
    fractions, invalid, finite = [], 0, True
    baselines = {str(p): [] for p in [0., .1, .5, 1.]}
    selected, items = [], []
    for i, key in enumerate(keys):
        image, distances = reader[i]
        finite &= bool(np.isfinite(image).all() and np.isfinite(distances).all())
        valid = (image[2:4] > .5).all(0)
        invalid += int((~valid).sum())
        d = torch.from_numpy(distances[None, :1])
        target = (d < .01).float()
        sw = (torch.exp(-d.square() / .08**2) * .9 + .1)
        sw *= torch.from_numpy(valid[None, None])
        sw /= sw.mean().clamp_min(1e-12)
        fractions.append(float(target[0, 0][torch.from_numpy(valid)].mean()))
        if split == 'val':
            for p, values in baselines.items():
                values.append(float(FocalTversky(target, torch.full_like(target, float(p)), sw=sw, pu=0)))
        # One positive crop per region for V6, two positive crops for V5.
        region = key.split('__')[0]
        if fractions[-1] > 0 and len(items) < 2:
            if not selected or folder.name == 'earth_dataset_v3' or region != selected[0].split('__')[0]:
                selected.append(key)
                items.append((image, distances))
    stats = dict(tiles=len(keys), empty_crops=sum(x == 0 for x in fractions),
                 mean_positive_fraction=float(np.mean(fractions)),
                 positive_fraction_quantiles=np.quantile(fractions, [0, .25, .5, .75, 1]).tolist(),
                 invalid_pixels=invalid, all_inputs_labels_finite=finite,
                 scope='Deterministic 256x256 center crops, same loader as fixed validation')
    if split == 'val':
        stats['constant_prediction_loss'] = {p: float(np.mean(v)) for p, v in baselines.items()}
    return stats, selected, items


def main():
    torch.set_num_threads(4)
    report = {'note': 'Read-only checkpoint inference and isolated logit gradients; no model updates. '
                      'BCE probes demonstrate gradient behavior, not validated training convergence.',
              'torch_version': torch.__version__, 'runs': {}}
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    for name, (run_name, folder_name) in RUNS.items():
        run, folder = HERE / run_name, HERE / folder_name
        cfg = json.loads((run / 'model_config.json').read_text())
        result = dict(run=run_name, epoch1=log_stats(run), config=cfg)
        result['train_crops'], keys, items = read_items(folder, 'train')
        result['val_crops'], _, _ = read_items(folder, 'val')
        result['best'] = json.loads((run / 'best_metrics.json').read_text())
        result['early_adaptation'] = [json.loads(x) for x in
                                     (run / 'adaptation_history.jsonl').read_text().splitlines()[:3]]
        summaries = [json.loads(p.read_text()) for p in (run / 'online').glob('epoch_*/summary.json')]
        result['online'] = dict(epochs=len(summaries),
                               candidate_pixels=sum(x['totals']['candidate_pixels'] for x in summaries),
                               accepted_components=sum(x['totals']['accepted_components'] for x in summaries))
        result['unimatch_final'] = json.loads((run / f'unimatch/epoch_{cfg["epochs"]:04d}.json').read_text())
        images = torch.from_numpy(np.stack([x[0] for x in items]))
        distances = torch.from_numpy(np.stack([x[1][:1] for x in items]))
        target = (distances < .01).float()
        sw = torch.exp(-distances.square() / .08**2) * .9 + .1
        sw *= (images[:, 2:4] > .5).all(1, keepdim=True)
        sw /= sw.mean()
        net = load_run_model(run / 'best_student/model.pth', device)
        net.eval()
        captured = []
        hook = net.tail[-2].register_forward_hook(lambda m, i, o: captured.append(o.detach().cpu()))
        with torch.no_grad():
            prediction = net(images.to(device)).cpu()
        hook.remove()
        logits = captured[0]
        margin = logits[:, :1] - torch.logsumexp(logits[:, 1:], 1, keepdim=True)
        result['probe'] = dict(keys=keys, label_positive_fraction=float(target.mean()),
                              prediction_mean=float(prediction[:, :1].mean()),
                              logit_margin_quantiles=np.quantile(margin.numpy(), [0, .5, .99, 1]).tolist(),
                              binary_probability_max_error=float((margin.sigmoid() - prediction[:, :1]).abs().max()),
                              saturated_tversky=gradient_probe(logits, target, sw, 'focal_tversky'),
                              unsaturated_tversky=gradient_probe(torch.zeros_like(logits), target, sw, 'focal_tversky'),
                              saturated_bce=gradient_probe(logits, target, sw, 'bce'),
                              uniform_loss={str(p): float(FocalTversky(target, torch.full_like(target, p), sw=sw, pu=0))
                                            for p in [.1, .5, .9, 1.]},
                              lora_B_nonzero_parameters=sum(int(p.count_nonzero()) for n, p in net.named_parameters()
                                                            if 'lora_B' in n))
        # All-background batch has no positive means; report only its complete logit gradient.
        empty = torch.zeros_like(target)
        z = torch.zeros_like(logits).requires_grad_(True)
        loss = FocalTversky(empty, z.softmax(1), sw=torch.ones_like(empty), pu=0)
        result['empty_batch_probe'] = dict(loss=float(loss.detach()),
                                          mean_abs_logit_gradient=float(torch.autograd.grad(loss, z)[0].abs().mean()))
        report['runs'][name] = result
        print(name, json.dumps({k: result[k] for k in ['epoch1', 'train_crops', 'val_crops', 'probe', 'empty_batch_probe']}), flush=True)
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out = HERE / 'diagnostics/lora_collapse.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print('Saved:', out, flush=True)


if __name__ == '__main__':
    main()
