"""Full-tile, original-label validation; invalid/padded pixels are excluded."""
import numpy as np
import torch
from grid_data import target_weights, sliding_predict


def pixel_metrics(prob, truth):
    p, y = np.concatenate(prob), np.concatenate(truth).astype(bool)
    pred = p >= .5
    tp, fp, fn = int((pred&y).sum()), int((pred&~y).sum()), int((~pred&y).sum())
    ap = None
    if y.any():
        order = np.argsort(-p, kind='stable')
        ps, ys = p[order], y[order]
        ends = np.r_[np.flatnonzero(np.diff(ps)), len(ps)-1]
        hits = np.cumsum(ys)[ends]
        precision = hits/(ends+1)
        recall = hits/y.sum()
        ap = float(np.sum(np.diff(np.r_[0., recall])*precision))
    return dict(AP=ap, threshold=.5, precision=tp/max(tp+fp, 1), recall=tp/max(tp+fn, 1),
                F1=2*tp/max(2*tp+fp+fn, 1), IoU=tp/max(tp+fp+fn, 1),
                valid_pixels=len(p), positive_pixels=int(y.sum()))


@torch.no_grad()
def validate(model, dataset, device, args, loss_func, preview=None):
    modes = [(m, m.training) for m in model.modules()]
    losses, probs, truths, examples = [], [], [], []
    try:
        model.eval()
        for k in range(len(dataset)):
            x, y = dataset[k]
            image = torch.from_numpy(x[None]).to(device)
            distances = torch.from_numpy(y[None]).to(device)
            target, sw = target_weights(image, distances, args.d0, args.dw, args.alpha)
            pred = sliding_predict(model, image)
            loss = loss_func(target, pred, sw=sw, pu=0.)
            if not torch.isfinite(loss) or not torch.isfinite(pred).all():
                raise FloatingPointError('Nonfinite validation output')
            losses.append(float(loss))
            v = x[2:4].min(0) > .5
            p = pred[0, 0].cpu().numpy()
            t = target[0, 0].cpu().numpy()
            probs.append(p[v]); truths.append(t[v])
            if preview and k < 4:
                examples.append((dataset.list[k], x[0], t, p, v))
        result = dict(loss=float(np.mean(losses)), tiles=len(dataset), **pixel_metrics(probs, truths))
        if preview:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(len(examples), 3, figsize=(10, 3*len(examples)), squeeze=False)
            for row, (key, terrain, truth, pred, valid) in zip(ax, examples):
                for a, arr, title in zip(row, (terrain, truth, pred), ('terrain', 'truth', 'k0 probability')):
                    a.imshow(np.ma.masked_where(~valid, arr), cmap='gray' if title == 'terrain' else 'viridis')
                    a.set_title(f'{key}: {title}'); a.axis('off')
            fig.tight_layout(); fig.savefig(preview, dpi=120); plt.close(fig)
        return result
    finally:
        for module, training in modes:
            module.training = training
