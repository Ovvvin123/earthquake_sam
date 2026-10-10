"""Read-only fixed-crop diagnostics; candidates are NOT training labels."""
from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation, label
from aio2_ema import _rng_state, _restore_rng


@contextmanager
def preserve_rng():
    state = _rng_state()
    try:
        yield
    finally:
        _restore_rng(state)


def probability_stats(probability, valid, threshold):
    values = np.asarray(probability)[valid]
    finite = values[np.isfinite(values)]
    result = {'pixels': int(values.size), 'nonfinite': int(values.size - finite.size)}
    result.update({name: None for name in ('mean', 'min', 'max', 'p50', 'p90', 'p99', 'above_threshold_fraction')})
    result['histogram'] = np.histogram(finite, bins=np.linspace(0, 1, 21))[0].tolist()
    if finite.size:
        result.update(mean=float(finite.mean()), min=float(finite.min()), max=float(finite.max()),
                      p50=float(np.quantile(finite, .5)), p90=float(np.quantile(finite, .9)),
                      p99=float(np.quantile(finite, .99)),
                      above_threshold_fraction=float((finite >= threshold).mean()))
    return result


def candidate_masks(probability, truth, valid, threshold, buffer_px, min_pixels):
    """Diagnostic threshold + 8-connected area filter, not the full O2C rule."""
    exclusion = binary_dilation(truth, iterations=buffer_px) if buffer_px else truth
    raw = (probability >= threshold) & np.isfinite(probability) & valid & ~exclusion
    components, count = label(raw, structure=np.ones((3, 3), dtype=bool))
    areas = np.bincount(components.ravel())
    keep = areas >= min_pixels
    keep[0] = False
    return raw, keep[components], int(count), int(keep.sum())


def _even(keys, count):
    if not keys or count <= 0:
        return []
    return [keys[i] for i in np.linspace(0, len(keys) - 1, min(count, len(keys)), dtype=int)]


class FixedPreview:
    def __init__(self, dataset_type, filename, keys, output_dir, n, m, d0,
                 count=6, threshold=.7, buffer_px=2, min_pixels=8):
        from h5py import File
        if count < 1 or not 0 < threshold < 1 or buffer_px < 0 or min_pixels < 1:
            raise ValueError('Invalid preview count, threshold, buffer or minimum area.')
        keys = sorted(set(keys))
        if not keys:
            raise ValueError('No training keys available for fixed preview.')
        # Deterministic coverage of both labeled and unlabeled central crops.
        positive, empty = [], []
        with File(filename, 'r') as h5:
            for key in keys:
                shape = h5[key]['k0'].shape
                if len(shape) != 2 or shape[0] < n or shape[1] < m:
                    raise ValueError(f'Preview crop does not fit {key}: {shape}')
                y, x = (shape[0] - n) // 2, (shape[1] - m) // 2
                distance = h5[key]['k0'][y:y+n, x:x+m]
                (positive if ((distance >= 0) & (distance < d0)).any() else empty).append(key)
        chosen = _even(positive, (count + 1) // 2) + _even(empty, count // 2)
        chosen += _even([key for key in keys if key not in chosen], count - len(chosen))
        chosen = sorted(chosen)
        dataset = dataset_type(filename, chosen, n=n, m=m, training=False, fixed_crop=True)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.threshold, self.buffer_px, self.min_pixels = threshold, buffer_px, min_pixels
        self.samples, descriptions = [], []
        for i, key in enumerate(chosen):
            image, distance = dataset[i]
            image = np.ascontiguousarray(image, dtype=np.float32)
            truth = np.ascontiguousarray(distance[0] < d0)
            valid = (image[2:4] > .5).all(axis=0) & np.isfinite(image).all(axis=0)
            self.samples.append((key, torch.from_numpy(image[None].copy()), truth, valid))
            digest = hashlib.sha256(image.tobytes() + truth.tobytes() + valid.tobytes()).hexdigest()
            descriptions.append({'key': key, 'sha256': digest,
                                 'positive_pixels': int(truth.sum()), 'valid_pixels': int(valid.sum())})
        manifest = {'crop': 'center; no augmentation', 'n': n, 'm': m, 'd0': d0,
                    'selection': 'fixed training crops, balanced by presence of original k0 label',
                    'samples': descriptions}
        path = self.output_dir / 'samples.json'
        if path.exists() and json.loads(path.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Fixed preview samples changed. Use a new preview output directory.')
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')

    @torch.no_grad()
    def write(self, student, teacher, epoch):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        directory = self.output_dir / f'epoch_{epoch:04d}'
        directory.mkdir(parents=True, exist_ok=True)
        modes = [(module, module.training) for module in student.modules()]
        records = []
        with preserve_rng():
            try:
                student.eval()
                for index, (key, cpu_image, truth, valid) in enumerate(self.samples):
                    inputs = cpu_image.to(next(student.parameters()).device)
                    sp = student(inputs)[0, 0].detach().float().cpu().numpy()
                    tp = teacher.predict(inputs)[0, 0].detach().float().cpu().numpy()
                    if sp.shape != truth.shape or tp.shape != truth.shape:
                        raise ValueError('Prediction and fixed-crop label dimensions differ.')
                    raw, candidate, raw_count, candidate_count = candidate_masks(
                        tp, truth, valid, self.threshold, self.buffer_px, self.min_pixels)
                    record = {'key': key, 'valid_pixels': int(valid.sum()),
                              'original_positive_fraction': float(truth.mean()),
                              'valid_positive_fraction': float(truth[valid].mean()) if valid.any() else None,
                              'student': probability_stats(sp, valid, self.threshold),
                              'teacher': probability_stats(tp, valid, self.threshold),
                              'raw_candidate_pixels': int(raw.sum()), 'raw_components': raw_count,
                              'filtered_candidate_pixels': int(candidate.sum()),
                              'filtered_components': candidate_count,
                              'candidate_fraction_valid': float(candidate.sum()/valid.sum()) if valid.any() else None}
                    records.append(record)
                    stem = f'sample_{index:02d}'
                    np.savez_compressed(directory / f'{stem}.npz', student=sp, teacher=tp,
                                        original_label=truth, valid=valid,
                                        raw_candidates=raw, filtered_candidates=candidate)
                    background = np.clip(cpu_image[0, 0].numpy() * .5 + .5, 0, 1)
                    fig = Figure(figsize=(13, 8), layout='constrained')
                    FigureCanvasAgg(fig)
                    axes = fig.subplots(2, 3)
                    axes[0, 0].imshow(background, cmap='gray', vmin=0, vmax=1)
                    axes[0, 0].set_title('Terrain (normalized CTX)')
                    overlay = np.zeros((*truth.shape, 4))
                    overlay[truth] = (0, 1, .3, .85)
                    axes[0, 1].imshow(background, cmap='gray', vmin=0, vmax=1)
                    axes[0, 1].imshow(overlay)
                    axes[0, 1].set_title('Original labels (green)')
                    im = axes[0, 2].imshow(np.ma.array(sp, mask=~valid), vmin=0, vmax=1, cmap='magma')
                    axes[0, 2].set_title('Student probability')
                    axes[1, 0].imshow(np.ma.array(tp, mask=~valid), vmin=0, vmax=1, cmap='magma')
                    axes[1, 0].set_title('EMA teacher probability')
                    fig.colorbar(im, ax=[axes[0, 2], axes[1, 0]], shrink=.65, label='Probability')
                    overlay[raw] = (0, 1, 1, .65)
                    overlay[candidate] = (1, 0, 1, .85)
                    axes[1, 1].imshow(background, cmap='gray', vmin=0, vmax=1)
                    axes[1, 1].imshow(overlay)
                    axes[1, 1].set_title('Candidates: cyan=raw, magenta=area-filtered')
                    for name, probability, color in [('Student', sp, '#d97706'), ('Teacher', tp, '#2563eb')]:
                        values = probability[valid & np.isfinite(probability)]
                        axes[1, 2].hist(values, bins=np.linspace(0, 1, 21), histtype='step',
                                        label=name, color=color, linewidth=1.5)
                    axes[1, 2].axvline(self.threshold, color='gray', linestyle='--')
                    axes[1, 2].set(title='Valid-pixel probability distribution', xlabel='Probability', ylabel='Pixels')
                    axes[1, 2].legend()
                    for ax in axes.ravel()[:5]:
                        ax.set_axis_off()
                    fig.suptitle(f'Epoch {epoch} | {key}\nCandidates are unverified; preview only, no label correction.',
                                 fontsize=11)
                    fig.savefig(directory / f'{stem}.png', dpi=120)
                    fig.clear()
            finally:
                for module, mode in modes:
                    module.training = mode
        report = {'epoch': epoch, 'ema_updates': teacher.num_updates, 'preview_only': True,
                  'threshold': self.threshold, 'buffer_px': self.buffer_px, 'min_pixels': self.min_pixels,
                  'histogram_edges': np.linspace(0, 1, 21).tolist(), 'samples': records}
        (directory / 'metrics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
                                                encoding='utf-8')
        return directory


class BatchDiagnostics:
    """k0 statistics on the actual augmented batches used for training."""
    def __init__(self, threshold=.7):
        self.threshold = threshold
        self.rows = []
        self.histogram = np.zeros(20, dtype=np.int64)

    def observe(self, batch, target, prediction, image, loss):
        truth = target[:, 0].detach().cpu().numpy() > .5
        prob = prediction[:, 0].detach().float().cpu().numpy()
        valid = (image[:, 2:4].detach().cpu().numpy() > .5).all(axis=1)
        valid &= np.isfinite(image.detach().cpu().numpy()).all(axis=1)
        stats = probability_stats(prob, valid, self.threshold)
        self.histogram += np.asarray(stats.pop('histogram'))
        positive = int(truth.sum())
        valid_positive = int((truth & valid).sum())
        row = {'batch': batch, 'loss': float(loss) if np.isfinite(loss) else None,
               'original_positive_pixels': positive, 'original_positive_fraction': float(truth.mean()),
               'valid_pixels': int(valid.sum()), 'valid_positive_pixels': valid_positive,
               'valid_positive_fraction': float(truth[valid].mean()) if valid.any() else None,
               'empty_label_samples': int((truth.reshape(len(truth), -1).sum(axis=1) == 0).sum()),
               'near_one_loss': bool(np.isfinite(loss) and loss >= .9999)}
        row.update({f'prediction_{key}': value for key, value in stats.items()})
        self.rows.append(row)

    def write(self, output_dir, epoch):
        directory = Path(output_dir) / f'epoch_{epoch:04d}'
        directory.mkdir(parents=True, exist_ok=True)
        if not self.rows:
            return
        with (directory / 'train_batches.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        groups = {}
        for name, predicate in [('empty_label', lambda r: r['original_positive_pixels'] == 0),
                                ('with_label', lambda r: r['original_positive_pixels'] > 0)]:
            rows = [r for r in self.rows if predicate(r)]
            finite_losses = [r['loss'] for r in rows if r['loss'] is not None]
            groups[name] = {'batches': len(rows), 'near_one_loss_batches': sum(r['near_one_loss'] for r in rows),
                            'mean_loss': float(np.mean(finite_losses)) if finite_losses else None}
        report = {'epoch': epoch, 'channel': 'k0', 'threshold': self.threshold,
                  'batches': len(self.rows), 'groups_by_original_label': groups,
                  'no_valid_pixel_batches': sum(r['valid_pixels'] == 0 for r in self.rows),
                  'nonfinite_loss_batches': sum(r['loss'] is None for r in self.rows),
                  'histogram_edges': np.linspace(0, 1, 21).tolist(), 'valid_prediction_histogram': self.histogram.tolist()}
        (directory / 'train_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print(f'Diagnostics epoch={epoch}: empty-label batches={groups["empty_label"]["batches"]}, '
              f'near-one loss: empty={groups["empty_label"]["near_one_loss_batches"]}, '
              f'with-label={groups["with_label"]["near_one_loss_batches"]}')
