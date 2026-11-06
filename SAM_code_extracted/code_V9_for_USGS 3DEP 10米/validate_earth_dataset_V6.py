"""CPU-only data/interface validation and atomic publication. No model or training."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import shutil

import h5py
import numpy as np
import torch
from tool import Generator_new
from fault_losses import supervision
from aio2_online import temporary_targets
from unimatch_fault import reliable_weights, positive_consistency

CODE = Path(__file__).resolve().parent
SRC = CODE / 'earth_dataset_v5'
STAGE = CODE / '.earth_dataset_V6.building'
DST = CODE / 'earth_dataset_V6'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    if DST.exists():
        raise FileExistsError('Refusing to overwrite published dataset')
    report_path = STAGE / 'qa/validation_report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    assert report['status'] == 'STRUCTURAL_PASS_PENDING_RUNTIME'
    torch.set_num_threads(1)
    keys = (STAGE / 'key_active.lst').read_text().split()
    train_keys = (STAGE / 'key_train.lst').read_text().split()
    args = SimpleNamespace(k0Only=True, swMode='single', isOther=False,
                           supervisionKey='label_known', onlineSupport=.5,
                           onlineThreshold=.9, onlineBuffer=0, onlineMaxFraction=1.,
                           onlineMinPixels=1, collapseGuard=True)
    loader = Generator_new(str(STAGE / 'faults.h5'), keys, training=False,
                           fixed_crop=True, supervision_key='label_known', known_crop=False)
    assert loader.list == keys
    marine_valid_pixels = supervised_pixels = 0
    marine_example = None
    with h5py.File(STAGE / 'faults.h5', 'r') as f, torch.no_grad():
        for index, key in enumerate(keys):
            image, distances = loader[index]
            loader.dataD.clear()  # Validation should not fill the training cache.
            assert image.shape == (7, 256, 256) and distances.shape == (4, 256, 256)
            mask = f[key]['label_known'][64:320, 64:320]
            raw = f[key]['image'][64:320, 64:320]
            np.testing.assert_array_equal(distances[3], mask)
            np.testing.assert_array_equal(distances[0] < .01, f[key]['k0'][64:320, 64:320] < .01)
            x, d = torch.from_numpy(image[None]), torch.from_numpy(distances[None])
            target, weights, valid = supervision(x, d, args, .01, .15, .1)
            marine = torch.from_numpy((raw < 0)[None, None]) & valid
            assert not target[marine].any() and not weights[marine].any()
            assert torch.equal(weights > 0, valid & (d[:, 3:4] > .5))
            marine_valid_pixels += int(marine.sum())
            supervised_pixels += int((weights > 0).sum())
            if marine_example is None and key in train_keys and bool(marine.all()):
                marine_example = (key, x.clone(), d.clone())
            if (index + 1) % 50 == 0:
                print(f'Production loader/supervision checked {index + 1}/{len(keys)}', flush=True)
        assert marine_example is not None and supervised_pixels > 0
        key, x, d = marine_example
        target, weights, valid = supervision(x, d, args, .01, .15, .1)
        assert valid.all() and not weights.any()
        # A synthetic confident patch exercises the real selection/consistency
        # functions on a real marine input. No model, optimizer, backward or GPU.
        probability = torch.zeros_like(target)
        probability[:, :, 10:14, 10:14] = .99
        corrected, pseudo_weights, selected, stats = temporary_targets(
            target, weights, probability, x, .4, args)
        assert int(selected.sum()) == 16 and (pseudo_weights[selected] > 0).all()
        assert (corrected[selected] == 1).all()
        consistency_weights = reliable_weights(target, selected, x, .4)
        assert int((consistency_weights > 0).sum()) == 16
        loss = positive_consistency(torch.full_like(target, .8), consistency_weights)
        assert torch.isfinite(loss) and loss > 0
        # A known pixel must not be overwritten by a teacher candidate.
        known_d = d.clone(); known_d[:, 3] = 1
        known_target, known_weights, _ = supervision(x, known_d, args, .01, .15, .1)
        _, _, rejected, _ = temporary_targets(known_target, known_weights, probability, x, .4, args)
        assert not rejected.any()

    # Confirm the complete training list retains wholly marine examples.
    train_loader = Generator_new(str(STAGE / 'faults.h5'), train_keys, training=True,
                                supervision_key='label_known', known_crop=False)
    assert train_loader.list == train_keys and key in train_loader.list
    # Test the real augmentation path without fitting a model.
    for seed in range(4):
        np.random.seed(seed)
        image, distances = train_loader[train_keys.index(key)]
        train_loader.dataD.clear()
        assert np.isin(distances[3], [0, 1]).all() and not distances[3].any()
        assert np.isfinite(image).all() and (image[2:4] > .5).all()

    # Regression: V5's existing marine supervised mask is still consumed normally.
    old_keys = (SRC / 'key_train.lst').read_text().split()
    with h5py.File(SRC / 'faults.h5', 'r') as f:
        old_key = next(k for k in old_keys if f[k]['label_known'][64:320, 64:320].any())
    old_loader = Generator_new(str(SRC / 'faults.h5'), [old_key], training=False,
                               fixed_crop=True, supervision_key='label_known')
    old_image, old_d = old_loader[0]
    old_target, old_weights, _ = supervision(torch.from_numpy(old_image[None]),
                                            torch.from_numpy(old_d[None]), args, .01, .15, .1)
    assert (old_weights > 0).any()
    with h5py.File(SRC / 'faults.h5', 'r') as f:
        raw = torch.from_numpy(f[old_key]['image'][64:320, 64:320][None, None])
        assert (raw[old_weights > 0] < 0).all()
    assert report['source_files_sha256'] == {p.name: digest(p) for p in SRC.iterdir() if p.is_file()}
    assert report['core_files_sha256'] == {name: digest(CODE / name) for name in report['core_files_sha256']}
    result = dict(status='PASS', checked_active_fixed_crops=len(keys),
                  loaded_training_tiles=len(train_keys), all_training_keys_retained=True,
                  valid_marine_pixels_in_fixed_crops=marine_valid_pixels,
                  supervised_pixels_in_fixed_crops=supervised_pixels,
                  marine_direct_supervision_pixels=0,
                  marine_example_key=key, marine_synthetic_pseudo_pixels=int(selected.sum()),
                  marine_consistency_selected_pixels=int((consistency_weights > 0).sum()),
                  marine_consistency_loss=float(loss), known_mask_blocks_teacher_overwrite=True,
                  augmentation_checks=4, v5_marine_supervision_regression='PASS',
                  source_v5_unchanged=True, training_code_unchanged=True,
                  model_constructed=False, backward_or_optimizer_called=False, training_launched=False,
                  note='Synthetic confidence verifies eligibility only; actual pseudo-label acceptance depends on the future teacher and thresholds.')
    (STAGE / 'qa/runtime_validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    report['status'] = 'PASS'
    report['production_interface_validation'] = 'PASS'
    report['published_dataset_name'] = 'earth_dataset_V6'
    report['dataset_hdf5_sha256'] = digest(STAGE / 'faults.h5')
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    shutil.copy2(Path(__file__), STAGE / 'qa' / Path(__file__).name)
    assert not DST.exists()
    assert STAGE.resolve().parent == CODE and DST.resolve().parent == CODE
    os.rename(STAGE, DST)
    print(json.dumps(result, indent=2), flush=True)
    print(f'Published: {DST}', flush=True)


if __name__ == '__main__':
    main()
