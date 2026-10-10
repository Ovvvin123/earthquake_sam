"""Fill the missing original UniMatch best-student training-region mosaic."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

from compare_three_version_lines import sha


def main():
    import torch
    import h5py
    from evaluate_cv import predict_split
    from torchUnet import conv_sam

    base = Path(__file__).resolve().parent
    os.chdir(base)
    run = base / 'train_v5_unimatch'
    output = run / 'train_line_evaluation'
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output already contains files; inspect before rerunning.')
    model_path = run / 'best_student/model.pth'
    evaluation = json.loads((run / 'cv_evaluation/student/metrics.json').read_text())
    model_hash = sha(model_path)
    assert model_hash == evaluation['model_sha256']
    keys = (base / 'earth_dataset_v3/key_train.lst').read_text().split()
    assert len(keys) == len(set(keys)) == 161
    assert set(keys) == set(evaluation['split_keys']['train'])
    for folder in ['train_v5_online', 'train_v5_unimatch_d001']:
        old = json.loads((base / folder / 'predictions_seen_unseen/manifest.json').read_text())
        assert set(keys) == {r['key'] for r in old['images'] if r['split'] == 'train_seen'}
    torch.set_num_threads(4)
    torch.manual_seed(0)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Device:', device, flush=True)
    model = conv_sam(32, dropout=0., isvit=False, isalign=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True), strict=True)
    model.eval()
    output.mkdir(exist_ok=True)
    data_path = base / 'earth_dataset_v3/faults.h5'
    with h5py.File(data_path, 'r') as h5:
        _, _, _, audit = predict_split(model, h5, keys, SimpleNamespace(crop=256, d0=.02),
                                       device, output, 'train_seen')
    report = dict(status='complete', model=str(model_path), model_sha256=model_hash,
                  epoch=json.loads((run/'best_student/metrics.json').read_text())['epoch'],
                  data=str(data_path), data_sha256=sha(data_path), keys=keys, audit=audit,
                  crop=256, d0=.02, device=device, threshold_for_later_counts=.5,
                  inference_script_sha256=sha(base/'evaluate_cv.py'),
                  protocol='Reuse evaluate_cv.predict_split: full-tile sliding windows and geographic overlap averaging; original labels only; no training')
    (output/'provenance.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
