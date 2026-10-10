from pathlib import Path
from h5py import File
import numpy as np
root = Path('train_v5_unimatch_smoke/cli_active_fixture')
root.mkdir(exist_ok=True)
keys = Path('earth_dataset_v3/key_train.lst').read_text().split()
with File('earth_dataset_v3/faults.h5', 'r') as h5:
    scores = []
    for key in keys:
        k0 = h5[key]['k0'][64:-64, 64:-64]
        scores.append((int(((k0 >= 0) & (k0 < .02)).sum()), key))
selected = [key for score, key in sorted(scores, reverse=True)[:6]]
(root/'key_train.lst').write_text(chr(10).join(selected)+chr(10))
val = Path('earth_dataset_v3/key_val.lst').read_text().split()[:2]
(root/'key_val.lst').write_text(chr(10).join(val)+chr(10))
print(sorted(scores, reverse=True)[:6])
