"""
set_k1_eq_k0.py  —  one-time dataset tweak for the "stronger" run.
The dip-slip channel k1 had too few labels and did not learn (AP 0.08, all-black
predictions). Here we set k1 = k0 (any fault) so BOTH output fault channels are
trained on fault PRESENCE, concentrating all capacity on the task that works.
Run once, in the venv, from inside "SAM code":
    python set_k1_eq_k0.py
"""
import argparse
import h5py
from project_paths import DATA_H5, resolve_input, require_file

ap = argparse.ArgumentParser()
ap.add_argument('--h5', default=str(DATA_H5), help='dataset to modify in place')
ap.add_argument('--yes', action='store_true', help='confirm the destructive in-place edit')
args = ap.parse_args()
H5 = require_file(resolve_input(args.h5), 'HDF5 dataset')
if not args.yes:
    raise SystemExit(f'Refusing to modify {H5} without --yes')
with h5py.File(H5, "a") as f:
    n = 0
    for k in list(f.keys()):
        g = f[k]
        if "k0" in g and "k1" in g:
            k0 = g["k0"][:]
            del g["k1"]
            g.create_dataset("k1", data=k0, compression="gzip", compression_opts=4)
            n += 1
    print(f"set k1 = k0 in {n} tiles of {H5}")
