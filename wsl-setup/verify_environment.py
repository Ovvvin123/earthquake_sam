import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import h5py
import geopandas as gpd
import netCDF4
import torch

code = Path.cwd()
source = Path('/mnt/f/Study————————/UCLA/Earthquake-SAM_交接/SAM_code_extracted/SAM code')
def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
files = ['sam2_hiera_small.pt', 'train_v5_local/model.pth', 'earth_dataset_v3/faults.h5']
hashes = {}
for name in files:
    value = sha(code / name)
    assert value == sha(source / name), name
    hashes[name] = value
splits = {name: (code / 'earth_dataset_v3' / f'key_{name}.lst').read_text().splitlines() for name in ('train', 'val', 'test')}
assert not set(splits['train']) & set(splits['val'])
assert not set(splits['train']) & set(splits['test'])
assert not set(splits['val']) & set(splits['test'])
with h5py.File('earth_dataset_v3/faults.h5') as data:
    for keys in splits.values():
        assert all(key in data for key in keys)
root = code.parents[1]
shapes = gpd.read_file(root / 'Qfaults_GIS/SHP/ca_offshore.shp')
nc_files = list((root / 'GEBCO_California').glob('*.nc'))
assert nc_files
with netCDF4.Dataset(nc_files[0]) as nc:
    nc_variables = list(nc.variables)
report = {'python': platform.python_version(), 'kernel': platform.release(),
          'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
          'packages': {name: metadata.version(name) for name in ('torch', 'torchvision', 'SAM-2', 'numpy', 'scipy', 'h5py', 'numba', 'geopandas', 'netCDF4')},
          'copied_files_sha256': hashes, 'split_counts': {name: len(keys) for name, keys in splits.items()},
          'source_fault_count': len(shapes), 'netcdf_variables': nc_variables}
Path('wsl_validation/environment.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
