"""Finalize existing staging tiles after the float32 sentinel compatibility fix.

Only invalid input cells and their NoData metadata may change. All valid DEM
values, labels, masks and splits are revalidated before atomic publication.
"""
import os
import json
import shutil
import h5py
import numpy as np
import rasterio as rio
from build_earth_dataset_v7 import CODE, STAGE, DST, NODATA, CORE, dump, digest, validate, gallery
from v7_dataset import validate_readers


def main():
    assert STAGE.exists() and not DST.exists()
    manifest=json.loads((STAGE/'manifest.json').read_text())
    plan=json.loads((STAGE/'provenance/crop_plan.json').read_text())
    before=json.loads((STAGE/'provenance/source_checksums.json').read_text())
    prior=STAGE/'qa/validation_report.json'
    core_before=(json.loads(prior.read_text())['core_sha256_before'] if prior.exists()
                 else {n:digest(CODE/n) for n in CORE})
    changed_tiles=changed_cells=0
    with h5py.File(STAGE/'faults.h5','r+') as f:
        f.attrs['nodata']=float(NODATA)
        for i,t in enumerate(manifest['tiles']):
            g=f[t['key']]
            invalid=~g['valid_input'][:].astype(bool)
            if invalid.any():
                changed_tiles+=1; changed_cells+=int(invalid.sum())
                for name in ('image','CTX','TW'):
                    a=g[name][:]
                    if a.ndim==3: a[:,invalid]=NODATA
                    else: a[invalid]=NODATA
                    g[name][:]=a
            g.attrs['nodata']=float(NODATA)
            with rio.open(STAGE/'rasters'/f"{t['key']}_elevation.tif",'r+') as tif:
                if invalid.any(): tif.write(g['image'][:],1)
                tif.nodata=float(NODATA)
            if (i+1)%50==0:
                print('NoData compatibility fixed',i+1,'/',len(manifest['tiles']),flush=True)
    manifest['provenance']['inputs']['nodata']=float(NODATA)
    manifest['provenance']['inputs']['nodata_reason']='strictly below legacy threshold after float32 scalar conversion'
    dump(STAGE/'manifest.json',manifest)
    gallery(manifest)
    print('Rechecking all stored elevations, masks, rasters and splits...',flush=True)
    report=validate(plan,manifest,core_before,before)
    report['nodata_float32_compatibility_fix']=dict(nodata=float(NODATA),
        tiles_with_invalid_pixels=changed_tiles,invalid_cells=changed_cells,
        valid_elevations_unchanged=True)
    dump(STAGE/'qa/validation_report.json',report)
    runtime=validate_readers(STAGE,manifest['tiles'])
    dump(STAGE/'qa/reader_validation.json',runtime)
    report['reader_validation']='PASS'
    dump(STAGE/'qa/validation_report.json',report)
    for name in ('build_earth_dataset_v7.py','v7_dataset.py','finalize_earth_dataset_v7.py'):
        shutil.copy2(CODE/name,STAGE/'provenance'/name)
    dump(STAGE/'progress.json',dict(status='complete',done=len(manifest['tiles']),total=len(manifest['tiles'])))
    assert STAGE.resolve().parent==CODE and DST.resolve().parent==CODE and not DST.exists()
    os.rename(STAGE,DST)
    print('PUBLISHED',DST,manifest['provenance']['counts'],flush=True)


if __name__=='__main__': main()
