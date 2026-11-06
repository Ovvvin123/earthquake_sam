"""Independent pointwise geometry and geographic partition audit; no training."""
from pathlib import Path
import json
import shutil
import h5py
import numpy as np
import geopandas as gpd
from pyproj import Transformer
from shapely import STRtree, points, contains_xy, union_all, buffer, box
from build_earth_dataset_v7 import DST, V6, ROOT, CODE, metric, footprints, dump, digest


def main():
    data=DST
    manifest=json.loads((data/'manifest.json').read_text())
    original=gpd.read_file(data/'provenance/usgs_land_faults.geojson').to_crs(4326)
    old_recipe=json.loads((V6/'approved_preview_recipe.json').read_text())
    assert len(original)==old_recipe['retained_source_feature_count']
    reference_hashes={}
    for relative,expected in old_recipe['source_sha256'].items():
        if relative.startswith('Qfaults_GIS'):
            p=ROOT/relative.replace('\\','/')
            reference_hashes[str(p)]=digest(p)
            assert reference_hashes[str(p)]==expected
    for name in ('key_train.lst','key_val.lst','key_test.lst','approved_preview_recipe.json'):
        reference_hashes[str(V6/name)]=digest(V6/name)
        shutil.copy2(V6/name,data/'provenance'/('v6_'+name))
    dump(data/'provenance/reference_checksums.json',reference_hashes)
    projected=original.to_crs(3310)
    angular_index=STRtree(original.geometry.to_numpy())
    metric_index=STRtree(projected.geometry.to_numpy())
    env=union_all([union_all(buffer(projected.geometry.to_numpy()[i:i+256],12000,quad_segs=16))
                   for i in range(0,len(projected),256)])
    fps=footprints(); active=metric(union_all(list(fps.values()))); test=metric(fps['test'])
    ll=Transformer.from_crs(3310,4326,always_xy=True)
    rng=np.random.default_rng(20261030)
    worst_m=worst_deg=0.; sampled=0; mismatch=0
    totals={s:dict(tiles=0,valid_pixels=0,known_pixels=0,trace_pixels=0,positive_pixels=0) for s in ('train','val','test')}
    with h5py.File(data/'faults.h5','r') as f:
        for i,t in enumerate(manifest['tiles']):
            rect=box(*t['bounds'])
            assert active.covers(rect)
            if t['split']=='test': assert test.buffer(-1000).covers(rect)
            else: assert not test.buffer(1000).intersects(rect)
            stats=totals[t['split']]; stats['tiles']+=1
            for k in ('valid_pixels','known_pixels','trace_pixels','positive_pixels'): stats[k]+=t[k]
            g=f[t['key']]; valid=g['valid_input'][:].astype(bool)
            idx=np.flatnonzero(valid); take=rng.choice(idx,size=min(64,len(idx)),replace=False)
            rows,cols=np.divmod(take,1024)
            x=t['bounds'][0]+(cols+.5)*30
            y=t['bounds'][3]-(rows+.5)*30
            lon,lat=ll.transform(x,y)
            im,dm=metric_index.query_nearest(points(x,y),all_matches=False,return_distance=True)
            ia,da=angular_index.query_nearest(points(lon,lat),all_matches=False,return_distance=True)
            dm=dm[np.argsort(im[0])]; da=da[np.argsort(ia[0])]
            stored_m=g['fault_distance_m'][:].flat[take]
            stored_a=g['k0'][:].flat[take]
            err_m=np.abs(stored_m-dm); eligible=da<.3499
            err_a=np.abs(stored_a[eligible]-da[eligible])
            assert err_m.max()<=7.51
            assert not eligible.any() or err_a.max()<=.0000501
            worst_m=max(worst_m,float(err_m.max()))
            if len(err_a): worst_deg=max(worst_deg,float(err_a.max()))
            expected=contains_xy(env,x,y)
            stored=g['label_known'][:].flat[take].astype(bool)
            mismatch+=int(np.count_nonzero(expected!=stored)); sampled+=len(take)
            assert np.array_equal(expected,stored)
            if (i+1)%50==0: print('Geometry audit',i+1,'/',len(manifest['tiles']),flush=True)
    source_name=V6/'qa/validation_report.json'
    for stats in totals.values():
        stats['valid_land_km2']=stats['valid_pixels']*.0009
        stats['supervised_land_km2']=stats['known_pixels']*.0009
        stats['unknown_valid_land_km2']=(stats['valid_pixels']-stats['known_pixels'])*.0009
    source_report=json.loads(source_name.read_text())
    build=json.loads((data/'qa/validation_report.json').read_text())
    assert build['source_sha256'][str(V6/'faults.h5')]==source_report['dataset_hdf5_sha256']
    report=dict(status='PASS',sampled_valid_pixel_centers=sampled,mask_point_mismatches=mismatch,
        sampled_max_angular_distance_error_degrees=worst_deg,sampled_max_metric_distance_error_m=worst_m,
        exact_vector_distance_used_as_independent_reference=True,
        tile_footprints_all_inside_v6_active_extent=True,original_v6_test_region_preserved=True,
        train_val_excluded_from_original_test_region_plus_1km=True,
        v6_hdf5_matches_original_verified_checksum=True,split_totals=totals,training_started=False)
    report.update(usgs_source_matches_v6_approved_catalog=True,land_clipped_source_features=len(original))
    dump(data/'qa/geometry_audit.json',report)
    build['independent_geometry_audit']='PASS'
    dump(data/'qa/validation_report.json',build)
    shutil.copy2(Path(__file__),data/'provenance'/Path(__file__).name)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
