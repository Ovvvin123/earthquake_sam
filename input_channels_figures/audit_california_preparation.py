"""Read-only audit of California raw data and existing v3 HDF5; writes JSON only."""
from pathlib import Path
import json
from collections import Counter
import numpy as np
import netCDF4, h5py, geopandas as gpd
from scipy.ndimage import distance_transform_edt

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'SAM_code_extracted/SAM code/earth_dataset_v3'
man=json.loads((DATA/'manifest.json').read_text())
info={t['key']:t for t in man['tiles']}
prov=man['provenance']
lists={s:(DATA/f'key_{s}.lst').read_text().split() for s in ['train','val','test','test_unseen']}
lists['all']=(DATA/'key.lst').read_text().split()
report={'provenance':prov,'lists':{s:len(v) for s,v in lists.items()}}
ncfile=ROOT/'GEBCO_California/gebco_2026_n50.0_s19.0_w-129.0_e-110.0.nc'
with netCDF4.Dataset(ncfile) as nc:
    lat=np.asarray(nc['lat'][:]);lon=np.asarray(nc['lon'][:])
    r=float(abs(lat[1]-lat[0])); b=prov['bbox']
    ii=np.where((lat>=b[1])&(lat<=b[3]))[0];jj=np.where((lon>=b[0])&(lon<=b[2]))[0]
    z=np.asarray(nc['elevation'][ii[0]:ii[-1]+1,jj[0]:jj[-1]+1],dtype='float32')
    latw=lat[ii];lonw=lon[jj]
    report['local_nc']={'file':str(ncfile),'shape':list(nc['elevation'].shape),'variables':list(nc.variables),'lat_first_last':[float(lat[0]),float(lat[-1])],'lon_first_last':[float(lon[0]),float(lon[-1])],'resolution_deg':r,'bbox_window_shape':list(z.shape),'global_attrs':{k:str(nc.getncattr(k)) for k in nc.ncattrs()}}
g=gpd.read_file(ROOT/'Qfaults_GIS/SHP/ca_offshore.shp').to_crs(4326)
g=g.cx[b[0]:b[2],b[1]:b[3]]
report['catalog']={'features_in_bbox':len(g),'columns':list(g.columns),'slip_sense':g['SLIP_SENSE'].fillna('MISSING').value_counts().to_dict(),'age':g['FLT_AGE'].fillna('MISSING').value_counts().to_dict(),'dip_slip_feature_count':int(g['SLIP_SENSE'].fillna('').str.contains('N|R').sum())}
pts=[]
for geom in g.geometry:
    for ln in (geom.geoms if geom.geom_type.startswith('Multi') else [geom]):
        xy=np.asarray(ln.coords)
        for (x0,y0),(x1,y1) in zip(xy[:-1],xy[1:]):
            n=max(2,int(np.hypot(x1-x0,y1-y0)/(r*.5))+1)
            pts.append(np.column_stack([np.linspace(x0,x1,n),np.linspace(y0,y1,n)]))
pts=np.vstack(pts)
rows={}; k0_verified=0; k0_maxerr=0.; image_exact=0; image_maxerr=0.; ctx_exact=0
with h5py.File(DATA/'faults.h5','r') as f:
    report['h5']={'groups':len(f),'first_group':next(iter(f)),'datasets':{name:{'shape':list(ds.shape),'dtype':str(ds.dtype)} for name,ds in f[next(iter(f))].items()}}
    for key in f:
        group=f[key]; a=group['image'][:]; c=group['CTX'][:]; k0=group['k0'][:];k1=group['k1'][:]
        i,j=map(int,key.split('_')[1:]); T=a.shape[0]
        raw=z[i:i+T,j:j+T]
        if raw.shape==a.shape:
            err=float(np.max(np.abs(a-raw)));image_maxerr=max(image_maxerr,err);image_exact+=int(np.array_equal(a,raw))
            lo,hi=np.percentile(raw[np.isfinite(raw)],[2,98]);hi=hi if hi>lo else lo+1
            ctx=(np.clip((raw-lo)/(hi-lo),0,1)*255).astype('float32')
            ctx_exact+=int(np.allclose(c,ctx,atol=1e-4))
        mk=np.zeros_like(a,dtype=bool)
        sel=(pts[:,0]>=lonw[j])&(pts[:,0]<=lonw[j+T-1])&(pts[:,1]>=latw[i])&(pts[:,1]<=latw[i+T-1])
        p=pts[sel];px=((p[:,0]-lonw[j])/r).round().astype(int);py=((p[:,1]-latw[i])/r).round().astype(int)
        good=(px>=0)&(px<T)&(py>=0)&(py<T);mk[py[good],px[good]]=True
        if mk.any():
            rebuilt=distance_transform_edt(~mk)*r;rebuilt[rebuilt>=.35]=999.;rebuilt=rebuilt.astype('float32')
        else: rebuilt=np.full_like(a,999.)
        err=float(np.max(np.abs(k0-rebuilt)));k0_maxerr=max(k0_maxerr,err);k0_verified+=int(np.array_equal(k0,rebuilt))
        valid=np.isfinite(a)&(a>-2147483647)
        rows[key]={'positive_elevation_fraction':float(np.mean(a[valid]>0)),'negative_elevation_fraction':float(np.mean(a[valid]<0)),'invalid_pixels':int((~valid).sum()),'min':float(a[valid].min()),'max':float(a[valid].max()),'k0_fault_pixels':int((k0==0).sum()),'k1_fault_pixels':int((k1==0).sum()),'k1_equals_k0':bool(np.array_equal(k0,k1))}
report['checks']={'image_tiles_exact_match_local_nc':image_exact,'image_max_abs_error_m':image_maxerr,'ctx_tiles_match_percentile_conversion':ctx_exact,'k0_tiles_exact_match_rebuilt_usgs':k0_verified,'k0_max_abs_error':k0_maxerr,'test_unseen_subset_of_test':set(lists['test_unseen'])<=set(lists['test'])}
report['split_summary']={}
for s,ks in lists.items():
    report['split_summary'][s]={'tiles':len(ks),'tiles_with_fault_pixels':sum(rows[k]['k0_fault_pixels']>0 for k in ks),'tiles_marked_positive_ge40pixels':sum(info[k]['is_positive'] for k in ks),'mean_positive_elevation_fraction':float(np.mean([rows[k]['positive_elevation_fraction'] for k in ks])),'tiles_with_positive_elevation':sum(rows[k]['positive_elevation_fraction']>0 for k in ks),'all_pixels_positive_tiles':sum(rows[k]['positive_elevation_fraction']==1 for k in ks),'tiles_with_k1_fault_pixels':sum(rows[k]['k1_fault_pixels']>0 for k in ks),'k1_equals_k0_tiles':sum(rows[k]['k1_equals_k0'] for k in ks),'invalid_pixels_total':sum(rows[k]['invalid_pixels'] for k in ks),'elevation_min':min(rows[k]['min'] for k in ks),'elevation_max':max(rows[k]['max'] for k in ks)}
report['footprint_overlap_pairs']={}
ext=prov['tile']*prov['res_deg']
for s,t in [('train','test'),('val','test'),('train','val')]:
    report['footprint_overlap_pairs'][s+'_'+t]=sum(abs(info[a]['lat_c']-info[b]['lat_c'])<ext and abs(info[a]['lon_c']-info[b]['lon_c'])<ext for a in lists[s] for b in lists[t])
report['note']='Elevation > 0 is a proxy for land content, not a coastline-based land/sea classification. Pixel proportions count overlapping tiles repeatedly. No training files were modified.'
out=ROOT/'input_channels_figures/california_preparation_audit_20260930.json'
out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
