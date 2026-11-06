"""
prepare_dataset_gem.py  —  GENERAL dataset builder (any region, any fault catalog)
=================================================================================
Builds a SAM training dataset from:
  * a GEBCO netCDF bathymetry/elevation file (regional tile OR the global file)
  * a fault catalog shapefile (GEM gem_active_faults_harmonized.shp, or USGS
    ca_offshore.shp, or any LineString catalog with a slip-type column)

Improvements over the California-only version ("new method", data side):
  * region selected by --bbox (works on the 4 GB global GEBCO file by slicing)
  * auto-detects the slip column: 'slip_type' (GEM, English) or 'SLIP_SENSE'
    (USGS, letter codes); k1 = dip-slip component (Normal/Reverse/Thrust)
  * 2-D CHECKERBOARD block split (train/val/test) instead of 3 latitude bands,
    keeping ~70/15/15 while staying leakage-free (boundary-crossing tiles dropped)
  * multi-azimuth hillshade input, k0/k1/k2 labels, hard-negative mining (kept)

Example (California region from the global GEBCO file):
  python prepare_dataset_gem.py \
     --nc GEBCO_2026.nc --shp gem-global-active-faults/shapefile/gem_active_faults_harmonized.shp \
     --bbox -125 32 -114 42 --out earth_dataset_gem
"""
import os, json, argparse, numpy as np, h5py, netCDF4 as nc, geopandas as gpd, warnings
from datetime import datetime
from scipy.ndimage import distance_transform_edt
from matplotlib.colors import LightSource
warnings.filterwarnings('ignore')

ap = argparse.ArgumentParser()
ap.add_argument('--nc', required=True, help='GEBCO netCDF (regional or global)')
ap.add_argument('--shp', required=True, help='fault catalog shapefile')
ap.add_argument('--bbox', nargs=4, type=float, required=True,
                metavar=('MINLON','MINLAT','MAXLON','MAXLAT'))
ap.add_argument('--out', default='earth_dataset_gem')
ap.add_argument('--tile', type=int, default=384)
ap.add_argument('--stride', type=int, default=64)
ap.add_argument('--maxdist', type=float, default=0.35)
ap.add_argument('--min_fault_px', type=int, default=40)
ap.add_argument('--block_deg', type=float, default=2.0, help='checkerboard block size (deg)')
ap.add_argument('--edge_pctl', type=float, default=96.0)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--no_split', action='store_true',
                help='keep ALL tiles as train (multi-region builds: split is region-level)')
args = ap.parse_args()

NODATA = -2147483647
HILLSHADE_AZ = [315, 75, 195]
OTHER_MIN_DIST_DEG = 0.03
RNG = np.random.default_rng(args.seed)
os.makedirs(args.out, exist_ok=True)
H5 = f"{args.out}/faults.h5"
minlon, minlat, maxlon, maxlat = args.bbox

print(f"Region bbox: lon {minlon}..{maxlon}  lat {minlat}..{maxlat}")
print("Loading GEBCO (slicing to bbox) ...")
d = nc.Dataset(args.nc)
lat = d.variables['lat'][:]; lon = d.variables['lon'][:]
res = float(abs(lat[1]-lat[0]))
im = np.where((lat >= minlat) & (lat <= maxlat))[0]
jm = np.where((lon >= minlon) & (lon <= maxlon))[0]
i0g, i1g = im[0], im[-1]+1; j0g, j1g = jm[0], jm[-1]+1
Z = d.variables['elevation'][i0g:i1g, j0g:j1g].astype('float32')
latw = lat[i0g:i1g]; lonw = lon[j0g:j1g]
print(f"  window {Z.shape}  res {res:.6f} deg")

print("Loading faults (clip to bbox) ...")
g = gpd.read_file(args.shp).to_crs(4326).cx[minlon:maxlon, minlat:maxlat]
# detect slip column + dip-slip subset
if 'slip_type' in g.columns:        # GEM (English words)
    dip = g[g['slip_type'].fillna('').str.contains('Normal|Reverse|Thrust', case=False)]
elif 'SLIP_SENSE' in g.columns:     # USGS ca_offshore (letter codes)
    dip = g[g['SLIP_SENSE'].fillna('').str.contains('N|R')]
else:
    dip = g.iloc[0:0]
print(f"  faults {len(g)} | dip-slip {len(dip)}")
if len(g) == 0:
    raise SystemExit("No faults in bbox — check --bbox / --shp.")

def densify(gdf, step):
    pts = []
    for geom in gdf.geometry:
        if geom is None: continue
        lines = geom.geoms if geom.geom_type.startswith('Multi') else [geom]
        for ln in lines:
            xy = np.asarray(ln.coords)
            for (x0,y0),(x1,y1) in zip(xy[:-1], xy[1:]):
                seg = max(2, int(np.hypot(x1-x0, y1-y0)/step)+1)
                pts.append(np.column_stack([np.linspace(x0,x1,seg), np.linspace(y0,y1,seg)]))
    return np.vstack(pts) if pts else np.empty((0,2))
pts_all = densify(g, res*0.5)
pts_dip = densify(dip, res*0.5)

# global terrain edges -> "other landform" hard-negative signal
gy, gx = np.gradient(np.nan_to_num(Z)); grad = np.hypot(gx, gy)
edge_thr = np.percentile(grad, args.edge_pctl)
fault_full = np.zeros(Z.shape, bool)
if len(pts_all):
    pxa = ((pts_all[:,0]-lonw[0])/res).round().astype(int)
    pya = ((pts_all[:,1]-latw[0])/res).round().astype(int)
    ok = (pxa>=0)&(pxa<Z.shape[1])&(pya>=0)&(pya<Z.shape[0]); fault_full[pya[ok],pxa[ok]] = True
other_full = (grad > edge_thr) & (distance_transform_edt(~fault_full)*res > OTHER_MIN_DIST_DEG)

def u8(a):
    v=a.astype('float32'); f=np.isfinite(v)
    if f.sum()<4: return np.zeros_like(v)
    lo,hi=np.percentile(v[f],[2,98]); hi=hi if hi>lo else lo+1
    o=np.clip((v-lo)/(hi-lo),0,1)*255; o[~f]=0; return o.astype('float32')
def dist_deg(mask):
    if mask.any():
        dd=distance_transform_edt(~mask)*res; dd[dd>=args.maxdist]=999.0; return dd.astype('float32')
    return np.full(mask.shape,999.0,'float32')
def raster(pts, ii, jj, T):
    mk=np.zeros((T,T),bool)
    if len(pts)==0: return mk
    sel=((pts[:,0]>=lonw[jj])&(pts[:,0]<=lonw[jj+T-1])&(pts[:,1]>=latw[ii])&(pts[:,1]<=latw[ii+T-1]))
    if sel.any():
        px=((pts[sel,0]-lonw[jj])/res).round().astype(int); py=((pts[sel,1]-latw[ii])/res).round().astype(int)
        o=(px>=0)&(px<T)&(py>=0)&(py<T); mk[py[o],px[o]]=True
    return mk

# --- latitude-band split (robust for coastal fault strips) ---
# The region is divided by latitude into: test (south) | train (middle) | val (north).
# A tile is kept only if its FULL latitude footprint lies inside one band; tiles
# crossing a band boundary are dropped (the no-leakage guard). This guarantees each
# split gets a continuous slab of coastline even when faults hug the shore.
_lat_lo_all, _lat_hi_all = float(latw.min()), float(latw.max())
_span = _lat_hi_all - _lat_lo_all
TEST_MAX  = _lat_lo_all + 0.20*_span     # south 20% -> test
TRAIN_MAX = _lat_lo_all + 0.80*_span     # middle 60% -> train ; north 20% -> val
def tile_split(ii, jj, T):
    if args.no_split:
        # v6 multi-region mode: train/val/test are WHOLE regions (handled by
        # build_multiregion.py), so inside one region nothing is dropped.
        return 'train'
    lo, hi = latw[ii], latw[ii+T-1]
    if hi < TEST_MAX:                 return 'test'
    if lo > TRAIN_MAX:                return 'val'
    if lo > TEST_MAX and hi < TRAIN_MAX: return 'train'
    return None   # crosses a band boundary -> drop (no leakage)

T = args.tile
ls = LightSource()
manifest=[]; counts={'train':0,'val':0,'test':0}; pos=0; neg=0
# ruggedness threshold for hard-negative selection
ruggs=[float(np.nanmean(grad[ii:ii+T,jj:jj+T]))
       for ii in range(0,Z.shape[0]-T,args.stride) for jj in range(0,Z.shape[1]-T,args.stride)]
rug_thr=np.percentile(ruggs,60) if ruggs else 0

with h5py.File(H5,'w') as f:
    for ii in range(0, Z.shape[0]-T, args.stride):
        for jj in range(0, Z.shape[1]-T, args.stride):
            sp = tile_split(ii,jj,T)
            if sp is None: continue
            fm = raster(pts_all,ii,jj,T); nfp=int(fm.sum()); is_pos = nfp>=args.min_fault_px
            rg = float(np.nanmean(grad[ii:ii+T,jj:jj+T]))
            if not is_pos and rg<rug_thr and RNG.random()>0.15: continue   # keep hard negatives
            tile=Z[ii:ii+T,jj:jj+T]; valid=np.isfinite(tile)
            CTX=u8(tile); CTX[~valid]=NODATA
            image=np.where(valid,tile,NODATA).astype('float32')
            TW=np.stack([u8(LightSource(azdeg=az,altdeg=45).hillshade(
                          np.nan_to_num(tile),vert_exag=0.001,dx=res*111000,dy=res*111000))
                         for az in HILLSHADE_AZ]).astype('float32')
            k0=dist_deg(fm); k1=dist_deg(raster(pts_dip,ii,jj,T)); k2=dist_deg(other_full[ii:ii+T,jj:jj+T])
            key=f"t_{ii}_{jj}"; grp=f.create_group(key)
            for nm,arr in [('CTX',CTX),('image',image),('TW',TW),('k0',k0),('k1',k1),('k2',k2)]:
                grp.create_dataset(nm,data=arr,compression='gzip',compression_opts=4)
            latc=float((latw[ii]+latw[ii+T-1])/2); lonc=float((lonw[jj]+lonw[jj+T-1])/2)
            grp.attrs.update(dict(lat_c=latc,lon_c=lonc,split=sp,n_fault_px=nfp,is_positive=bool(is_pos)))
            manifest.append(dict(key=key,split=sp,lat_c=latc,lon_c=lonc,n_fault_px=nfp,is_positive=bool(is_pos)))
            counts[sp]+=1; pos+=is_pos; neg+=(not is_pos)

for sp in ['train','val','test']:
    ks=[m['key'] for m in manifest if m['split']==sp]
    open(f"{args.out}/key_{sp}.lst",'w').write('\n'.join(ks)+('\n' if ks else ''))
open(f"{args.out}/key.lst",'w').write('\n'.join(m['key'] for m in manifest)+'\n')
json.dump({'provenance':dict(created=datetime.utcnow().isoformat()+'Z',nc=args.nc,shp=args.shp,
          bbox=args.bbox,n_tiles=len(manifest),counts=counts,n_pos=int(pos),n_neg=int(neg),
          res_deg=res,tile=T,stride=args.stride,block_deg=args.block_deg,
          label=dict(k0='any fault',k1='dip-slip (Normal/Reverse/Thrust)',k2='other landform')),
          'tiles':manifest}, open(f"{args.out}/manifest.json",'w'), indent=2)
print(f"\nDONE  tiles={len(manifest)}  train/val/test={counts}  pos={pos} neg={neg}")
print(f"  -> {H5}  + key_train/val/test.lst + manifest.json")
