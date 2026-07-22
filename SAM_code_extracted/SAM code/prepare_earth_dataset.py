"""
prepare_earth_dataset.py  —  publication-grade dataset builder
==================================================================
Adapts the Mars SAM fault-detection pipeline (trainV10.py / tool.py
Generator_new) to Earth seafloor, with the rigour needed for a paper:

  * Inputs that emphasise lineaments (multi-azimuth hillshade), not raw grayscale
  * Fault TYPING from the USGS slip-sense attribute (not a forced normal/reverse
    scheme that does not exist offshore)
  * An explicit "other-landform" hard-negative channel (k2) to fight the
    documented fault-vs-other-landform confusion (Guo & Jiang, 2025)
  * Hard-negative tile mining (rugged, fault-free seafloor)
  * Spatially-blocked train / val / test split (NO leakage between adjacent,
    overlapping tiles — required for honest generalisation numbers)
  * Full provenance metadata stored per tile and in a manifest

Label channels (each = angular distance in degrees to nearest feature,
                 capped at 0.35 deg -> 999, matching the Mars convention):
  k0 : ANY fault                       (presence; primary target)
  k1 : faults with a DIP-SLIP component (slip_sense contains 'N' or 'R')
  k2 : OTHER landforms (strong bathymetric scarps/ridges FAR from any fault)

Input channels written per tile (consumed by Generator_new as 7 channels:
  CTX, image(mole), valid_0, valid_1, TW[0..2]):
  CTX   : robust-stretched bathymetry, 0-255            (nodata = -2147483647)
  image : raw bathymetry in metres (the "mole" channel)  (nodata sentinel)
  TW    : 3x multi-azimuth hillshade (az 315/75/195), 0-255

Stored tiles are 384 px; Generator_new random-crops 256 (so tiles MUST be >256).
"""
import argparse
import os, json, numpy as np, h5py, netCDF4 as nc, geopandas as gpd, warnings
from datetime import datetime
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from matplotlib.colors import LightSource
from project_paths import CODE_DIR, GEBCO_FILE, QFAULTS_FILE, resolve_input, resolve_output, require_file
warnings.filterwarnings('ignore')

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument('--nc', default=str(GEBCO_FILE), help='regional GEBCO netCDF file')
ap.add_argument('--shp', default=str(QFAULTS_FILE), help='fault catalog shapefile')
ap.add_argument('--out', default=str(CODE_DIR / 'earth_dataset_local'), help='output dataset directory')
args = ap.parse_args()

NC = require_file(resolve_input(args.nc), 'GEBCO netCDF file')
SHP = require_file(resolve_input(args.shp), 'fault shapefile')
OUTDIR = resolve_output(args.out)
OUTDIR.mkdir(parents=True, exist_ok=True)
H5 = OUTDIR / 'faults.h5'

# ---- parameters ----
TILE   = 384            # stored tile size (px); must be > CROP
CROP   = 256            # crop size used at training time (--mn 256)
STRIDE = 64             # tile stride (px); heavy overlap -> more samples
MAXDIST_DEG = 0.35      # cap, matches Mars (>=0.35 -> 999)
NODATA = -2147483647
MIN_FAULT_PIX = 40      # tile is "positive" if >= this many fault px
HILLSHADE_AZ = [315, 75, 195]      # 3 illumination directions, 120 deg apart
OTHER_EDGE_PCTL = 96    # gradient percentile defining "strong" terrain edges
OTHER_MIN_DIST_DEG = 0.03  # an "other landform" edge must be this far from any fault
NEG_RUGGED_PCTL = 60    # hard negatives: keep fault-free tiles above this ruggedness
RNG = np.random.default_rng(42)

# spatially-blocked split: three CONTIGUOUS latitude blocks. A tile is kept
# only if its FULL latitude footprint lies inside one block; tiles that cross a
# block boundary are dropped (this is the no-leakage guard).
TEST_LAT_MAX  = 35.0    # test  = tiles entirely south of this
TRAIN_LAT     = (35.0, 38.0)  # train = tiles entirely within this band
VAL_LAT_MIN   = 38.0    # val   = tiles entirely north of this

print("Loading GEBCO ...")
d = nc.Dataset(NC)
lat = d.variables['lat'][:]; lon = d.variables['lon'][:]
res = float(abs(lat[1]-lat[0]))
print(f"  res {res:.6f} deg/px ({res*3600:.1f}\")")

print("Loading faults ...")
g = gpd.read_file(SHP).to_crs(4326)
g_dip = g[g['SLIP_SENSE'].fillna('').str.contains('N|R')]   # dip-slip component
print(f"  all faults {len(g)} | dip-slip-component {len(g_dip)}")

def densify(gdf, step):
    pts=[]
    for geom in gdf.geometry:
        if geom is None: continue
        lines = geom.geoms if geom.geom_type.startswith('Multi') else [geom]
        for ln in lines:
            xy=np.asarray(ln.coords)
            for (x0,y0),(x1,y1) in zip(xy[:-1],xy[1:]):
                seg=max(2,int(np.hypot(x1-x0,y1-y0)/step)+1)
                pts.append(np.column_stack([np.linspace(x0,x1,seg),np.linspace(y0,y1,seg)]))
    return np.vstack(pts) if pts else np.empty((0,2))
pts_all = densify(g, res*0.5)
pts_dip = densify(g_dip, res*0.5)
print(f"  densified pts: all {len(pts_all)}  dip {len(pts_dip)}")

# ---- working window = fault bbox + margin ----
fb=g.total_bounds; m=0.3
i0g,i1g = np.where((lat>=fb[1]-m)&(lat<=fb[3]+m))[0][[0,-1]]
j0g,j1g = np.where((lon>=fb[0]-m)&(lon<=fb[2]+m))[0][[0,-1]]
Z = d.variables['elevation'][i0g:i1g, j0g:j1g].astype('float32')
latw = lat[i0g:i1g]; lonw = lon[j0g:j1g]
print(f"  window {Z.shape}  lat {latw.min():.2f}..{latw.max():.2f}  lon {lonw.min():.2f}..{lonw.max():.2f}")

# ---- global terrain-edge statistics (for "other landforms" + ruggedness) ----
gy,gx = np.gradient(np.nan_to_num(Z))
grad = np.hypot(gx,gy)
edge_thr = np.percentile(grad, OTHER_EDGE_PCTL)
print(f"  edge threshold (p{OTHER_EDGE_PCTL}) = {edge_thr:.3f}")

# global fault distance (deg) for the whole window, to define "far from fault"
fault_full = np.zeros(Z.shape, bool)
pxa=((pts_all[:,0]-lonw[0])/res).round().astype(int)
pya=((pts_all[:,1]-latw[0])/res).round().astype(int)
ok=(pxa>=0)&(pxa<Z.shape[1])&(pya>=0)&(pya<Z.shape[0])
fault_full[pya[ok],pxa[ok]]=True
dist_fault_full = distance_transform_edt(~fault_full)*res
other_full = (grad>edge_thr) & (dist_fault_full>OTHER_MIN_DIST_DEG)   # strong edges, not faults
print(f"  'other landform' pixels: {other_full.mean()*100:.1f}% of window")

ls = LightSource()
def u8(a):
    v=a.astype('float32'); f=np.isfinite(v)
    if f.sum()<4: return np.zeros_like(v)
    lo,hi=np.percentile(v[f],[2,98]); hi=hi if hi>lo else lo+1
    o=np.clip((v-lo)/(hi-lo),0,1)*255; o[~f]=0; return o.astype('float32')
def dist_deg(mask):
    if mask.any():
        dd=distance_transform_edt(~mask)*res; dd[dd>=MAXDIST_DEG]=999.0; return dd.astype('float32')
    return np.full(mask.shape,999.0,'float32')
def raster(pts, ii, jj):
    mk=np.zeros((TILE,TILE),bool)
    sel=((pts[:,0]>=lonw[jj])&(pts[:,0]<=lonw[jj+TILE-1])&
         (pts[:,1]>=latw[ii])&(pts[:,1]<=latw[ii+TILE-1]))
    if sel.any():
        px=((pts[sel,0]-lonw[jj])/res).round().astype(int)
        py=((pts[sel,1]-latw[ii])/res).round().astype(int)
        o=(px>=0)&(px<TILE)&(py>=0)&(py<TILE); mk[py[o],px[o]]=True
    return mk

def split_of(ii):
    # keep a tile only if its FULL latitude footprint is inside ONE block;
    # boundary-crossing tiles are dropped -> no spatial leakage between splits.
    lo, hi = latw[ii], latw[ii+TILE-1]
    if hi < TEST_LAT_MAX:                       return 'test'
    if lo > VAL_LAT_MIN:                        return 'val'
    if lo > TRAIN_LAT[0] and hi < TRAIN_LAT[1]: return 'train'
    return None

# ---- build tiles ----
rugged_thr=None
# first pass to get ruggedness distribution of candidate tiles
ruggs=[]
coords=[]
for ii in range(0, Z.shape[0]-TILE, STRIDE):
    for jj in range(0, Z.shape[1]-TILE, STRIDE):
        coords.append((ii,jj))
        sub=grad[ii:ii+TILE,jj:jj+TILE]; ruggs.append(float(np.nanmean(sub)))
rugged_thr=np.percentile(ruggs, NEG_RUGGED_PCTL)
print(f"  candidate tiles {len(coords)}  ruggedness p{NEG_RUGGED_PCTL}={rugged_thr:.3f}")

manifest=[]; counts={'train':0,'val':0,'test':0}; pos=0; neg=0
with h5py.File(H5,'w') as f:
    for (ii,jj),rg in zip(coords,ruggs):
        sp=split_of(ii)
        if sp is None: continue                      # boundary-guard drop
        fm_all=raster(pts_all,ii,jj); nfp=int(fm_all.sum())
        is_pos = nfp>=MIN_FAULT_PIX
        if not is_pos:
            # keep only HARD negatives (rugged, fault-free); plus a few easy ones
            if rg<rugged_thr and RNG.random()>0.15: continue
        tile=Z[ii:ii+TILE,jj:jj+TILE]
        valid=np.isfinite(tile)
        CTX=u8(tile); CTX[~valid]=NODATA
        image=np.where(valid,tile,NODATA).astype('float32')
        # multi-azimuth hillshade (3 illumination directions, 120 deg apart)
        TW=np.stack([u8(LightSource(azdeg=az,altdeg=45).hillshade(
                        np.nan_to_num(tile),vert_exag=0.001,dx=res*111000,dy=res*111000))
                     for az in HILLSHADE_AZ]).astype('float32')
        k0=dist_deg(fm_all)
        k1=dist_deg(raster(pts_dip,ii,jj))
        k2=dist_deg(other_full[ii:ii+TILE,jj:jj+TILE])
        key=f"t_{ii}_{jj}"
        grp=f.create_group(key)
        for nm,arr in [('CTX',CTX),('image',image),('TW',TW),('k0',k0),('k1',k1),('k2',k2)]:
            grp.create_dataset(nm,data=arr,compression='gzip',compression_opts=4)
        latc=float((latw[ii]+latw[ii+TILE-1])/2); lonc=float((lonw[jj]+lonw[jj+TILE-1])/2)
        grp.attrs.update(dict(lat_c=latc,lon_c=lonc,split=sp,n_fault_px=nfp,
                              is_positive=bool(is_pos),ruggedness=float(rg)))
        manifest.append(dict(key=key,split=sp,lat_c=latc,lon_c=lonc,
                             n_fault_px=nfp,is_positive=bool(is_pos),ruggedness=float(rg)))
        counts[sp]+=1; pos+=is_pos; neg+=(not is_pos)

# ---- write split lists + manifest + provenance ----
for sp in ['train','val','test']:
    ks=[m['key'] for m in manifest if m['split']==sp]
    with open(OUTDIR / f"key_{sp}.lst",'w') as fk: fk.write('\n'.join(ks)+('\n' if ks else ''))
with open(OUTDIR / "key.lst",'w') as fk:
    fk.write('\n'.join(m['key'] for m in manifest)+'\n')
prov=dict(created=datetime.utcnow().isoformat()+'Z',
          bathymetry="GEBCO 2026 15-arcsec", fault_catalog="USGS ca_offshore.shp (Qfaults 2020)",
          n_tiles=len(manifest), counts=counts, n_positive=int(pos), n_negative=int(neg),
          tile_px=TILE, crop_px=CROP, stride_px=STRIDE, res_deg=res,
          label_def=dict(k0="any fault", k1="dip-slip-component fault (slip_sense ~ N|R)",
                         k2="other landform (strong edge >%.3f deg from any fault)"%OTHER_MIN_DIST_DEG,
                         cap_deg=MAXDIST_DEG, nodata_value=999),
          input_channels=dict(CTX="bathymetry 0-255", image="bathymetry metres (std-normalised at load)",
                              TW="multi-azimuth hillshade az=%s"%HILLSHADE_AZ),
          split=dict(method="contiguous latitude blocks; boundary-crossing tiles dropped (no leakage)",
                     test="lat < %.1f"%TEST_LAT_MAX, train="%.1f < lat < %.1f"%TRAIN_LAT,
                     val="lat > %.1f"%VAL_LAT_MIN))
json.dump({'provenance':prov,'tiles':manifest}, open(OUTDIR / "manifest.json",'w'), indent=2)

print(f"\nDONE  tiles={len(manifest)}  train/val/test={counts}  pos={pos} neg={neg}")
print(f"  -> {H5}")
print(f"  -> key_train.lst / key_val.lst / key_test.lst / manifest.json")
