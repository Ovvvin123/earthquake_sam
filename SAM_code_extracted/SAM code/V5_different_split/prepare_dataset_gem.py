"""Build all non-overlapping terrain tiles; split separately with make_random_split.py."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import h5py
import netCDF4
import geopandas as gpd
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion
from matplotlib.colors import LightSource
from grid_core import BASE, local_path, windows


def stretch(a, valid):
    out = np.zeros(a.shape, np.float32)
    if valid.any():
        lo, hi = np.percentile(a[valid], [2, 98])
        out[valid] = np.clip((a[valid]-lo)/max(float(hi-lo), 1e-6), 0, 1)*255
    return out


def raster(g, lat, lon, res):
    mask = np.zeros((len(lat), len(lon)), bool)
    for geom in g.geometry:
        if geom is None or geom.is_empty:
            continue
        for line in (geom.geoms if geom.geom_type == 'MultiLineString' else [geom]):
            xy = np.asarray(line.coords)[:, :2]
            for a, b in zip(xy[:-1], xy[1:]):
                count = max(2, int(np.linalg.norm(b-a)/(res*.5))+1)
                pts = np.linspace(a, b, count)
                x = np.rint((pts[:, 0]-lon[0])/res).astype(int)
                y = np.rint((pts[:, 1]-lat[0])/res).astype(int)
                ok = (x >= 0)&(x < len(lon))&(y >= 0)&(y < len(lat))
                mask[y[ok], x[ok]] = True
    return mask


def distance(mask, res, cap):
    if not mask.any():
        return np.full(mask.shape, 999, np.float32)
    d = (distance_transform_edt(~mask)*res).astype(np.float32)
    d[d >= cap] = 999
    return d


def build(args):
    if args.tile != args.stride or args.tile < 2:
        raise ValueError('Non-overlapping coverage requires stride == tile >= 2')
    out = local_path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Use a new empty output directory: {out}')
    minlon, minlat, maxlon, maxlat = args.bbox
    if minlon >= maxlon or minlat >= maxlat:
        raise ValueError('Invalid bbox')
    with netCDF4.Dataset(args.nc) as ds:
        lat = np.asarray(ds['lat'][:]); lon = np.asarray(ds['lon'][:])
        iy = np.flatnonzero((lat >= minlat)&(lat <= maxlat))
        ix = np.flatnonzero((lon >= minlon)&(lon <= maxlon))
        if min(len(ix), len(iy)) < 2:
            raise ValueError('Region must contain at least 2x2 source pixels')
        raw = ds['elevation'][iy[0]:iy[-1]+1, ix[0]:ix[-1]+1]
        z = np.ma.asarray(raw, dtype=np.float32).filled(np.nan)
        lat, lon = lat[iy], lon[ix]
    res = float(lat[1]-lat[0])
    if res <= 0 or not np.allclose(np.diff(lat), res) or not np.allclose(np.diff(lon), res):
        raise ValueError('Expected ascending regular GEBCO latitude/longitude grid')
    valid = np.isfinite(z)
    if not valid.any():
        raise ValueError('No valid elevation in region')
    # Nearest valid fill is only for derivatives; invalid pixels never become labels/inputs.
    nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = z[tuple(nearest)]
    grad = np.hypot(*np.gradient(filled))
    safe = binary_erosion(valid, border_value=1)
    threshold = np.percentile(grad[safe if safe.any() else valid], args.edge_pctl)
    g = gpd.read_file(args.shp).to_crs(4326).cx[minlon:maxlon, minlat:maxlat]
    if 'slip_type' in g.columns:
        dip = g[g.slip_type.fillna('').str.contains('Normal|Reverse|Thrust', case=False)]
    elif 'SLIP_SENSE' in g.columns:
        dip = g[g.SLIP_SENSE.fillna('').str.contains('N|R')]
    else:
        dip = g.iloc[:0]
    fault = raster(g, lat, lon, res)
    fault_d = distance(fault, res, args.maxdist)
    other = (grad > threshold)&safe&(fault_d > .03)
    labels = [fault_d, distance(raster(dip, lat, lon, res), res, args.maxdist),
              distance(other, res, args.maxdist)]
    # Derivatives precede tiling/padding: no synthetic ridge along padded tile edges.
    shades = [LightSource(azdeg=az, altdeg=45).hillshade(
        filled, vert_exag=.001, dx=res*111000, dy=res*111000) for az in (315, 75, 195)]
    out.mkdir(parents=True, exist_ok=True)
    tiles, skipped = [], []
    with h5py.File(out/'faults.h5', 'x') as h5:
        for i, j, h, w in windows(*z.shape, args.tile):
            key = f't_{i}_{j}'
            v = valid[i:i+h, j:j+w]
            if not v.any():
                skipped.append(dict(key=key, row=i, col=j, height=h, width=w, reason='no_valid_pixels'))
                continue
            def pad(a, fill=0):
                b = np.full((*a.shape[:-2], args.tile, args.tile), fill, dtype=a.dtype)
                b[..., :h, :w] = a
                return b
            ctx = stretch(z[i:i+h, j:j+w], v)
            depth = np.where(v, z[i:i+h, j:j+w], 0).astype(np.float32)
            tw = np.stack([stretch(a[i:i+h, j:j+w], v) for a in shades])
            arrays = dict(CTX=pad(ctx), image=pad(depth), TW=pad(tw), valid=pad(v.astype(np.uint8)))
            for n, a in enumerate(labels):
                arrays[f'k{n}'] = pad(np.where(v, a[i:i+h, j:j+w], 999).astype(np.float32), 999)
            nfp = int((fault[i:i+h, j:j+w]&v).sum())
            meta = dict(key=key, split='unassigned', row=i, col=j, height=h, width=w,
                lat_c=float((lat[i]+lat[i+h-1])/2), lon_c=float((lon[j]+lon[j+w-1])/2),
                bbox=[float(lon[j]-res/2), float(lat[i]-res/2),
                      float(lon[j+w-1]+res/2), float(lat[i+h-1]+res/2)],
                valid_pixels=int(v.sum()), valid_fraction=float(v.sum()/args.tile**2),
                n_fault_px=nfp, is_positive=nfp > 0,
                dense_positive=nfp >= args.min_fault_px)
            group = h5.create_group(key)
            for name, a in arrays.items():
                group.create_dataset(name, data=a, compression='gzip', compression_opts=4)
            for name, value in meta.items():
                group.attrs[name] = value
            tiles.append(meta)
    provenance = dict(created=datetime.now(timezone.utc).isoformat(),
        nc=str(Path(args.nc).resolve()), shp=str(Path(args.shp).resolve()), bbox=args.bbox,
        shape=list(z.shape), res_deg=res, tile=args.tile, stride=args.stride,
        n_tiles=len(tiles), valid_pixels=int(valid.sum()), counts={'unassigned': len(tiles)},
        edge_policy='pad_invalid', negative_sampling='keep_all',
        label_distance='full_region_degrees', maxdist=args.maxdist,
        positive_definition='at least one catalog fault pixel on valid terrain',
        split='unassigned')
    (out/'manifest.json').write_text(json.dumps(dict(provenance=provenance, tiles=tiles,
        skipped=skipped), indent=2), encoding='utf-8')
    (out/'key.lst').write_text(''.join(t['key']+'\n' for t in tiles), encoding='utf-8')
    print(f'Built {len(tiles)} tiles; skipped {len(skipped)} empty tiles; shape={z.shape}; {out}')


def main():
    root = BASE.parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nc', default=str(root/'GEBCO_Global/GEBCO_2026_global/GEBCO_2026.nc'))
    p.add_argument('--shp', default=str(root/'Qfaults_GIS/SHP/ca_offshore.shp'))
    p.add_argument('--bbox', nargs=4, type=float, default=[-125, 31.5, -116.5, 41])
    p.add_argument('--out', default='datasets/ca_stride384')
    p.add_argument('--tile', type=int, default=384)
    p.add_argument('--stride', type=int, default=384)
    p.add_argument('--maxdist', type=float, default=.35)
    p.add_argument('--edge_pctl', type=float, default=96)
    p.add_argument('--min_fault_px', type=int, default=40, help='statistics only; never drops a tile')
    build(p.parse_args())

if __name__ == '__main__':
    main()
