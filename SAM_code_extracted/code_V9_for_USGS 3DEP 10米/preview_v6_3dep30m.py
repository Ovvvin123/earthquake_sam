"""Four exact-extent GEBCO / actual V7 source 30 m mosaic comparisons; read only inputs."""
from pathlib import Path
import hashlib
import html
import json
import math
import shutil
from datetime import datetime, timezone

import h5py
import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import Window, from_bounds as window_from_bounds
from rasterio.transform import from_bounds
from shapely.geometry import box
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from PIL import Image

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[1]
OLD = CODE.parent / 'SAM code/diagnostics/v6_3dep10m_comparison_v1'
DATA = CODE.parent / 'SAM code/Earth_dataset_v7'
MOSAIC = ROOT / 'USGS 3DEP 30米/mosaic/usgs_3dep_30m_epsg3310.tif'
OUT = CODE / 'diagnostics/v6_3dep30m_comparison_v1'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def save_tif(path, data, transform, crs):
    with rio.open(path, 'w', driver='GTiff', width=data.shape[1], height=data.shape[0],
                  count=1, dtype='float32', crs=crs, transform=transform,
                  nodata=-999999., compress='deflate') as dst:
        dst.write(np.where(np.isfinite(data), data, -999999.).astype('float32'), 1)
        dst.update_tags(PURPOSE='Comparison preview; not a training dataset', SOURCE=str(MOSAIC))


def warp(data, transform, crs, shape, target_transform, method=Resampling.bilinear):
    target = np.full(shape, np.nan, np.float32)
    reproject(data.astype('float32'), target, src_transform=transform, src_crs=crs,
              src_nodata=np.nan, dst_transform=target_transform, dst_crs='EPSG:4326',
              dst_nodata=np.nan, resampling=method)
    return target


def hillshade(data, dx, dy):
    dzdy, dzdx = np.gradient(data, dy, dx)
    # Same northwest light, altitude45, vertical exaggeration1 in both resolutions.
    return np.clip((.5*dzdx+.5*dzdy+np.sqrt(.5))/np.sqrt(1+dzdx**2+dzdy**2), 0, 1)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'rasters').mkdir()
    old = json.loads((OLD/'report.json').read_text())
    manifest = json.loads((DATA/'manifest.json').read_text())
    assert manifest['provenance']['resolution_m'] == 30
    assert Path(manifest['provenance']['elevation_source']).name == MOSAIC.name
    sources = [MOSAIC, DATA/'faults.h5', DATA/'manifest.json', DATA/'provenance/usgs_land_faults.geojson']
    before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in sources}
    faults = gpd.read_file(DATA/'provenance/usgs_land_faults.geojson').to_crs(4326)
    records = []
    with rio.open(MOSAIC) as src, h5py.File(DATA/'faults.h5', 'r') as h5:
        assert src.crs.to_epsg() == 3310 and src.res == (30., 30.)
        for sample in old['samples']:
            name, label = sample['id'], sample['label']
            bounds = tuple(sample['bbox'])
            print('Building 30m comparison:', name, flush=True)
            coarse_path = OLD/sample['gebco_tif']
            assert sha(coarse_path) == old['artifacts_sha256'][sample['gebco_tif']]
            with rio.open(coarse_path) as gebco:
                coarse = gebco.read(1, masked=True).filled(np.nan)
                ct = gebco.transform
                assert np.allclose(tuple(gebco.bounds), bounds, atol=1e-9, rtol=0)
                assert coarse.shape == (36, 36)
            # Keep the original bbox exactly. Geographic display grid is 1 arcsecond;
            # true source and exported native window remain projected 30x30 metres.
            fine_shape = (540, 540)
            ft = from_bounds(*bounds, fine_shape[1], fine_shape[0])
            projected = transform_bounds('EPSG:4326', src.crs, *bounds, densify_pts=41)
            w = window_from_bounds(*projected, src.transform)
            c0, r0 = math.floor(w.col_off)-3, math.floor(w.row_off)-3
            c1, r1 = math.ceil(w.col_off+w.width)+3, math.ceil(w.row_off+w.height)+3
            window = Window(c0, r0, c1-c0, r1-r0)
            native = src.read(1, window=window, masked=True).filled(np.nan)
            nt = src.window_transform(window)
            fine = warp(native, nt, src.crs, fine_shape, ft)
            averaged = warp(native, nt, src.crs, coarse.shape, ct, Resampling.average)
            # Positive-elevation display convention matches the old preview;
            # it is not a newly inferred coastline or a vertical-datum correction.
            show_coarse, show_fine, show_avg = [np.where(a>0, a, np.nan) for a in (coarse, fine, averaged)]
            native_land = np.where(native>0, native, np.nan)
            fine_hill = warp(hillshade(native_land, 30., 30.), nt, src.crs, fine_shape, ft)
            latitude = (bounds[1]+bounds[3])/2
            coarse_hill = hillshade(show_coarse, ct.a*111320*np.cos(np.deg2rad(latitude)), abs(ct.e)*111320)
            # Read actual stored V7 masks, retaining split gaps and dataset exclusions.
            native_known = np.zeros(native.shape, np.float32)
            native_covered = np.zeros(native.shape, np.float32)
            used_tiles = []
            source_agreement = []
            for tile in manifest['tiles']:
                rr, cc = tile['row'], tile['col']
                y0, y1 = max(r0, rr), min(r1, rr+1024)
                x0, x1 = max(c0, cc), min(c1, cc+1024)
                if y1<=y0 or x1<=x0:
                    continue
                sr = np.s_[y0-rr:y1-rr, x0-cc:x1-cc]
                dr = np.s_[y0-r0:y1-r0, x0-c0:x1-c0]
                group = h5[tile['key']]
                native_known[dr] = group['label_known'][sr]
                native_covered[dr] = 1
                valid = group['valid_input'][sr] > 0
                stored = group['image'][sr]
                if valid.any():
                    delta = float(np.max(np.abs(stored[valid]-native[dr][valid])))
                    assert delta < 1e-4, (name, tile['key'], delta)
                    source_agreement.append(delta)
                used_tiles.append(dict(key=tile['key'], split=tile['split']))
            known = warp(native_known, nt, src.crs, fine_shape, ft, Resampling.nearest) > .5
            covered = warp(native_covered, nt, src.crs, fine_shape, ft, Resampling.nearest) > .5
            mask_path = OUT/'rasters'/f'{name}_v7_label_known_display.tif'
            save_tif(mask_path, known.astype('float32'), ft, 'EPSG:4326')
            save_tif(OUT/'rasters'/f'{name}_3dep_30m_native_window.tif', native, nt, src.crs)
            save_tif(OUT/'rasters'/f'{name}_3dep_30m_display.tif', fine, ft, 'EPSG:4326')
            save_tif(OUT/'rasters'/f'{name}_3dep_30m_on_v6_grid.tif', averaged, ct, 'EPSG:4326')
            shutil.copyfile(coarse_path, OUT/'rasters'/f'{name}_gebco_v6.tif')
            ext = (bounds[0], bounds[2], bounds[1], bounds[3])
            vals = np.concatenate([a[np.isfinite(a)] for a in (show_coarse, show_fine, show_avg)])
            vmin, vmax = float(vals.min()), float(vals.max())
            color = plt.get_cmap('terrain').copy(); color.set_bad('#dce5eb')
            gray = plt.get_cmap('gray').copy(); gray.set_bad('#dce5eb')
            fig, axes = plt.subplots(2, 3, figsize=(16, 11), layout='constrained')
            for ax, array, title in zip(axes[0], (show_coarse, show_fine, show_avg),
                    ('GEBCO / V6: 15 arc-sec', 'USGS 3DEP: actual 30 m mosaic', '30 m 3DEP averaged to V6 grid')):
                im = ax.imshow(array, origin='upper', extent=ext, cmap=color,
                               vmin=vmin, vmax=vmax, interpolation='nearest')
                ax.set_title(title, fontsize=12)
            fig.colorbar(im, ax=list(axes[0]), label='Elevation (m); shared scale; source vertical datums', shrink=.85, pad=.02)
            for ax, array, title in [(axes[1,0], coarse_hill, 'GEBCO hillshade'),
                                     (axes[1,1], fine_hill, '30 m 3DEP hillshade')]:
                ax.imshow(array, origin='upper', extent=ext, cmap=gray, vmin=0, vmax=1, interpolation='nearest')
                ax.set_title(title, fontsize=12)
            ax = axes[1,2]
            ax.imshow(show_fine, origin='upper', extent=ext, cmap=color, vmin=vmin, vmax=vmax, interpolation='nearest')
            local = faults.loc[faults.intersects(box(*bounds))].copy()
            local.geometry = local.geometry.intersection(box(*bounds))
            if len(local):
                local.plot(ax=ax, color='#df1747', linewidth=.85)
            xs = ft.c+(np.arange(fine_shape[1])+.5)*ft.a
            ys = ft.f+(np.arange(fine_shape[0])+.5)*ft.e
            if known.any() and not known.all():
                ax.contour(xs, ys, known, levels=[.5], colors=['#00c9db'], linewidths=.8)
            ax.set_title('USGS land faults (red) / V7 12 km mask (cyan)', fontsize=10.5)
            for ax in axes.flat:
                ax.set(xlim=ext[:2], ylim=ext[2:], xlabel='Longitude', ylabel='Latitude')
                ax.set_aspect(1/np.cos(np.deg2rad(latitude)))
                ax.xaxis.set_major_locator(MaxNLocator(5))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax.tick_params(labelsize=9)
            fig.suptitle(f'{label} | same extent as the previous comparison, north-up\n'
                         'Actual Earth_dataset_v7 source: 30 x 30 m mosaic; hillshade derived from elevation', fontsize=15)
            fig.savefig(OUT/f'{name}.png', dpi=150)
            plt.close(fig)
            records.append(dict(id=name, label=label, bbox=list(bounds), figure=f'{name}.png',
                same_extent_as_10m_preview=True, source_crs=str(src.crs), source_pixel_size_m=list(src.res),
                native_window_shape=list(native.shape), display_shape=list(fine_shape), display_crs='EPSG:4326',
                display_res_deg=[ft.a, abs(ft.e)], coarse_grid_shape=list(coarse.shape),
                shared_elevation_range=[vmin,vmax], v7_tiles=used_tiles,
                v7_stored_coverage_fraction=float(covered.mean()), known_fraction=float(known.mean()),
                dataset_source_max_difference_m=max(source_agreement) if source_agreement else None,
                fault_features=len(local)))
    for path in sources:
        assert before[str(path)] == (path.stat().st_size, path.stat().st_mtime_ns)
    for record in records:
        with Image.open(OUT/record['figure']) as img:
            img.verify()
    report = dict(status='complete', created_utc=datetime.now(timezone.utc).isoformat(),
                  source_mosaic=str(MOSAIC), source_metadata=json.loads((MOSAIC.parent/'report.json').read_text()),
                  samples=records, inputs_unchanged=True, training_started=False,
                  notes=['No vertical datum conversion; this is not a DEM accuracy/error map.',
                         '30m projected source is warped to a 1-arcsecond geographic display grid only for aligned panels.',
                         'Non-positive/missing elevation is hidden for the same display convention as the earlier comparison.',
                         'Cyan uses stored V7 label_known (12km policy), not d0 target thickness; masks may be clipped by land and tile coverage.',
                         'Hillshade uses azimuth315, altitude45, vertical exaggeration1; unlike the model-specific normalized TW channels.'])
    report['artifacts_sha256'] = {str(p.relative_to(OUT)):sha(p) for p in list(OUT.glob('*.png'))+list((OUT/'rasters').glob('*.tif'))}
    (OUT/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    cards = []
    for r in records:
        name = r['id']
        cards.append(f'<section><h2>{html.escape(r["label"])}</h2><p>与原10米预览完全相同的范围：{r["bbox"]}</p>'
                     f'<a href="{name}.png"><img src="{name}.png" alt="{html.escape(r["label"])} 30米对比"></a>'
                     f'<p><a href="{name}.png">打开原图</a> · '
                     f'<a href="rasters/{name}_3dep_30m_native_window.tif">原30米投影网格裁剪（含边缘余量）</a> · '
                     f'<a href="rasters/{name}_3dep_30m_on_v6_grid.tif">聚合至GEBCO网格</a></p>'
                     f'<p>与V7已存储有效高程核对，最大差值 {r["dataset_source_max_difference_m"]} 米。</p></section>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GEBCO与当前30米3DEP对比</title><style>body{font-family:system-ui,"Microsoft YaHei",sans-serif;max-width:1450px;margin:auto;padding:24px;background:#eef2f6;color:#183243;line-height:1.7}section{background:white;border-radius:12px;padding:24px;margin:20px 0}img{width:100%;height:auto}a{color:#176da1}.note{padding:16px;border-left:4px solid #cf922b;background:#fff6e6}h1{font-size:28px}</style>
<h1>GEBCO / 当前 USGS 3DEP 30米：同区域对比</h1>
<p>使用 Earth_dataset_v7 实际来源：EPSG:3310、30×30米大图。四处地理范围与此前10米对比图完全一致。</p>
<section><p>上排：GEBCO原15角秒高程、当前30米3DEP高程、30米3DEP聚合回GEBCO网格。每张图共用高程色标。</p>
<p>下排：两者采用相同光照参数的地形阴影，以及现有USGS陆地断层（红）和V7实际12公里监督mask边界（青）。</p>
<p>30米数据为投影坐标网格，为使各图经纬度范围严格一致，显示时重投影到1角秒网格；下载的原网格裁剪仍为30×30米。训练的1024×1024图块未改动。</p>
<p class="note">灰蓝色沿用旧预览的非正高程/缺失值隐藏方式，不是新海岸线。未做垂直基准统一，不将高程差异解释为误差。阴影由真实高度计算，不是影像，也不是训练TW通道的归一化结果。青线显示监督范围，不表示断层带宽；d0=0.005未改变12公里mask。</p></section>'''
    page += ''.join(cards) + '<p><a href="report.json">数据来源、范围、校验与处理记录</a></p><p>只生成预览；未修改训练数据、配置或启动训练。</p></html>'
    (OUT/'index.html').write_text(page, encoding='utf-8')
    print('COMPLETE', OUT, flush=True)


if __name__ == '__main__':
    main()
