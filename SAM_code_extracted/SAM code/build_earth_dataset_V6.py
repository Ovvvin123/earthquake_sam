"""Build the approved 12 km USGS land dataset in staging; never train.

Publish only after validate_earth_dataset_V6.py passes the real loader/loss checks.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import shutil

import geopandas as gpd
import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt
from shapely import box, buffer, contains_xy, get_coordinates, length, segmentize, union_all
from preview_v6_land_envelope import land_geometry
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[1]
SRC = CODE / 'earth_dataset_v5'
DST = CODE / 'earth_dataset_V6'
STAGE = CODE / '.earth_dataset_V6.building'
PREVIEW = CODE / 'diagnostics/v6_land_envelope_preview_v2_8_12_15km/preview_recipe.json'
CORE = ['tool.py', 'config.py', 'trainV10.py', 'aio2_online.py',
        'unimatch_fault.py', 'fault_losses.py', 'known_crops.py']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def line_parts(geom):
    if geom.geom_type in ('LineString', 'LinearRing'):
        yield geom
    elif hasattr(geom, 'geoms'):
        for part in geom.geoms:
            yield from line_parts(part)


def main():
    if DST.exists() or STAGE.exists():
        raise FileExistsError('Destination/staging exists; refusing overwrite')
    preview = json.loads(PREVIEW.read_text(encoding='utf-8'))
    for name, expected in preview['source_sha256'].items():
        assert digest(ROOT / Path(name.replace('\\', '/'))) == expected, name
    before = {p.name: digest(p) for p in SRC.iterdir() if p.is_file()}
    core_before = {name: digest(CODE / name) for name in CORE}
    manifest = json.loads((SRC / 'manifest.json').read_text(encoding='utf-8'))
    metadata = {t['key']: t for t in manifest['tiles']}
    lists = {s: (SRC / f'key_{s}.lst').read_text().split() for s in ('train', 'val', 'test')}
    splits = {key: s for s, keys in lists.items() for key in keys}
    assert len(splits) == sum(map(len, lists.values())) == preview['active_tile_count']
    size = manifest['provenance']['tile']
    res = manifest['provenance']['res_deg']
    first = manifest['tiles'][0]
    ii, jj = map(int, first['key'].split('_')[1:])
    x0 = first['lon_c'] - ((size - 1) / 2 + jj) * res
    y0 = first['lat_c'] - ((size - 1) / 2 + ii) * res
    height = max(int(k.split('_')[1]) + size for k in metadata)
    width = max(int(k.split('_')[2]) + size for k in metadata)
    z = np.full((height, width), np.nan, np.float32)
    used = np.zeros(z.shape, bool)
    print('Reading active V5 footprint...', flush=True)
    with h5py.File(SRC / 'faults.h5', 'r') as f:
        assert set(f) == set(metadata)
        for key in sorted(splits):
            i, j = map(int, key.split('_')[1:])
            region = np.s_[i:i + size, j:j + size]
            image = f[key]['image'][:]
            np.testing.assert_array_equal(z[region][used[region]], image[used[region]])
            z[region] = image
            used[region] = True
    valid = used & np.isfinite(z) & (z > -2147483647)
    land = valid & (z > 0)
    land_geom = land_geometry(land, x0, y0, res)
    assert land_geom.is_valid and np.isclose(land_geom.area, land.sum() * res**2)
    assert int(land.sum()) == preview['positive_elevation_land_pixels']
    print('Clipping existing USGS fault lines to approved land footprint...', flush=True)
    catalog = gpd.read_file(ROOT / 'Qfaults_GIS/SHP/Qfaults_US_Database.shp').to_crs(4326)
    subset = catalog.iloc[catalog.sindex.query(box(*land_geom.bounds), predicate='intersects')].copy()
    subset.geometry = subset.geometry.intersection(land_geom)
    subset = subset.loc[~subset.geometry.is_empty & (length(subset.geometry.to_numpy()) > 0)]
    assert len(subset) == preview['retained_source_feature_count']
    metric = subset.to_crs(3310).geometry.to_numpy()
    pieces = []
    for start in range(0, len(metric), 256):
        pieces.append(union_all(buffer(metric[start:start + 256], 12000, quad_segs=16)))
    envelope = gpd.GeoSeries([union_all(pieces)], crs=3310).to_crs(4326).iloc[0]
    xs = x0 + np.arange(width) * res
    ys = y0 + np.arange(height) * res
    known = np.zeros(z.shape, bool)
    for start in range(0, height, 128):
        xx, yy = np.meshgrid(xs, ys[start:start + 128])
        known[start:start + 128] = contains_xy(envelope, xx, yy)
    known &= land
    assert int(known.sum()) == preview['variants']['12']['candidate_pixels']
    print(f'Approved 12 km mask reproduced: {known.sum()} pixels', flush=True)

    # Match the existing angular-distance label convention, but rasterize once
    # on the full grid so overlapping tiles have identical labels at their edges.
    trace = np.zeros(z.shape, bool)
    for geom in subset.geometry:
        for line in line_parts(geom):
            xy = get_coordinates(segmentize(line, res * .5))
            cols = np.rint((xy[:, 0] - x0) / res).astype(int)
            rows = np.rint((xy[:, 1] - y0) / res).astype(int)
            inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
            rows, cols = rows[inside], cols[inside]
            on_land = land[rows, cols]
            trace[rows[on_land], cols[on_land]] = True
    assert trace.any() and not (trace & ~known).any()
    distance = (distance_transform_edt(~trace) * res).astype(np.float32)
    distance[(distance >= .35) | ~land] = 999.
    positive = distance < .01
    assert not (positive & ~known).any()
    assert np.all(distance[~land] == 999)
    print(f'USGS-only trace pixels={trace.sum()}, d0=.01 positives={positive.sum()}', flush=True)

    STAGE.mkdir()
    qa = STAGE / 'qa'
    qa.mkdir()
    for s in lists:
        shutil.copy2(SRC / f'key_{s}.lst', STAGE / f'key_{s}.lst')
    (STAGE / 'key.lst').write_text('\n'.join(sorted(metadata)) + '\n')
    (STAGE / 'key_active.lst').write_text('\n'.join(sorted(splits)) + '\n')
    shutil.copy2(SRC / 'manifest.json', STAGE / 'manifest_source_v5.json')
    shutil.copy2(PREVIEW, STAGE / 'approved_preview_recipe.json')
    rows = []
    with h5py.File(SRC / 'faults.h5', 'r') as source, h5py.File(STAGE / 'faults.h5', 'w') as dest:
        dest.attrs.update(dataset_version='earth_dataset_V6', supervision_key='label_known',
                          supervision_domain='land_elevation_gt_zero', buffer_km=12,
                          label_source='Qfaults_US_Database.shp; land-clipped; k0 only',
                          auxiliary_channels='k1/k2=999 unused; require k0Only, isOther=False')
        for key in sorted(source):
            i, j = map(int, key.split('_')[1:])
            region = np.s_[i:i + size, j:j + size]
            a = source[key]
            b = dest.create_group(key)
            for name in ('CTX', 'image', 'TW'):
                source.copy(a[name], b, name=name)
            for name, array in [('k0', distance[region]),
                                ('k1', np.full((size, size), 999, np.float32)),
                                ('k2', np.full((size, size), 999, np.float32)),
                                ('label_known', known[region].astype(np.uint8)),
                                ('fault_trace', trace[region].astype(np.uint8))]:
                b.create_dataset(name, data=array, compression='gzip', compression_opts=4)
            for name, value in a.attrs.items():
                b.attrs[name] = value
            m, p, tr = known[region], positive[region], trace[region]
            b.attrs.update(split=splits.get(key, 'excluded_buffer'),
                           n_fault_px=int(tr.sum()), is_positive=bool(tr.sum() >= 40),
                           known_pixels=int(m.sum()), label_source='USGS land only')
            b['k0'].attrs.update(units='grid angular degrees; EDT pixels * res_deg',
                                 invalid_distance=999., positive_threshold_d0=.01)
            for name in ('k1', 'k2'):
                b[name].attrs['meaning'] = 'unused compatibility channel; all 999; not an annotation'
            b['label_known'].attrs['meaning'] = '1=approved land supervision; 0=unknown, eligible for pseudo labels if input valid'
            b['fault_trace'].attrs['meaning'] = 'Rasterized existing USGS land fault trace before d0 threshold'
            rows.append(dict(key=key, split=splits.get(key, 'excluded_buffer'),
                             known_pixels=int(m.sum()), trace_pixels=int(tr.sum()),
                             positive_pixels=int(p.sum()), known_background_pixels=int((m & ~p).sum()),
                             marine_pixels=int((a['image'][:] < 0).sum()),
                             center_known_pixels=int(m[64:320, 64:320].sum())))
    print('HDF5 written; verifying all input arrays, labels, masks and overlaps...', flush=True)
    with h5py.File(SRC / 'faults.h5', 'r') as source, h5py.File(STAGE / 'faults.h5', 'r') as dest:
        assert set(source) == set(dest)
        for key in source:
            i, j = map(int, key.split('_')[1:])
            region = np.s_[i:i + size, j:j + size]
            a, b = source[key], dest[key]
            for name in ('CTX', 'image', 'TW'):
                np.testing.assert_array_equal(a[name][:], b[name][:])
            for name, expected in [('k0', distance[region]), ('label_known', known[region]), ('fault_trace', trace[region])]:
                np.testing.assert_array_equal(b[name][:], expected)
            assert np.all(b['k1'][:] == 999) and np.all(b['k2'][:] == 999)
            assert b['label_known'].dtype == np.uint8 and b['k0'].dtype == np.float32
            assert abs(b.attrs['lon_c'] - (x0 + (j + (size - 1) / 2) * res)) < 1e-8
            assert abs(b.attrs['lat_c'] - (y0 + (i + (size - 1) / 2) * res)) < 1e-8
            raw = b['image'][:]
            excluded = ~np.isfinite(raw) | (raw <= 0)
            assert not b['label_known'][:][excluded].any()
            assert np.all(b['k0'][:][excluded] == 999)
    row_map = {r['key']: r for r in rows}
    split_stats = {s: {'tiles': len(rs),
                      'tiles_with_supervision': sum(r['known_pixels'] > 0 for r in rs),
                      'tiles_without_supervision_retained': sum(r['known_pixels'] == 0 for r in rs),
                      'center_crops_without_supervision': sum(r['center_known_pixels'] == 0 for r in rs)}
                   for s in ('train', 'val', 'test', 'excluded_buffer')
                   for rs in [[r for r in rows if r['split'] == s]]}
    tiles = [{**t, **{n: row_map[t['key']][n] for n in ('split', 'known_pixels')},
              'n_fault_px': row_map[t['key']]['trace_pixels'],
              'is_positive': row_map[t['key']]['trace_pixels'] >= 40} for t in manifest['tiles']]
    prov = dict(manifest['provenance'])
    prov.update(created=datetime.now(timezone.utc).isoformat(), dataset_version='earth_dataset_V6',
                source_dataset='earth_dataset_v5', source_provenance=manifest['provenance'],
                shp='Qfaults_GIS/SHP/Qfaults_US_Database.shp',
                n_pos=sum(t['is_positive'] for t in tiles), n_neg=sum(not t['is_positive'] for t in tiles),
                label={'k0': 'existing USGS land fault distance', 'k1': 'unused; all 999', 'k2': 'unused; all 999'},
                label_recipe={'rasterization': 'densify clipped lines to <=0.5 pixel, nearest pixel center, retain land',
                              'distance': 'global EDT * res_deg, cap >=0.35 to 999; outside active positive-elevation land=999',
                              'positive_threshold_d0': .01, 'metric_mask_distance_is_separate_from_label_distance': True},
                supervision={'key': 'label_known', 'buffer_km': 12, 'closing_km': 0,
                             'metric_crs': 'EPSG:3310', 'domain': 'active V5 footprint AND valid elevation>0',
                             'verified_survey_completeness': False},
                auxiliary_labels_disabled=True)
    (STAGE / 'manifest.json').write_text(json.dumps({'provenance': prov, 'tiles': tiles}, indent=2), encoding='utf-8')
    with (qa / 'tile_statistics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    profile = {'status': 'DATASET_ONLY_NO_TRAINING_STARTED',
               'dataset': 'earth_dataset_V6/key_train.lst', 'dataFile': 'earth_dataset_V6/faults.h5',
               'supervisionKey': 'label_known', 'k0Only': True, 'isOther': False, 'd0': .01,
               'knownCrop': False, 'fixedVal': True,
               'feedback_when_training_is_authorized': {'emaTeacher': True, 'onlineCorrection': True, 'unimatch': True},
               'notes': ['Dataset options, not a complete training/resume configuration.',
                         'Keep knownCrop off to retain wholly unknown/marine tiles in the feedback data stream.',
                         'Valid ocean may supply confidence-filtered positive pseudo labels and positive consistency; not guaranteed each epoch.',
                         'Choose new V6 output directory and configure LoRA/loss/teacher schedule separately before training.']}
    (STAGE / 'dataset_usage.json').write_text(json.dumps(profile, indent=2), encoding='utf-8')
    (STAGE / 'README_label_known.txt').write_text(
        'earth_dataset_V6：用户确认的 12 公里 USGS 陆地监督数据集。\n'
        '沿用 V5 全部 339 组输入及原 train/val/test=161/17/38 清单；123 组 excluded_buffer 不加入训练。\n'
        'CTX/image/TW 原样保留，包括海洋。k0 仅由现有 USGS 陆地断层重建；海洋及有效范围外=999。\n'
        'k0 保持旧接口的网格角度距离（全局 EDT×res_deg），d0=0.01；12 km 是监督 mask 宽度，不是断层宽度。\n'
        'k1/k2 均为 999 的未使用兼容通道，不引入海洋断层、滑动类型或地形边缘标签；必须 k0Only，isOther=False。\n'
        'label_known=1：正高程陆地断层外扩12km范围；0：未知，不计真实标签监督，不代表负类。\n'
        '仅采用高程>0作为陆地近似，不增加州界裁剪；负高程陆地和零高程被排除。\n'
        '未来伪标签/一致性训练使用完整 key_train.lst，supervisionKey=label_known；不要开启 knownCrop 排除纯海洋图块。\n'
        'dataset_usage.json 记录必要数据参数，不是完整启动命令；本次未启动训练。\n'
        '原有海洋训练继续使用 earth_dataset_v5 及其原配置，无需修改训练代码。\n'
        '沿用原空间划分及其既有地理重叠，不重新划分；几何 mask 不保证目录完整性。\n'
        'qa/index.html 查看输入/标签/mask 对齐；qa/validation_report.json 查看核验结果。\n', encoding='utf-8')
    assert before == {p.name: digest(p) for p in SRC.iterdir() if p.is_file()}
    assert core_before == {name: digest(CODE / name) for name in CORE}
    for s in lists:
        assert digest(STAGE / f'key_{s}.lst') == before[f'key_{s}.lst']
    report = dict(status='STRUCTURAL_PASS_PENDING_RUNTIME', source_files_sha256=before,
                  core_files_sha256=core_before, source_v5_unchanged=True, training_code_unchanged=True,
                  h5_groups=len(rows), active_tiles=len(splits), split_statistics=split_stats,
                  unique_known_pixels=int(known.sum()), unique_trace_pixels=int(trace.sum()),
                  unique_positive_pixels_d0_001=int(positive.sum()),
                  unique_known_background_pixels=int((known & ~positive).sum()),
                  mask_matches_approved_12km_preview=True, all_input_arrays_identical=True,
                  overlapping_tile_arrays_match_global_grid=True,
                  marine_supervised_pixels=0, marine_usgs_label_pixels=0,
                  global_mask_sha256=hashlib.sha256(known.astype('uint8').tobytes()).hexdigest(),
                  global_trace_sha256=hashlib.sha256(trace.astype('uint8').tobytes()).hexdigest(),
                  split_lists_unchanged=True, training_launched=False)
    (qa / 'validation_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    selected = []
    eligible = [r['key'] for r in rows if r['split'] != 'excluded_buffer' and r['positive_pixels'] > 0]
    for lat, lon in [(40., -123.8), (37.5, -122.), (34., -118.2)]:
        selected.append(min((k for k in eligible if k not in selected),
                            key=lambda k: (metadata[k]['lat_c'] - lat)**2 + (metadata[k]['lon_c'] - lon)**2))
    selected.append(next(r['key'] for r in rows if r['split'] == 'train' and r['marine_pixels'] == size * size))
    font_path = Path('C:/Windows/Fonts/msyh.ttc')
    if not font_path.exists(): font_path = Path('/mnt/c/Windows/Fonts/msyh.ttc')
    font = FontProperties(fname=str(font_path))
    with h5py.File(STAGE / 'faults.h5', 'r') as f:
        fig, axes = plt.subplots(len(selected), 3, figsize=(13, 15))
        for rr, key in enumerate(selected):
            gr = f[key]
            m = gr['label_known'][:].astype(bool)
            p = gr['k0'][:] < .01
            lon, lat = gr.attrs['lon_c'], gr.attrs['lat_c']
            extent = [lon - size * res / 2, lon + size * res / 2,
                      lat - size * res / 2, lat + size * res / 2]
            axes[rr, 0].imshow(gr['TW'][0], origin='lower', extent=extent, cmap='gray')
            overlay = np.zeros((size, size, 4)); overlay[p] = [1, .1, .1, .85]
            axes[rr, 0].imshow(overlay, origin='lower', extent=extent)
            axes[rr, 1].imshow(m, origin='lower', extent=extent, cmap=ListedColormap(['#eeeeee', '#60b977']), vmin=0, vmax=1)
            classes = np.zeros((size, size), np.uint8); classes[m] = 1; classes[p & m] = 2
            axes[rr, 2].imshow(classes, origin='lower', extent=extent, cmap=ListedColormap(['#eeeeee', '#b8d9e9', '#d93d45']), vmin=0, vmax=2)
            for ax, title in zip(axes[rr], ['原始地形 + USGS陆地标签', '12公里监督 mask（绿=有效）', '灰=未知 / 蓝=监督背景 / 红=断层']):
                ax.set_title(title, fontproperties=font, fontsize=10)
                ax.set(xlabel='Longitude', ylabel='Latitude')
            axes[rr, 0].text(0, 1.12, f'{key} / {splits[key]} / known={m.mean():.1%}', transform=axes[rr, 0].transAxes, fontsize=10)
        fig.tight_layout(h_pad=3)
        fig.savefig(qa / 'alignment_contact_sheet.png', dpi=145)
        plt.close(fig)
    shutil.copy2(PREVIEW.parent / 'land_mask_comparison.png', qa / 'approved_preview_comparison.png')
    (qa / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>earth_dataset_V6 核验</title><style>body{font:17px/1.7 sans-serif;max-width:1400px;margin:30px auto;padding:20px}img{width:100%}</style><h1>earth_dataset_V6 · USGS 陆地标签 + 12公里监督 mask</h1><p>原V5输入和划分保留；海洋无直接监督标签，仍保留为伪标签和一致性训练的有效输入。尚未启动训练。</p><p><a href="validation_report.json">结构核验</a> · <a href="runtime_validation.json">生产接口核验</a> · <a href="tile_statistics.csv">逐图统计</a></p><h2>地形、真实标签及监督范围</h2><img src="alignment_contact_sheet.png"><h2>此前确认的对比预览（采用中间12公里方案）</h2><img src="approved_preview_comparison.png">', encoding='utf-8')
    for path in (Path(__file__), CODE / 'preview_v6_land_envelope.py'):
        shutil.copy2(path, qa / path.name)
    print(json.dumps(report, indent=2), flush=True)
    print('Staged. Run validate_earth_dataset_V6.py to check production interfaces and publish.', flush=True)


if __name__ == '__main__':
    main()
