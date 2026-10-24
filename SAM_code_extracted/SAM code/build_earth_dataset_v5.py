"""Copy v3 to a new v5 with the user-approved coastal label_known envelope.

No training. Refuses existing destination/staging folders. All source files are read-only.
The mask is an approved geometric assumption, not a verified survey-completeness map.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
import shutil
import numpy as np
import h5py
import geopandas as gpd
from shapely import contains_xy
from shapely.ops import unary_union
import netCDF4
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import ListedColormap

CODE=Path(__file__).resolve().parent; ROOT=CODE.parents[1]
SRC=CODE/'earth_dataset_v3'; DST=CODE/'earth_dataset_v5'; STAGE=CODE/'.earth_dataset_v5.building'
PREVIEW=CODE/'diagnostics/v5_coastal_envelope_preview_v1/preview_recipe.json'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(4*1024*1024),b''):h.update(data)
    return h.hexdigest()

def main():
    if DST.exists() or STAGE.exists():raise FileExistsError('Destination or staging exists; refusing overwrite')
    preview=json.loads(PREVIEW.read_text()); recipe=preview['recipe']
    files=[p for p in SRC.iterdir() if p.is_file()]
    before={p.name:digest(p) for p in files}
    for name,expected in preview['source_hashes'].items():
        p=ROOT/'Qfaults_GIS/SHP'/name if name.endswith('.shp') else SRC/name
        assert digest(p)==expected, f'Changed preview source: {name}'
    man=json.loads((SRC/'manifest.json').read_text())
    metadata={t['key']:t for t in man['tiles']}
    lists={s:(SRC/f'key_{s}.lst').read_text().split() for s in ['train','val','test']}
    splits={k:s for s,ks in lists.items() for k in ks}
    assert len(splits)==sum(map(len,lists.values()))
    T=man['provenance']['tile'];res=man['provenance']['res_deg']
    first=man['tiles'][0];i,j=map(int,first['key'].split('_')[1:])
    x0=first['lon_c']-((T-1)/2+j)*res;y0=first['lat_c']-((T-1)/2+i)*res
    H=max(int(k.split('_')[1])+T for k in metadata);W=max(int(k.split('_')[2])+T for k in metadata)
    z=np.full((H,W),np.nan,np.float32);used=np.zeros((H,W),bool);p0=used.copy()
    terrain_overlap_conflicts=0
    with h5py.File(SRC/'faults.h5','r') as f:
        assert set(f)==set(metadata)
        for k in sorted(splits):
            i,j=map(int,k.split('_')[1:]);g=f[k];a=g['image'][:]
            region=z[i:i+T,j:j+T];seen=np.isfinite(region)
            terrain_overlap_conflicts+=int(np.count_nonzero(region[seen]!=a[seen]))
            region[:]=a;used[i:i+T,j:j+T]=True;p0[i:i+T,j:j+T]|=g['k0'][:]<.01
    assert terrain_overlap_conflicts==0
    valid=used&np.isfinite(z)&(z>-2147483647)
    g=gpd.read_file(ROOT/'Qfaults_GIS/SHP/ca_offshore.shp').to_crs(3310)
    shape=unary_union(g.geometry.buffer(recipe['buffer_km']*1000))
    shape=shape.buffer(recipe['closing_km']*1000).buffer(-recipe['closing_km']*1000)
    shape=gpd.GeoSeries([shape],crs=3310).to_crs(4326).iloc[0]
    xs=x0+np.arange(W)*res;ys=y0+np.arange(H)*res
    inside=np.zeros((H,W),bool)
    for start in range(0,H,128):
        xx,yy=np.meshgrid(xs,ys[start:start+128]);inside[start:start+128]=contains_xy(shape,xx,yy)
    known=inside&valid&(z<0)
    assert known.sum()==preview['candidate_pixels']
    assert (known&p0).sum()==preview['positive_pixels_inside_candidate']
    assert not (known & ~valid).any() and not (known&(z>=0)).any()
    print('Approved envelope reproduced:',int(known.sum()),'pixels',flush=True)
    STAGE.mkdir(); QA=STAGE/'qa';QA.mkdir()
    for p in files:
        if p.name!='manifest.json':shutil.copy2(p,STAGE/p.name)
    shutil.copy2(SRC/'manifest.json',STAGE/'manifest_source_v3.json')
    shutil.copy2(PREVIEW,STAGE/'approved_preview_recipe.json')
    rows=[]
    with h5py.File(STAGE/'faults.h5','r+') as f:
        for k in sorted(f):
            i,j=map(int,k.split('_')[1:]);group=f[k];mask=known[i:i+T,j:j+T].astype('uint8')
            assert mask.shape==group['image'].shape==(T,T)
            assert 'label_known' not in group
            ds=group.create_dataset('label_known',data=mask,compression='gzip',compression_opts=4)
            ds.attrs['meaning']='1=approved approximate supervision region; 0=unknown/excluded'
            ds.attrs['coverage_basis']='user_approved_geometric_envelope_v1_not_verified_survey_coverage'
            old_split=str(group.attrs.get('split',''))
            group.attrs['source_v3_split_attribute']=old_split
            group.attrs['split']=splits.get(k,'excluded_buffer')
            pos=group['k0'][:]<.01; center=mask[64:320,64:320]
            rows.append(dict(key=k,split=splits.get(k,'excluded_buffer'),known_pixels=int(mask.sum()),
                known_positive_pixels=int((pos&(mask>0)).sum()),known_background_pixels=int((~pos&(mask>0)).sum()),
                center_known_pixels=int(center.sum()),center_known_positive_pixels=int((pos[64:320,64:320]&(center>0)).sum())))
        f.attrs['dataset_version']='earth_dataset_v5'
        f.attrs['supervision_key']='label_known'
        f.attrs['supervision_basis']='approved geometric coastal envelope; not verified survey completeness'
    print('New HDF5 written. Verifying every original array and mask...',flush=True)
    reconstructed=np.zeros((H,W),bool);coverage=np.zeros((H,W),bool)
    pair_count=0
    with h5py.File(SRC/'faults.h5','r') as src,h5py.File(STAGE/'faults.h5','r') as dst:
        for k in src:
            a,b=src[k],dst[k];i,j=map(int,k.split('_')[1:]);m=b['label_known'][:]
            assert set(b)==set(a)|{'label_known'} and m.dtype==np.uint8 and np.isin(m,[0,1]).all()
            for name in a:np.testing.assert_array_equal(a[name][:],b[name][:])
            np.testing.assert_array_equal(m,known[i:i+T,j:j+T])
            assert abs(float(b.attrs['lon_c'])-(x0+(j+(T-1)/2)*res))<1e-9
            assert abs(float(b.attrs['lat_c'])-(y0+(i+(T-1)/2)*res))<1e-9
            seen=coverage[i:i+T,j:j+T];previous=reconstructed[i:i+T,j:j+T]
            np.testing.assert_array_equal(previous[seen],m[seen])
            previous[:]=m;seen[:]=True
        np.testing.assert_array_equal(reconstructed,known)
        for x,k in enumerate(sorted(splits)):
            i,j=map(int,k.split('_')[1:])
            for q in sorted(splits)[x+1:]:
                ii,jj=map(int,q.split('_')[1:])
                pair_count+=int(abs(i-ii)<T and abs(j-jj)<T)
        # Independent geographic anchor: compare actual raster pixel centers and elevations to GEBCO.
        ncpath=ROOT/'GEBCO_California/gebco_2026_n50.0_s19.0_w-129.0_e-110.0.nc'
        with netCDF4.Dataset(ncpath) as nc:
            lat=np.asarray(nc['lat'][:]);lon=np.asarray(nc['lon'][:])
            assert np.all(np.diff(lat)>0) and np.all(np.diff(lon)>0)
            base_i=int(np.argmin(abs(lat-y0)));base_j=int(np.argmin(abs(lon-x0)))
            assert abs(lat[base_i]-y0)<1e-8 and abs(lon[base_j]-x0)<1e-8
            for k in dst:
                i,j=map(int,k.split('_')[1:])
                np.testing.assert_array_equal(dst[k]['image'][:],np.asarray(nc['elevation'][base_i+i:base_i+i+T,base_j+j:base_j+j+T]))
    # Actual production loader: all active fixed-center crops preserve the new mask exactly.
    from tool import Generator_new
    loader=Generator_new(str(STAGE/'faults.h5'),sorted(splits),training=False,fixed_crop=True,supervision_key='label_known')
    with h5py.File(STAGE/'faults.h5','r') as f:
        for index,k in enumerate(sorted(splits)):
            x,d=loader[index];np.testing.assert_array_equal(d[3],f[k]['label_known'][64:320,64:320])
            assert x.shape==(7,256,256) and d.shape==(4,256,256)
    del loader
    print('Full-array, geographic and production-loader checks passed.',flush=True)
    counts={s:len(ks) for s,ks in lists.items()}
    totals={s:{'tiles':len(rs),'tiles_with_known_pixels':sum(r['known_pixels']>0 for r in rs),
        'tiles_with_zero_known_pixels':sum(r['known_pixels']==0 for r in rs),
        'fixed_center_crops_with_zero_known_pixels':sum(r['center_known_pixels']==0 for r in rs)}
        for s in ['train','val','test','excluded_buffer'] for rs in [[r for r in rows if r['split']==s]]}
    source_after={p.name:digest(p) for p in files};assert source_after==before
    for s in ['train','val','test']:
        assert digest(STAGE/f'key_{s}.lst')==before[f'key_{s}.lst']
    provenance=dict(man['provenance'])
    provenance['source_dataset']='earth_dataset_v3';provenance['source_provenance']=man['provenance']
    provenance.update(dataset_version='earth_dataset_v5',created=datetime.now(timezone.utc).isoformat(),
        counts={**counts,'excluded_buffer':len(rows)-len(splits)},n_tiles=len(rows),active_tiles=len(splits),
        supervision=dict(key='label_known',recipe=recipe,basis='user_approved_approximate_coastal_envelope',
            verified_survey_completeness=False,positive_threshold_d0=.01))
    newtiles=[]
    rowmap={r['key']:r for r in rows}
    for t in man['tiles']:
        newtiles.append({**t,'source_v3_manifest_split':t['split'],'split':splits.get(t['key'],'excluded_buffer'),
            'known_pixels':rowmap[t['key']]['known_pixels']})
    (STAGE/'manifest.json').write_text(json.dumps(dict(provenance=provenance,tiles=newtiles),indent=2),encoding='utf-8')
    with (QA/'tile_mask_statistics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report=dict(status='PASS',source=str(SRC),destination=str(DST),source_files_sha256_before=before,
        source_files_unchanged=True,all_original_arrays_identical=True,original_arrays_checked=len(rows)*6,
        h5_groups=len(rows),label_known_groups=len(rows),mask_dtype='uint8',mask_shape=[T,T],
        binary_values_only=True,all_geographic_centers_match=True,all_images_match_local_GEBCO=True,
        latitude_row_direction='south_to_north',overlap_consistency=True,active_overlapping_tile_pairs=pair_count,
        fixed_center_production_loader_checks=len(splits),split_files_unchanged=True,
        unique_known_pixels=int(known.sum()),unique_known_positive_pixels=int((known&p0).sum()),
        unique_known_background_pixels=int((known&~p0).sum()),unique_used_pixels=int(valid.sum()),
        retained_marine_positive_pixels=int((known&p0).sum()),excluded_nonnegative_elevation_positive_pixels=int((p0&valid&(z>=0)).sum()),
        global_mask_sha256=hashlib.sha256(known.astype('uint8').tobytes()).hexdigest(),split_statistics=totals,
        preview_pixel_totals_match=True,training_launched=False,
        limitations=['Approved geometric approximation, not verified negative survey coverage.',
                    'Original train/validation geographic overlap retained; no new split.',
                    'Zero-known tiles/crops remain in original lists; do not train without --supervisionKey label_known.'])
    (QA/'validation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    # Inspect representative active tiles: north, central coast, south, land-only.
    selected=[]
    eligible=[r for r in rows if r['split']!='excluded_buffer' and r['known_pixels']>0]
    for lat,lon in [(40,-124.2),(38,-123),(36.5,-122),(34.4,-120.7),(32.8,-117.8)]:
        options=sorted(eligible,key=lambda r:(metadata[r['key']]['lat_c']-lat)**2+(metadata[r['key']]['lon_c']-lon)**2)
        selected.append(next(r['key'] for r in options if r['key'] not in selected))
    selected.append('t_1408_1344')
    font=FontProperties(fname='/mnt/c/Windows/Fonts/msyh.ttc')
    with h5py.File(STAGE/'faults.h5','r') as f:
        fig,axs=plt.subplots(len(selected),3,figsize=(13,4*len(selected)))
        for rr,k in enumerate(selected):
            gr=f[k];m=gr['label_known'][:].astype(bool);pos=gr['k0'][:]<.01;terrain=gr['TW'][0]
            a=float(gr.attrs['lon_c']);b=float(gr.attrs['lat_c']);ext=[a-T*res/2,a+T*res/2,b-T*res/2,b+T*res/2]
            labels=np.zeros((T,T),np.uint8);labels[m&~pos]=1;labels[m&pos]=2;labels[~m&pos]=3
            axs[rr,0].imshow(terrain,origin='lower',extent=ext,cmap='gray')
            overlay=np.zeros((T,T,4));overlay[pos]=[1,.05,.05,.8]
            axs[rr,0].imshow(overlay,origin='lower',extent=ext)
            axs[rr,1].imshow(m,origin='lower',extent=ext,cmap=ListedColormap(['#e4e5e7','#60b977']),vmin=0,vmax=1)
            axs[rr,2].imshow(labels,origin='lower',extent=ext,cmap=ListedColormap(['#e4e5e7','#b8d9e9','#d93d45','#f4aa30']),vmin=0,vmax=3)
            for col,title in enumerate(['地形与原始断层标签','label_known：绿色=1','灰=未知 蓝=监督背景 红=监督断层 橙=忽略断层']):
                axs[rr,col].set_title(title,fontproperties=font,fontsize=10)
                axs[rr,col].set_xlabel('Longitude');axs[rr,col].set_ylabel('Latitude')
            axs[rr,0].text(0,1.14,f'{k} / {splits[k]} / known={m.mean():.1%}',transform=axs[rr,0].transAxes,fontsize=12)
        fig.tight_layout(h_pad=3);fig.savefig(QA/'alignment_contact_sheet.png',dpi=130);plt.close(fig)
    shutil.copy2(CODE/'diagnostics/v5_coastal_envelope_preview_v1/coastal_envelope_overview.png',QA/'approved_envelope_overview.png')
    (QA/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>earth_dataset_v5 对齐检查</title><style>body{font:17px sans-serif;max-width:1400px;margin:30px auto;padding:20px}img{width:100%}</style><h1>earth_dataset_v5 掩膜对齐检查</h1><p>已按用户确认的沿岸几何范围添加 label_known。原始地形和断层标签不变。此范围是几何近似，未宣称完整调查覆盖。</p><p><a href="validation_report.json">验证报告</a> · <a href="tile_mask_statistics.csv">逐图监督统计</a></p><img src="approved_envelope_overview.png"><h2>北段、中段、南段及陆地对照</h2><img src="alignment_contact_sheet.png">',encoding='utf-8')
    (STAGE/'README_label_known.txt').write_text('earth_dataset_v5：从 earth_dataset_v3 独立复制，全部 339 个图块新增 label_known。\n'
        '保留 train=161、val=17、test=38；123 个未使用图块仍保留，未加入训练。\n'
        '已确认范围：断层线 EPSG:3310 外扩 8 km，4 km 闭合，排除非负高程和原使用范围外像素。\n'
        '这是一版用户确认的几何近似监督范围，不是已核实的完整调查区域。\n'
        '训练必须同时指定 --dataset earth_dataset_v5/key_train.lst --supervisionKey label_known。\n'
        '不要只替换数据路径而遗漏 supervisionKey。掩膜 0 是忽略，不是负类。\n'
        '原始 k0/k1/k2 和输入数组未修改；所有原列表保持字节一致，尚未剔除无监督图块。\n'
        '原训练/验证地理重叠仍保留。本次没有训练模型或启动四组实验。\n'
        'manifest_source_v3.json 保留原始元数据；新 manifest 和 HDF5 split 属性按实际名单更新。\n'
        'qa/index.html 查看对齐图；qa/validation_report.json 查看完整检查结果。\n',encoding='utf-8')
    # Keep a reproducible copy of this builder and verify source is unchanged at publication.
    shutil.copy2(Path(__file__),QA/'build_earth_dataset_v5.py')
    assert not DST.exists()
    os.rename(STAGE,DST)
    print(json.dumps({k:report[k] for k in ['status','h5_groups','unique_known_pixels','unique_known_positive_pixels','unique_known_background_pixels','split_statistics']},indent=2))
    print('Published:',DST)

if __name__=='__main__':main()
