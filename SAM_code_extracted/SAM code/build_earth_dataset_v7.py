"""Build 1024-square, 30 m USGS land tiles. Never imports or launches training.

Run in the existing WSL GIS environment. --plan-only writes a reviewable plan.
New data is staged and published only after exhaustive structural validation.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing

import geopandas as gpd
import h5py
import numpy as np
import rasterio as rio
from rasterio.features import rasterize
from rasterio.windows import Window
from rasterio.transform import array_bounds
from pyproj import Transformer
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.spatial import cKDTree
from shapely import box, buffer, get_coordinates, length, segmentize, set_precision, union_all
from shapely.geometry import mapping
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from PIL import Image

from preview_v7_fov import load_v6

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[1]
V6 = CODE / 'earth_dataset_V6'
MOSAIC = ROOT / 'USGS 3DEP 30米/mosaic/usgs_3dep_30m_epsg3310.tif'
SHP = ROOT / 'Qfaults_GIS/SHP/Qfaults_US_Database.shp'
DST = CODE / 'Earth_dataset_v7'
STAGE = CODE / '.Earth_dataset_v7.building'
PLAN = CODE / 'diagnostics/v7_30m_1024_plan'
N = 1024
RES = 30
GAP = 1000  # each side of the geographic train/val boundary
NODATA = np.float32(-3000000000)  # below cutoff even after float32 scalar rounding
SPLITS = ('train', 'val', 'test')
CORE = ('tool.py','config.py','trainV10.py','aio2_online.py','unimatch_fault.py','torchUnet.py')
TO_LL = Transformer.from_crs(3310, 4326, always_xy=True)


def dump(p, v):
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def metric(g):
    return gpd.GeoSeries([segmentize(g, .02)], crs=4326).to_crs(3310).iloc[0]


def footprints():
    m = json.loads((V6/'manifest.json').read_text())
    size, r = m['provenance']['tile'], m['provenance']['res_deg']
    tiles = {t['key']: t for t in m['tiles']}
    out = {}
    for split in SPLITS:
        shapes = []
        for key in (V6/f'key_{split}.lst').read_text().split():
            t = tiles[key]
            h = size*r/2
            shapes.append(box(t['lon_c']-h,t['lat_c']-h,t['lon_c']+h,t['lat_c']+h))
        out[split] = union_all(set_precision(shapes,1e-8))
    return out


def geometry_context():
    land_ll, _, _, _, _ = load_v6()
    fps = footprints()
    active = metric(union_all(list(fps.values())))
    oldtest = metric(fps['test'])
    land = metric(land_ll)
    catalog = gpd.read_file(SHP).to_crs(4326)
    sub = catalog.iloc[catalog.sindex.query(land_ll, predicate='intersects')].copy()
    sub.geometry = sub.geometry.intersection(land_ll)
    sub = sub.loc[~sub.geometry.is_empty & (length(sub.geometry.to_numpy()) > 0)].copy()
    lines = sub.to_crs(3310)
    env = union_all([union_all(buffer(lines.geometry.to_numpy()[i:i+256],12000,quad_segs=16))
                     for i in range(0,len(lines),256)])
    return active, oldtest, land, sub, lines, env


def make_plan(active, oldtest, land):
    selected, excluded = [], []
    with rio.open(MOSAIC) as src:
        for row in range(0, src.height-N+1,N):
            for col in range(0, src.width-N+1,N):
                win = Window(col,row,N,N)
                bounds = list(rio.windows.bounds(win,src.transform))
                rect = box(*bounds)
                if not rect.intersects(land):
                    continue
                rec = dict(key=f't_{row}_{col}',row=row,col=col,bounds=bounds)
                if not active.covers(rect):
                    rec['reason']='crosses_v6_active_extent'
                    excluded.append(rec); continue
                if oldtest.buffer(-GAP).covers(rect):
                    rec['split']='test'
                elif oldtest.buffer(GAP).intersects(rect):
                    rec['reason']='crosses_original_test_region_or_1km_guard'
                    excluded.append(rec); continue
                else:
                    rec['split']='development'
                rec['legacy_land_fraction']=land.intersection(rect).area/rect.area
                if rec['legacy_land_fraction'] < .01:
                    rec['reason']='less_than_1pct_legacy_land'
                    excluded.append(rec); continue
                selected.append(rec)
    dev = [t for t in selected if t['split']=='development']
    # A northern contiguous validation holdout, determined only by geography.
    # Target ~20% of development land area before removing boundary tiles.
    thresholds = sorted(set(t['bounds'][1] for t in dev))
    candidates = []
    for cut in thresholds:
        tr = [t for t in dev if t['bounds'][3] <= cut-GAP]
        va = [t for t in dev if t['bounds'][1] >= cut+GAP]
        if len(tr)<10 or len(va)<5:
            continue
        fraction = sum(t['legacy_land_fraction'] for t in va)/sum(t['legacy_land_fraction'] for t in tr+va)
        candidates.append((abs(fraction-.2),cut,tr,va))
    _, cut, train, val = min(candidates, key=lambda x:(x[0],x[1]))
    accepted = {t['key']:s for s,ts in [('train',train),('val',val)] for t in ts}
    final = []
    for t in selected:
        if t['split']=='test':
            final.append(t)
        elif t['key'] in accepted:
            t['split']=accepted[t['key']]; final.append(t)
        else:
            t['reason']='crosses_train_val_boundary_or_1km_guard'; excluded.append(t)
    assert all(any(t['split']==s for t in final) for s in SPLITS)
    plan = dict(dataset='Earth_dataset_v7',pixel_size_m=RES,tile_size=N,stride=N,
                geographic_crs='EPSG:3310',no_overlap_even_within_splits=True,
                train_val_northing_boundary_m=cut,boundary_guard_each_side_m=GAP,
                original_test_region_preserved=True,old_train_val_repartitioned=True,
                target_val_fraction_of_development_land=.2,
                selection='Full 1024 footprint inside V6 active region; >=1% legacy land; no label-based balancing',
                tiles=final,excluded=excluded,
                counts={s:sum(t['split']==s for t in final) for s in SPLITS})
    return plan


def geojson(path, records):
    dump(path,dict(type='FeatureCollection',crs={'type':'name','properties':{'name':'EPSG:3310'}},
        features=[dict(type='Feature',geometry=mapping(box(*t['bounds'])),
                       properties={k:v for k,v in t.items() if k!='bounds'}) for t in records]))


def split_plot(plan,land,path):
    fig, ax = plt.subplots(figsize=(8,11))
    gpd.GeoSeries([land],crs=3310).plot(ax=ax,color='#ededed',edgecolor='none')
    colors=dict(train='#438dcc',val='#edaa41',test='#b46ac0')
    from matplotlib.patches import Rectangle,Patch
    for t in plan['tiles']:
        x,y,x1,y1=t['bounds']
        ax.add_patch(Rectangle((x,y),x1-x,y1-y,fc=colors[t['split']],ec='white',lw=.25,alpha=.8))
    ax.legend(handles=[Patch(color=colors[s],label=f"{s}: {plan['counts'][s]}") for s in SPLITS])
    ax.set(title='Earth_dataset_v7 | 30 m | 1024 x 1024 | disjoint footprints',
           xlabel='EPSG:3310 easting (m)',ylabel='EPSG:3310 northing (m)')
    ax.set_aspect('equal'); fig.tight_layout(); fig.savefig(path,dpi=140); plt.close(fig)


def burn(geom, shape, transform):
    if geom.is_empty:
        return np.zeros(shape, np.uint8)
    return rasterize([(geom,1)],out_shape=shape,transform=transform,dtype='uint8')


def stretch(a,valid):
    lo,hi=np.percentile(a[valid],[2,98])
    hi=max(float(hi),float(lo)+1e-8)
    out=np.rint(np.clip((a-lo)/(hi-lo),0,1)*255).astype('float32')
    out[~valid]=NODATA
    return out, [float(lo),float(hi)]


def tile_arrays(src,sid,t,land,lines,envelope,angular_tree,metric_tree):
    # Halo is used only for derivatives. It is not a model input or a label sample.
    halo=8
    win=Window(t['col']-halo,t['row']-halo,N+2*halo,N+2*halo)
    tr=src.window_transform(win)
    raw=src.read(1,window=win,boundless=True,fill_value=src.nodata)
    rect=box(*rio.windows.bounds(win,src.transform))
    lm=burn(land.intersection(rect),raw.shape,tr).astype(bool)
    base=lm & np.isfinite(raw) & (raw!=src.nodata) & (raw>0)
    valid=binary_erosion(base,iterations=1,border_value=0)
    sl=np.s_[halo:halo+N,halo:halo+N]
    vc=valid[sl]
    if vc.mean()<.01:
        return None
    # Fill only to avoid invalid-value derivative spikes; filled locations stay invalid.
    filled=raw.copy()
    if not base.all():
        inds=distance_transform_edt(~base,return_distances=False,return_indices=True)
        filled[~base]=raw[tuple(inds[:,~base])]
    ctx,stretch_range=stretch(raw[sl],vc)
    tw=[]
    for az in (315,75,195):
        shade=LightSource(azdeg=az,altdeg=45).hillshade(filled,vert_exag=.001,dx=RES,dy=RES)
        channel,_=stretch(shade[sl],vc)
        tw.append(channel)
    central=src.window_transform(Window(t['col'],t['row'],N,N))
    known=burn(envelope.intersection(rect),raw.shape,tr)[sl].astype(bool)&vc
    local=lines.iloc[lines.sindex.query(rect,predicate='intersects')]
    geoms=[g.intersection(rect) for g in local.geometry]
    geoms=[(g,1) for g in geoms if not g.is_empty]
    trace=(rasterize(geoms,out_shape=raw.shape,transform=tr,dtype='uint8',all_touched=True)[sl]
           if geoms else np.zeros((N,N),np.uint8))
    trace=trace.astype(bool)&vc&known
    rows,cols=np.nonzero(vc)
    xx=central.c+(cols+.5)*RES
    yy=central.f-(rows+.5)*RES
    lon,lat=TO_LL.transform(xx,yy)
    d=angular_tree.query(np.column_stack([lon,lat]),workers=2)[0]
    dm=metric_tree.query(np.column_stack([xx,yy]),workers=2)[0]
    k0=np.full((N,N),999,np.float32)
    k0[vc]=np.where(d<.35,d,999).astype('float32')
    distance_m=np.full((N,N),-1,np.float32); distance_m[vc]=dm
    image=np.where(vc,raw[sl],NODATA).astype('float32')
    source=sid.read(1,window=Window(t['col'],t['row'],N,N)); source[~vc]=0
    arrays=dict(image=image,CTX=ctx,TW=np.stack(tw),k0=k0,
                k1=np.full((N,N),999,np.float32),k2=np.full((N,N),999,np.float32),
                label_known=known.astype('uint8'),fault_trace=trace.astype('uint8'),
                valid_input=vc.astype('uint8'),fault_distance_m=distance_m,source_id=source)
    assert not ((k0<.01)&~known).any()
    return arrays,central,stretch_range


def export_tifs(key,a,tr):
    profile=dict(driver='GTiff',height=N,width=N,crs='EPSG:3310',transform=tr,
                 compress='deflate',tiled=True,blockxsize=256,blockysize=256)
    with rio.open(STAGE/'rasters'/f'{key}_elevation.tif','w',count=1,dtype='float32',nodata=float(NODATA),**profile) as f:
        f.write(a['image'],1)
        f.update_tags(units='metres',vertical_datum='NAVD88, unchanged from source',row_order='north_to_south')
    with rio.open(STAGE/'rasters'/f'{key}_masks.tif','w',count=4,dtype='uint8',**profile) as f:
        for j,(name,arr) in enumerate([
            ('valid_input',a['valid_input']),('label_known',a['label_known']),
            ('fault_trace',a['fault_trace']),('positive_d0_0p01',(a['k0']<.01).astype('uint8'))],1):
            f.write(arr,j); f.set_band_description(j,name)


def preview_tile(t,a):
    valid=a['valid_input'].astype(bool); known=a['label_known'].astype(bool)
    z=a['image']; shade=np.clip(a['TW'][0]/255,0,1)
    terrain=(plt.get_cmap('terrain')(np.clip(z/4000,0,1))[...,:3]*255).astype('uint8')
    terrain[~valid]=[222,231,236]
    gray=np.repeat((shade*255).astype('uint8')[...,None],3,axis=2)
    gray[~valid]=[222,231,236]
    overlay=gray.copy()
    overlay[known]=(overlay[known]*.72+np.array([46,204,113])*.28).astype('uint8')
    from scipy.ndimage import binary_dilation
    visible=binary_dilation(a['fault_trace'].astype(bool),iterations=1)&valid
    overlay[visible]=[240,45,65]
    edge=known & ~binary_erosion(known)
    overlay[edge]=[5,180,75]
    classes=np.full((N,N,3),[220,225,230],np.uint8)
    classes[valid&~known]=[170,176,182]
    classes[known]=[168,207,231]
    classes[(a['k0']<.01)&known]=[220,52,68]
    composite=np.concatenate([terrain,overlay,classes],axis=1)
    path=STAGE/'qa/tiles'/f"{t['key']}.png"
    Image.fromarray(composite).save(path,optimize=False)
    thumb=Image.fromarray(overlay); thumb.thumbnail((192,192))
    thumb.save(STAGE/'qa/thumbs'/f"{t['key']}.jpg",quality=85)


def gallery(manifest):
    records=manifest['tiles']
    js=json.dumps(records,ensure_ascii=False).replace('</','<\\/')
    ramp=(plt.get_cmap('terrain')(np.linspace(0,1,512))[None,:,:3]*255).astype('uint8')
    Image.fromarray(np.repeat(ramp,24,axis=0)).save(STAGE/'qa/elevation_legend.png')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Earth_dataset_v7 逐图图库</title>
<style>body{margin:0;background:#edf2f6;color:#223144;font:16px/1.6 system-ui,sans-serif}header,main{max-width:1600px;margin:auto;padding:20px}header{background:white}h1{margin:0;font-size:26px}.note{color:#526477}.controls{display:flex;gap:10px;flex-wrap:wrap;margin:15px 0}button,select,input{padding:9px;border:1px solid #acbac8;border-radius:6px;background:white;font:inherit}#detail{background:white;padding:16px;border-radius:10px}#preview{width:100%;display:block}#labels{display:grid;grid-template-columns:repeat(3,1fr);text-align:center;font-weight:bold;background:#eaf0f6;padding:8px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(155px,1fr));gap:12px;margin-top:24px}.card{border:2px solid transparent;cursor:pointer;background:white;border-radius:8px;padding:6px;font-size:12px}.card img{width:100%;display:block}.card.selected{border-color:#197ac1}a{color:#126eab}.metrics{margin:8px 0}#map{max-width:520px;width:100%}</style>
<header><h1>Earth_dataset_v7 · 逐图预览</h1><p>USGS 3DEP 30 米 · 1024 × 1024 · 每图 30.72 × 30.72 公里 · USGS 陆地断层 · 12 公里监督范围</p>
<p class="note">每张预览的三个面板均保留 1024 × 1024 像素。左：固定 0–4000 米地形色标；中：阴影地形、绿色监督 mask、红色断层线；右：红=断层目标，蓝=监督背景，深灰=有效但未知，浅灰=无效。阴影和 CTX 使用每图 2–98% 拉伸。</p>
<div style="display:flex;align-items:center;gap:8px">高程色标：0 m <img src="elevation_legend.png" width="256" height="12" alt="固定地形高程色标"> 4000 m（以上饱和）</div>
<p class="note">红色目标沿用 V6 的 d0=0.01 角度距离阈值，不是 12 公里宽的断层；中图红线为显示稍加粗。3DEP 海洋占位值不参与训练。接缝未自动平滑。此图库是数据预览，不是模型预测，未启动训练。</p>
<p class="note">沿海部分块状边界继承 V6 的陆地定义；并非从 30 米 DEM 新提取的精细海岸线。</p>
<a href="validation_report.json">验证报告</a> · <a href="../manifest.json">制作清单</a> · <a href="../README.txt">使用说明</a></header>
<main><details><summary>查看空间划分（训练/验证/测试图块没有重叠）</summary><img id="map" src="split_map.png"></details>
<div class="controls"><select id="split"><option value="all">全部分区</option><option>train</option><option>val</option><option>test</option></select><input id="query" placeholder="查找图块编号"><button id="prev">← 上一张</button><button id="next">下一张 →</button><span id="counter"></span></div>
<section id="detail"><h2 id="title"></h2><p id="info" class="metrics"></p><p id="links"></p><div id="labels"><span>高程（米）</span><span>USGS 断层 + 12 km mask</span><span>实际监督目标 / 未知区域</span></div><a id="full" target="_blank"><img id="preview"></a></section><div id="cards" class="grid"></div></main>
<script>const rows=__DATA__;let filtered=rows,index=0;const $=id=>document.getElementById(id);function show(i){if(!filtered.length){$('counter').textContent='无匹配结果';$('detail').style.display='none';return;} $('detail').style.display='block';index=(i+filtered.length)%filtered.length;const t=filtered[index];$('title').textContent=t.key+' · '+t.split;$('counter').textContent=(index+1)+' / '+filtered.length;$('info').textContent='中心 '+t.lon_c.toFixed(5)+', '+t.lat_c.toFixed(5)+' ｜有效地形 '+(100*t.valid_pixels/1048576).toFixed(1)+'% ｜监督范围 '+(100*t.known_pixels/1048576).toFixed(1)+'% ｜断层线像素 '+t.trace_pixels+' ｜目标像素 '+t.positive_pixels;$('preview').src='tiles/'+t.key+'.png';$('full').href=$('preview').src;$('links').innerHTML='<a href="tiles/'+t.key+'.png" target="_blank">打开原尺寸预览</a> · <a href="../rasters/'+t.key+'_elevation.tif">高程 GeoTIFF</a> · <a href="../rasters/'+t.key+'_masks.tif">四波段 mask GeoTIFF</a>';document.querySelectorAll('.card').forEach(c=>c.classList.toggle('selected',c.dataset.key===t.key));location.hash=t.key;}function filter(){filtered=rows.filter(t=>($('split').value==='all'||t.split===$('split').value)&&t.key.includes($('query').value.trim()));$('cards').replaceChildren();filtered.forEach((t,i)=>{const c=document.createElement('div');c.className='card';c.dataset.key=t.key;c.innerHTML='<img loading="lazy" src="thumbs/'+t.key+'.jpg"><b>'+t.key+'</b><br>'+t.split+' · mask '+(100*t.known_pixels/1048576).toFixed(0)+'%';c.onclick=()=>{show(i);$('detail').scrollIntoView({behavior:'smooth',block:'start'});};$('cards').appendChild(c);});show(0);}const initial=location.hash.slice(1);$('split').onchange=filter;$('query').oninput=filter;$('prev').onclick=()=>show(index-1);$('next').onclick=()=>show(index+1);document.addEventListener('keydown',e=>{if(['INPUT','SELECT'].includes(e.target.tagName))return;if(e.key==='ArrowLeft')show(index-1);if(e.key==='ArrowRight')show(index+1);});filter();const found=filtered.findIndex(t=>t.key===initial);if(found>=0)show(found);</script></html>'''
    (STAGE/'qa/index.html').write_text(page.replace('__DATA__',js),encoding='utf-8')


def validate_chunk(tiles):
    with rio.Env(GDAL_CACHEMAX=64*1024**2), h5py.File(STAGE/'faults.h5','r') as f, rio.open(MOSAIC) as src:
        for t in tiles:
            g=f[t['key']]
            # Read each compressed dataset once, rather than repeatedly decoding it.
            a={n:g[n][:] for n in g}
            valid=a['valid_input'].astype(bool); known=a['label_known'].astype(bool)
            assert all(a[n].shape==(N,N) for n in a if n!='TW')
            assert a['TW'].shape==(3,N,N)
            assert all(np.isfinite(v).all() for v in a.values())
            raw=src.read(1,window=Window(t['col'],t['row'],N,N))
            np.testing.assert_array_equal(a['image'][valid],raw[valid])
            for n in ('image','CTX'):
                assert (a[n][~valid]==NODATA).all()
            assert (a['TW'][:,~valid]==NODATA).all()
            assert not (known&~valid).any()
            assert not ((a['k0']<.01)&~known).any()
            assert not (a['fault_trace'].astype(bool)&~known).any()
            assert (a['k1']==999).all() and (a['k2']==999).all()
            with rio.open(STAGE/'rasters'/f"{t['key']}_elevation.tif") as z:
                assert z.shape==(N,N) and z.res==(30,30) and z.crs.to_epsg()==3310
                assert z.nodata==float(NODATA) and list(z.bounds)==t['bounds']
                np.testing.assert_array_equal(z.read(1),a['image'])
                tr=z.transform; crs=z.crs
            with rio.open(STAGE/'rasters'/f"{t['key']}_masks.tif") as m:
                assert m.transform==tr and m.crs==crs
                for band,name in [(1,'valid_input'),(2,'label_known'),(3,'fault_trace')]:
                    np.testing.assert_array_equal(m.read(band),a[name])
                np.testing.assert_array_equal(m.read(4),(a['k0']<.01).astype('uint8'))
            with Image.open(STAGE/'qa/tiles'/f"{t['key']}.png") as image:
                assert image.size==(3*N,N); image.verify()
    return len(tiles)


def validate(plan,manifest,core_before,source_before):
    tiles=manifest['tiles']; shapes={s:union_all([box(*t['bounds']) for t in tiles if t['split']==s]) for s in SPLITS}
    separation={}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        area=shapes[a].intersection(shapes[b]).area
        gap=shapes[a].distance(shapes[b])
        assert area==0 and gap>=GAP-1e-6
        separation[f'{a}__{b}']=dict(overlap_m2=area,min_distance_m=gap)
    assert sum(box(*t['bounds']).area for t in tiles)==union_all([box(*t['bounds']) for t in tiles]).area
    expected=set(t['key'] for t in tiles)
    with h5py.File(STAGE/'faults.h5','r') as f:
        assert set(f)==expected
    total=0
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        for count in pool.map(validate_chunk,[tiles[j::4] for j in range(4)]):
            total+=count; print('Validated',total,'/',len(tiles),flush=True)
    assert total==len(tiles)
    for s in SPLITS:
        keys=(STAGE/f'key_{s}.lst').read_text().split()
        assert set(keys)=={t['key'] for t in tiles if t['split']==s}
        assert keys and any(t['known_pixels'] and t['trace_pixels'] for t in tiles if t['split']==s)
    core_after={n:digest(CODE/n) for n in CORE}
    after={p:digest(Path(p)) for p in source_before}
    assert after==source_before, 'Source data changed during build'
    return dict(status='PASS',dataset='Earth_dataset_v7',counts=manifest['provenance']['counts'],
                tile_size=[N,N],resolution_m=RES,all_geotiffs_hdf5_aligned=True,
                all_valid_elevations_identical_to_mosaic=True,all_previews_verified=True,
                split_overlap_and_distance=separation,no_within_split_overlap=True,
                marine_direct_supervision_pixels=0,source_sha256=after,
                source_files_unchanged=True,training_code_unchanged=(core_before==core_after),
                core_sha256_before=core_before,core_sha256_after=core_after,
                training_started=False,model_constructed=False,
                hdf5_sha256=digest(STAGE/'faults.h5'))


def make_record(t,arrays,tr,stretch_range):
    record=dict(t)
    record.update(valid_pixels=int(arrays['valid_input'].sum()),known_pixels=int(arrays['label_known'].sum()),
                  trace_pixels=int(arrays['fault_trace'].sum()),positive_pixels=int((arrays['k0']<.01).sum()),
                  unknown_valid_pixels=int((arrays['valid_input']-arrays['label_known']).sum()),
                  transform=list(tr)[:6],ctx_percentile_2_98=stretch_range,
                  source_ids=[int(v) for v in np.unique(arrays['source_id']) if v])
    record['lon_c'],record['lat_c']=TO_LL.transform((t['bounds'][0]+t['bounds'][2])/2,(t['bounds'][1]+t['bounds'][3])/2)
    return record


def build(plan,ctx,resume=False):
    if DST.exists() or (STAGE.exists() and not resume):
        raise FileExistsError('Destination or staging already exists; refusing overwrite')
    if resume and not STAGE.exists():
        raise FileNotFoundError('No staging directory to resume')
    active,oldtest,land,source_lines,lines,envelope=ctx
    core_before={n:digest(CODE/n) for n in CORE}
    source_paths=[MOSAIC,V6/'faults.h5',V6/'manifest.json']+list(SHP.parent.glob(SHP.stem+'.*'))
    before={str(p):digest(p) for p in source_paths}
    expected=json.loads((MOSAIC.parent/'report.json').read_text())['qa']['output_sha256']
    assert before[str(MOSAIC)]==expected
    if resume:
        assert before==json.loads((STAGE/'provenance/source_checksums.json').read_text())
        assert plan==json.loads((STAGE/'provenance/crop_plan.json').read_text())
    for p in ('qa/tiles','qa/thumbs','rasters','provenance'):
        (STAGE/p).mkdir(parents=True,exist_ok=True)
    dump(STAGE/'provenance/crop_plan.json',plan)
    dump(STAGE/'provenance/source_checksums.json',before)
    for p in [V6/'manifest.json',MOSAIC.parent/'report.json',MOSAIC.parent/'source_manifest.json']:
        shutil.copy2(p,STAGE/'provenance'/('v6_manifest.json' if p.parent==V6 else p.name))
    source_lines.to_file(STAGE/'provenance/usgs_land_faults.geojson',driver='GeoJSON')
    angular=cKDTree(get_coordinates(segmentize(source_lines.geometry.to_numpy(),.0001)))
    metric_tree=cKDTree(get_coordinates(segmentize(lines.geometry.to_numpy(),15.)))
    records=[]; empty=[]
    _=lines.sindex
    def task(t):
        with rio.open(MOSAIC) as src, rio.open(MOSAIC.parent/'source_id_30m.tif') as sid:
            result=tile_arrays(src,sid,t,land,lines,envelope,angular,metric_tree)
        if result is None: return t,None,None
        arrays,tr,stretch_range=result
        record=make_record(t,arrays,tr,stretch_range)
        export_tifs(t['key'],arrays,tr)
        preview_tile(record,arrays)
        return t,arrays,record
    with h5py.File(STAGE/'faults.h5','a' if resume else 'w') as f:
        f.attrs.update(dataset_version='Earth_dataset_v7',pixel_size_m=RES,crs='EPSG:3310',
                       row_order='north_to_south',tile_size=N,buffer_m=12000,
                       supervision_key='label_known',k0_units='angular_degrees',nodata=float(NODATA),training_started=False)
        remaining=[]
        for t in plan['tiles']:
            key=t['key']
            complete=key in f and all((STAGE/p).exists() for p in
                (f'rasters/{key}_elevation.tif',f'rasters/{key}_masks.tif',f'qa/tiles/{key}.png',f'qa/thumbs/{key}.jpg'))
            if complete:
                g=f[key]
                # Repair early staging tiles if their sentinel was indistinguishable
                # from the legacy cutoff after NumPy float32 scalar conversion.
                if g.attrs.get('nodata') != float(NODATA):
                    invalid=~g['valid_input'][:].astype(bool)
                    for name in ('image','CTX','TW'):
                        a=g[name][:]
                        if a.ndim==3: a[:,invalid]=NODATA
                        else: a[invalid]=NODATA
                        g[name][:]=a
                    with rio.open(STAGE/'rasters'/f'{key}_elevation.tif','r+') as tif:
                        tif.write(g['image'][:],1)
                        tif.nodata=float(NODATA)
                    g.attrs['nodata']=float(NODATA)
                if 'record_json' in g.attrs:
                    record=json.loads(g.attrs['record_json'])
                else:
                    arrays={name:g[name][:] for name in g}
                    with rio.open(STAGE/'rasters'/f'{key}_elevation.tif') as tif: tr=tif.transform
                    v=arrays['valid_input'].astype(bool)
                    stretch_range=np.percentile(arrays['image'][v],[2,98]).tolist()
                    record=make_record(t,arrays,tr,stretch_range)
                records.append(record)
            else:
                if key in f: del f[key]  # incomplete temporary group only
                remaining.append(t)
        print('Resumed completed tiles:',len(records),'remaining:',len(remaining),flush=True)
        iterator=iter(remaining)
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending={pool.submit(task,t) for t in [next(iterator,None) for _ in range(4)] if t is not None}
            while pending:
                done,pending=wait(pending,return_when=FIRST_COMPLETED)
                for future in done:
                    t,arrays,record=future.result()
                    if arrays is None:
                        empty.append(dict(**t,reason='less_than_1pct_valid_3dep_land'))
                    else:
                        gr=f.create_group(t['key'])
                        for name,a in arrays.items():
                            gr.create_dataset(name,data=a,compression='gzip',compression_opts=1,shuffle=True)
                        gr.attrs.update(split=t['split'],lon_c=record['lon_c'],lat_c=record['lat_c'],
                                        row=t['row'],col=t['col'],transform=record['transform'],
                                        n_fault_px=record['trace_pixels'],is_positive=record['trace_pixels']>0,
                                        known_pixels=record['known_pixels'],nodata=float(NODATA),record_json=json.dumps(record))
                        gr['k0'].attrs.update(units='angular degrees, Euclidean lon/lat distance',positive_threshold_d0=.01,
                                             method='nearest original land-clipped polyline sample; <=0.0001 degree spacing',invalid_distance=999.)
                        gr['fault_distance_m'].attrs.update(units='metres, EPSG:3310',invalid_distance=-1.,sample_spacing_m=15.)
                        records.append(record)
                    n_done=len(records)+len(empty)
                    f.flush()
                    print(f"Built {n_done}/{len(plan['tiles'])} {t['key']} {t['split']}",flush=True)
                    dump(STAGE/'progress.json',dict(status='building',done=n_done,total=len(plan['tiles']),last_key=t['key']))
                    new=next(iterator,None)
                    if new is not None: pending.add(pool.submit(task,new))
    records.sort(key=lambda t:(t['row'],t['col']))
    counts={s:sum(t['split']==s for t in records) for s in SPLITS}
    provenance=dict(dataset_version='Earth_dataset_v7',created_utc=datetime.now(timezone.utc).isoformat(),
        source_dataset_for_extent_and_label_policy='earth_dataset_V6',elevation_source=str(MOSAIC),
        resolution_m=RES,tile=N,stride=N,crs='EPSG:3310',row_order='north_to_south',counts=counts,n_tiles=len(records),
        mask=dict(buffer_m=12000,source='existing USGS land-clipped faults',quad_segs=16,
                  domain='V6 active positive-elevation land proxy AND finite positive 3DEP; one-pixel valid erosion for terrain derivatives',
                  unknown_is_not_negative=True,verified_survey_completeness=False),
        label=dict(k0='angular nearest-polyline distance, cap >=0.35 to 999',positive_threshold_d0=.01,
                   angular_sample_spacing_deg=.0001,angular_sample_max_distance_error_deg=.00005,
                   change_from_v6='rasterized coarse-grid EDT replaced by sampled original-vector distance at 30 m pixel centers',
                   fault_distance_m='additional metric distance, not consumed as k0',k1_k2='unused: 999'),
        inputs=dict(CTX='2–98 percentile stretched elevation, float32 0..255',image='unscaled elevation metres',
                    TW='3 hillshades: az315/75/195 alt45, vert_exag0.001, dx=dy=30m, 2–98 percentile stretch',
                    valid_input='land only; ocean placeholder and NoData invalid',nodata=float(NODATA),halo_pixels=8),
        splitting={k:v for k,v in plan.items() if k not in ('tiles','excluded','counts')},
        seam_correction='none; original audited mosaic unchanged',training_started=False)
    manifest=dict(provenance=provenance,tiles=records)
    dump(STAGE/'manifest.json',manifest)
    dump(STAGE/'provenance/excluded_tiles.json',plan['excluded']+empty)
    for s in SPLITS:
        (STAGE/f'key_{s}.lst').write_text(''.join(t['key']+'\n' for t in records if t['split']==s))
    for filename in ('key.lst','key_active.lst'):
        (STAGE/filename).write_text(''.join(t['key']+'\n' for t in records))
    geojson(STAGE/'tile_footprints_epsg3310.geojson',records)
    actual_plan=dict(plan,tiles=records,counts=counts)
    split_plot(actual_plan,land,STAGE/'qa/split_map.png')
    readme='''Earth_dataset_v7：30米 USGS 3DEP 陆地数据集；1024×1024；每图30.72×30.72公里。
1. 输入来自已核验的 EPSG:3310 30m 大图，无再次重采样，无接缝平滑；原大图及 V6 保留。
2. 原 V6 活跃图块并集为研究区；仅保留完整落入研究区、有效陆地>=1%的图块。
3. 使用原 V6 正高程陆地几何裁剪现有 USGS 断层，再外扩12000m。沿用原陆地近似，不是新增精细海岸线。
4. valid_input 还要求3DEP有效且高程>0，并内缩1像素避免无效值污染坡度。零值海面、负高程和无数据不参与训练。
5. label_known=1是允许监督的区域；0不是无断层。有效的未知陆地保留供未来伪标签和一致性训练。
6. 全部图块 stride=1024，无图块面积重叠。测试区沿用原 V6 测试地理范围；训练/验证重新做连续地理划分；边界各留1km防护，并剔除跨边界图块。实际留白通常更宽，见qa/validation_report.json。
7. k0保持旧接口的角度距离，d0=0.01；在30m像素中心查询加密后的原始线，非旧384像素标签拉伸。不是统一米数的标签宽度。fault_distance_m另存米制距离，不可直接替代k0输入旧损失。
8. fault_trace为30m栅格断层线；positive_d0_0p01是实际训练阈值目标；12km是监督范围，不是断层宽度。k1/k2=999，只使用k0，禁止把地形边缘变成断层标签。
9. faults.h5保存输入、掩码、标签和地理属性；rasters包含逐图高程及四波段mask GeoTIFF；qa/index.html是全部图块预览。
10. 新栅格北在上。HDF5、GeoTIFF、PNG方向完全一致；请按transform定位，不要套用旧GEBCO行号经纬度公式。
11. 当前仅制作和验证数据，未训练。旧Generator_new可在training=False,fixed_crop=True,m=n=1024时读取。默认训练裁剪/缩放不适合等尺寸1024，需后续单独适配；不能仅改mn后直接启动旧训练。
12. v7_dataset.py提供完整1024读取及可选翻转/转置，不缩放、不随机裁小；旧训练脚本尚未接入。无需改动原有海洋训练代码。
13. 原V6训练/验证地理范围曾重叠，因此新划分的指标不可直接当作与旧V6完全同条件对照。源断层位置精度不会因30m网格而自动提高。
'''
    (STAGE/'README.txt').write_text(readme,encoding='utf-8')
    dump(STAGE/'dataset_usage.json',dict(status='DATASET_ONLY_NO_TRAINING',dataset='Earth_dataset_v7/key_train.lst',
        dataFile='Earth_dataset_v7/faults.h5',supervisionKey='label_known',k0Only=True,isOther=False,d0=.01,
        stored_shape=[N,N],recommended_full_tile_reader='v7_dataset.FullTileDataset',
        training_script_integrated=False,marine_input_policy='invalid in this land-only 3DEP dataset; original marine workflows unchanged'))
    gallery(manifest)
    print('Checking every tile, raster, mask and split...',flush=True)
    report=validate(plan,manifest,core_before,before)
    dump(STAGE/'qa/validation_report.json',report)
    # CPU loader checks do not instantiate a model, optimizer, or training loop.
    from v7_dataset import validate_readers
    runtime=validate_readers(STAGE,records)
    dump(STAGE/'qa/reader_validation.json',runtime)
    report['reader_validation']='PASS'
    dump(STAGE/'qa/validation_report.json',report)
    for name in ('build_earth_dataset_v7.py','v7_dataset.py'):
        shutil.copy2(CODE/name,STAGE/'provenance'/name)
    dump(STAGE/'progress.json',dict(status='complete',done=len(records),total=len(records)))
    assert STAGE.resolve().parent==CODE and DST.resolve().parent==CODE and not DST.exists()
    os.rename(STAGE,DST)
    print('PUBLISHED',DST,counts,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    print('Loading V6 land geometry and existing USGS fault source...',flush=True)
    ctx=geometry_context()
    active,oldtest,land,source_lines,lines,envelope=ctx
    plan=make_plan(active,oldtest,land)
    PLAN.mkdir(parents=True,exist_ok=True)
    dump(PLAN/'plan.json',plan)
    geojson(PLAN/'tile_footprints_epsg3310.geojson',plan['tiles'])
    split_plot(plan,land,PLAN/'split_map.png')
    print('PLAN',json.dumps(plan['counts']), 'cut',plan['train_val_northing_boundary_m'],flush=True)
    if args.plan_only:
        return
    build(plan,ctx,args.resume)


if __name__=='__main__':
    main()
