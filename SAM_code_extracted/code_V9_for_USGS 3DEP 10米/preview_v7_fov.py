"""Review-only nested 5.12/10.24/20.48 km DEM windows. No dataset or training creation.

Run with diagnostics/.python-3dep-preview on PYTHONPATH in the existing WSL GIS environment.
"""
from pathlib import Path
import argparse
import hashlib
import html
import json
import math
from datetime import datetime, timezone

import geopandas as gpd
import h5py
import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject, transform_bounds
from pyproj import Transformer
from shapely import box, buffer, length, union_all
from shapely.geometry import Point, mapping, shape
from shapely.ops import nearest_points
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from PIL import Image

from preview_v6_land_envelope import land_geometry

CODE=Path(__file__).resolve().parent
ROOT=CODE.parents[1]
DATA=CODE/'earth_dataset_V6'
DEM=ROOT/'USGS 3DEP 10米'
OUT=CODE/'diagnostics/v7_fov_preview_v1'
PIXEL=10.; SIZES=(512,1024,2048); HALO=64
TO_METRIC=Transformer.from_crs(4326,3310,always_xy=True)
TO_LONLAT=Transformer.from_crs(3310,4326,always_xy=True)
SITE_TEXT={
 'point_reyes':('Point Reyes coastal fault belt','Point Reyes 海岸断层带','海岸与陆海边界'),
 'santa_cruz':('Santa Cruz mountain region','Santa Cruz 山地区域','沟谷密集的山地'),
 'carrizo':('Carrizo Plain','Carrizo 平原','干旱平原与线性地貌'),
 'san_bernardino':('San Bernardino mountain front','San Bernardino 山前带','山前、盆地与分支断层')}

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def dump(path,data): path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def write_tif(path,a,tr,dtype='float32',nodata=-999999.):
    with rio.open(path,'w',driver='GTiff',height=a.shape[0],width=a.shape[1],count=1,
                  dtype=dtype,crs='EPSG:3310',transform=tr,nodata=nodata,compress='deflate',tiled=True) as f:
        f.write(np.where(np.isfinite(a),a,nodata).astype(dtype),1)
        f.update_tags(purpose='FOV preview only; not earth_dataset_v7',pixel_size_m='10')

def load_v6():
    m=json.loads((DATA/'manifest.json').read_text()); res=m['provenance']['res_deg']; size=m['provenance']['tile']
    active_keys=set().union(*[(DATA/f'key_{s}.lst').read_text().split() for s in ('train','val','test')])
    first=m['tiles'][0]; i,j=map(int,first['key'].split('_')[1:])
    x0=first['lon_c']-(j+(size-1)/2)*res; y0=first['lat_c']-(i+(size-1)/2)*res
    H=max(int(k.split('_')[1])+size for k in active_keys); W=max(int(k.split('_')[2])+size for k in active_keys)
    z=np.full((H,W),np.nan,np.float32); active=np.zeros((H,W),bool); known=active.copy()
    with h5py.File(DATA/'faults.h5','r') as f:
        for k in sorted(active_keys):
            i,j=map(int,k.split('_')[1:]); sl=np.s_[i:i+size,j:j+size]
            a=f[k]['image'][:]; seen=active[sl]
            assert np.array_equal(z[sl][seen],a[seen],equal_nan=True)
            z[sl]=a; active[sl]=True; known[sl]|=f[k]['label_known'][:].astype(bool)
    land=active&np.isfinite(z)&(z>0)
    assert land.sum()==2148932 and known.sum()==1612935
    geom=land_geometry(land,x0,y0,res)
    tr=from_origin(x0-res/2,y0+(H-.5)*res,res,res)
    return geom,tr,np.flipud(land),np.flipud(active),np.flipud(known)

def draw_pair(top,bottom,site,arrays,n,show_boxes=True):
    start=(2048-n)//2; sl=np.s_[start:start+n,start:start+n]
    z=arrays['elevation'][sl]; hs=arrays['hillshade'][sl]; known=arrays['known'][sl]
    active=arrays['active'][sl]; land=arrays['land'][sl]
    half=n*PIXEL/2000; extent=(-half,half,-half,half)
    display=z.copy(); display[(z<=0)&~land]=np.nan
    gray=plt.get_cmap('gray').copy(); gray.set_bad('#dce7eb')
    cmap=plt.get_cmap('terrain').copy(); cmap.set_bad('#dce7eb')
    top.imshow(display,extent=extent,origin='upper',cmap=cmap,vmin=site['vmin'],vmax=site['vmax'],interpolation='nearest')
    hdisplay=hs.copy(); hdisplay[~np.isfinite(display)]=np.nan
    bottom.imshow(hdisplay,extent=extent,origin='upper',cmap=gray,vmin=0,vmax=1,interpolation='nearest')
    # Green fill makes even fully covered windows explicit; no invented buffer edge.
    overlay=np.zeros((*known.shape,4),dtype=np.float32); overlay[known]=[.10,.75,.30,.16]
    bottom.imshow(overlay,extent=extent,origin='upper',interpolation='nearest')
    coords=(np.arange(n)+.5)*.01-half; ys=coords[::-1]
    for ax in (top,bottom):
        if known.any() and not known.all(): ax.contour(coords,ys,known,levels=[.5],colors=['#00a54e'],linewidths=1)
        if active.any() and not active.all():
            ax.contour(coords,ys,active,levels=[.5],colors=['#bf00a5'],linewidths=1)
            excluded=np.zeros((*active.shape,4),np.float32); excluded[~active]=[.5,.5,.5,.35]
            ax.imshow(excluded,extent=extent,origin='upper')
        for geom in site['plot_lines']:
            def parts(g):
                if g.geom_type=='LineString': yield g
                elif hasattr(g,'geoms'):
                    for p in g.geoms: yield from parts(p)
            for line in parts(geom):
                xy=np.asarray(line.coords); ax.plot((xy[:,0]-site['cx'])/1000,(xy[:,1]-site['cy'])/1000,color='#e51839',lw=.7)
        ax.plot(0,0,marker='+',color='#0526b5',ms=9,mew=1.3)
        ax.set(xlim=extent[:2],ylim=extent[2:],xlabel='East of center (km)',ylabel='North of center (km)',aspect='equal')
        ax.tick_params(labelsize=8)
    if show_boxes:
        for smaller,color in [(512,'#b85c00'),(1024,'#773bb4')]:
            if smaller<n:
                h=smaller*PIXEL/2000
                top.add_patch(Rectangle((-h,-h),2*h,2*h,fill=False,ec=color,lw=1.1,ls='--'))
    frac=100*known.sum()/max(1,land.sum())
    top.set_title(f'{n*.01:.2f} x {n*.01:.2f} km | {n} x {n} pixels',fontsize=12)
    bottom.set_title(f'Hillshade + 12 km mask | {frac:.1f}% of V6 land',fontsize=10)

def render(site,arrays):
    site=dict(site); site['plot_lines']=[shape(g) for g in site['lines_metric']]
    fig,axs=plt.subplots(2,3,figsize=(18,12))
    fig.subplots_adjust(left=.055,right=.91,bottom=.12,top=.89,wspace=.23,hspace=.25)
    for i,n in enumerate(SIZES): draw_pair(axs[0,i],axs[1,i],site,arrays,n)
    norm=matplotlib.colors.Normalize(site['vmin'],site['vmax'])
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap='terrain'),cax=fig.add_axes([.935,.56,.012,.28]),label='Elevation (m NAVD88); shared across all three views')
    handles=[Line2D([],[],color='#e51839',label='Existing USGS land fault'),Patch(color='#23bd4c',alpha=.3,label='12 km supervision mask'),
             Line2D([],[],marker='+',color='#0526b5',ls='',label='Same center'),Line2D([],[],color='#b85c00',ls='--',label='5.12 km footprint'),
             Line2D([],[],color='#773bb4',ls='--',label='10.24 km footprint')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.015),ncol=3,fontsize=10)
    fig.suptitle(f"{site['english']} | 10 m grid, EPSG:3310 (grid north up)\nCenter: {site['lon']:.5f}, {site['lat']:.5f} | {site['center_fault_name']}",fontsize=14)
    fig.savefig(OUT/f"{site['id']}_comparison.png",dpi=150); plt.close(fig)
    for n in SIZES:
        fig,axs=plt.subplots(1,2,figsize=(13,6),layout='constrained')
        draw_pair(axs[0],axs[1],site,arrays,n,False)
        fig.suptitle(f"{site['english']} | {n*.01:.2f} km FOV",fontsize=14)
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap='terrain'),ax=axs[0],label='Elevation (m)',shrink=.7)
        fig.savefig(OUT/f"{site['id']}_{n}.png",dpi=180); plt.close(fig)

def html_page(report):
    cards=[]
    for s in report['sites']:
        rows=''.join(f'<tr><td>{v["km"]:.2f} km</td><td>{v["pixels"]} × {v["pixels"]}</td><td>{v["valid_percent"]:.2f}%</td><td>{v["land_percent"]:.1f}%</td><td>{v["known_of_land_percent"]:.1f}%</td><td><a href="{s["id"]}_{v["pixels"]}.png">放大图</a> · <a href="rasters/{s["id"]}_{v["pixels"]}_elevation.tif">高程TIF</a> · <a href="rasters/{s["id"]}_{v["pixels"]}_known.tif">mask TIF</a></td></tr>' for v in s['views'])
        cards.append(f'''<section id="{s['id']}"><h2>{s['chinese']}</h2><p>{s['description']}。中心：{s['lon']:.5f}°，{s['lat']:.5f}°；中心附近断层：{html.escape(s['center_fault_name'])}。</p>
        <p>本窗口读取 {len(s['source_tiles'])} 个原始瓦片：{', '.join(s['source_tiles'])}。绿色为按 V6 规则重建的 12 公里监督 mask；不是断层正类宽度。</p>
        <a href="{s['id']}_comparison.png"><img loading="lazy" src="{s['id']}_comparison.png" alt="{s['chinese']} 三种视野对比"></a>
        <table><tr><th>边长</th><th>10米像元数</th><th>高程有效覆盖</th><th>V6陆地占比</th><th>陆地内mask覆盖</th><th>查看／下载</th></tr>{rows}</table></section>''')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V7 裁剪尺度预览</title><style>
    body{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#f3f5f6;color:#1b303d;max-width:1550px;margin:auto;padding:24px;line-height:1.7}section{background:white;padding:24px;margin:22px 0;border-radius:12px}img{width:100%;height:auto}a{color:#12698d}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}.note{background:#fff5db;border-left:4px solid #c39833;padding:15px}nav a{margin-right:18px}code{background:#e8edf1;padding:3px}</style>
    <h1>earth_dataset_v7 · 同中心裁剪尺度预览</h1><p>仅用于选择视野；尚未制作 earth_dataset_v7，未启动训练，也未修改 V6 或原始 3DEP 瓦片。</p>
    <nav>'''+''.join(f'<a href="#{s["id"]}">{s["chinese"]}</a>' for s in report['sites'])+'''</nav>
    <section><h2>如何比较</h2><p>每处三个视野来自同一中心、同一张 10 米 EPSG:3310 网格，分别是512、1024、2048像素，边长5.12、10.24、20.48公里。小图是大图的精确中心子数组，没有分别拉伸或重算高程；每处三种视野共用高程色标。图上方为高程，下方为同光照条件的地形阴影。</p>
    <p>红线：现有 USGS 陆地断层；绿色填色及边界：12公里监督mask；蓝色十字：共同中心；橙／紫虚线：较小视野的地理范围。坐标轴为相对中心的公里数，采用投影网格北向上。点击各尺度“放大图”可检查细节。</p>
    <p class="note">12公里是从断层向两侧外扩的监督范围。最大视野半边长只有10.24公里，中心又靠近断层，因此很多窗口的全部陆地都在mask内。这是正常的几何结果；不绘制不存在的mask边界，也不把mask当作断层正类带。</p>
    <p>mask按原V6规则：现有USGS断层先裁到V6有效陆地，再外扩12000米、与陆地和有效高程相交，在10米网格重栅格化。陆地定义仍沿用V6粗网格的高程&gt;0代理，故海岸mask可能呈阶梯状；它不是新的10米精确海岸线。原V6的label_known还另外做了最近邻对照，差异写入报告。</p>
    <p>读取周边64像素作为重采样和阴影计算邻域，最终视野不包含这些外围像元。原始USGS瓦片按固定优先级填入同一网格，没有做高程羽化或接缝校正。缺失或水域背景不表示GEBCO海底；本次预览只比较陆地3DEP。</p>
    <p class="note">网页把不同像素数的图缩放到同样显示宽度，仅用于比较地理上下文。不能把20.48公里图直接缩成256像素后仍称作10米输入：那相当于80米采样。此次没有决定模型输入尺寸、断层正类宽度或训练配置。</p></section>'''+''.join(cards)+'''
    <section><h2>记录</h2><p><a href="report.json">范围、源瓦片、mask比例及验证报告</a> · <a href="preview_v7_fov.py">生成脚本快照</a></p><p>选择尺度时重点看：断层走向是否有足够上下文、谷地和山脊是否能区分、多个分支是否仍可辨认。这里展示的是数据与标签，尚无模型预测。</p></section></html>'''
    (OUT/'index.html').write_text(page,encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--render-only',action='store_true'); args=ap.parse_args()
    if args.render_only:
        report=json.loads((OUT/'report.json').read_text())
        for s in report['sites']:
            with np.load(OUT/f"{s['id']}_arrays.npz") as a: render(s,a)
        html_page(report)
        (OUT/'preview_v7_fov.py').write_text(Path(__file__).read_text(),encoding='utf-8')
        return
    if OUT.exists(): raise FileExistsError(f'Refusing existing output {OUT}; use --render-only for plot changes')
    OUT.mkdir(); (OUT/'rasters').mkdir()
    source_paths=[DATA/'faults.h5',DATA/'manifest.json']+[DATA/f'key_{s}.lst' for s in ('train','val','test')]
    stat_before={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in source_paths}
    print('Read-only V6 land footprint and supervision reconstruction',flush=True)
    land_geom,oldtr,oldland,oldactive,oldknown=load_v6()
    catalog=gpd.read_file(ROOT/'Qfaults_GIS/SHP/Qfaults_US_Database.shp',bbox=land_geom.bounds).to_crs(4326)
    previous=json.loads((CODE/'diagnostics/v6_3dep10m_comparison_v1/report.json').read_text())
    inventory=json.loads((DEM/'remote_file_metadata.json').read_text())
    inventory=sorted(inventory,key=lambda x:(x['publication_date'],x['tile']),reverse=True)
    used_files={}; sites=[]
    for target in previous['samples']:
        sid=target['id']; print('Prepare',sid,flush=True)
        b=target['bbox']; tx,ty=TO_METRIC.transform((b[0]+b[2])/2,(b[1]+b[3])/2)
        # The 35-km query radius includes the maximum window corner radius plus its 12-km buffer needs.
        query=transform_bounds('EPSG:3310','EPSG:4326',tx-35000,ty-35000,tx+35000,ty+35000,densify_pts=21)
        local=catalog.iloc[catalog.sindex.query(box(*query),predicate='intersects')].copy()
        local.geometry=local.geometry.intersection(land_geom)
        local=local.loc[~local.geometry.is_empty & (length(local.geometry.to_numpy())>0)].to_crs(3310)
        assert len(local)
        near=local.geometry.distance(Point(tx,ty)); nearest=local.iloc[int(np.argmin(near))]
        snapped=nearest_points(Point(tx,ty),nearest.geometry)[1]
        cx=round(snapped.x/10)*10; cy=round(snapped.y/10)*10; lon,lat=TO_LONLAT.transform(cx,cy)
        N=2048+2*HALO; tr=from_origin(cx-N*5,cy+N*5,10,10)
        extent=rio.transform.array_bounds(N,N,tr)
        geographic=transform_bounds('EPSG:3310','EPSG:4326',*extent,densify_pts=21)
        elevation=np.full((N,N),np.nan,np.float32); source_idx=np.zeros((N,N),np.uint8); used=[]
        for item in inventory:
            bb=item['bounding_box']; footprint=box(bb['minX'],bb['minY'],bb['maxX'],bb['maxY'])
            if not footprint.intersects(box(*geographic)): continue
            path=DEM/item['url'].rsplit('/',1)[1]
            assert path.stat().st_size==item['bytes']
            used_files[str(path)]=(path.stat().st_size,path.stat().st_mtime_ns)
            with rio.open(path) as src, WarpedVRT(src,crs='EPSG:3310',transform=tr,width=N,height=N,
                    resampling=Resampling.bilinear,nodata=-999999.,dtype='float32') as vrt:
                data=vrt.read(1,masked=True).filled(np.nan)
                take=np.isnan(elevation)&np.isfinite(data)
                if take.any():
                    elevation[take]=data[take]; used.append(item['tile']); source_idx[take]=len(used)
        assert np.isfinite(elevation).any()
        land=np.zeros((N,N),np.uint8); active=land.copy(); oldmask=land.copy()
        for src,dst in [(oldland,land),(oldactive,active),(oldknown,oldmask)]:
            reproject(src.astype('uint8'),dst,src_transform=oldtr,src_crs=4326,dst_transform=tr,dst_crs=3310,
                      resampling=Resampling.nearest,dst_nodata=0)
        land=land.astype(bool); active=active.astype(bool); oldmask=oldmask.astype(bool)
        # Buffer complete local source lines before intersecting the output: outside-window faults count.
        pieces=[]
        for start in range(0,len(local),256): pieces.append(union_all(buffer(local.geometry.to_numpy()[start:start+256],12000,quad_segs=16)))
        envelope=union_all(pieces)
        known=rasterize([(envelope,1)],out_shape=(N,N),transform=tr,dtype='uint8').astype(bool)
        known &= land & active & np.isfinite(elevation)
        dy,dx=np.gradient(elevation,10,10)
        hs=np.clip((.5*dx+.5*dy+np.sqrt(.5))/np.sqrt(1+dx*dx+dy*dy),0,1)
        center=np.s_[HALO:HALO+2048,HALO:HALO+2048]
        arrays={k:a[center] for k,a in dict(elevation=elevation,hillshade=hs,known=known,land=land,active=active,legacy_known=oldmask,source_id=source_idx).items()}
        np.savez_compressed(OUT/f'{sid}_arrays.npz',**arrays)
        maxbox=box(cx-10240,cy-10240,cx+10240,cy+10240)
        lines=local.loc[local.intersects(maxbox)].copy(); lines.geometry=lines.geometry.intersection(maxbox)
        vals=arrays['elevation'][np.isfinite(arrays['elevation']) & (arrays['elevation']>0)]
        english,chinese,desc=SITE_TEXT[sid]
        site=dict(id=sid,english=english,chinese=chinese,description=desc,cx=cx,cy=cy,lon=lon,lat=lat,
            center_fault_name=str(nearest['fault_name']),center_fault_section=str(nearest['section_na']),
            center_fault_scale=str(nearest['scale']),center_fault_certainty=str(nearest['certainty']),
            source_tiles=used,lines_metric=[mapping(g) for g in lines.geometry],vmin=float(vals.min()),vmax=float(vals.max()),views=[])
        for n in SIZES:
            st=(2048-n)//2; sl=np.s_[st:st+n,st:st+n]; tt=from_origin(cx-n*5,cy+n*5,10,10)
            elev=arrays['elevation'][sl]; known=arrays['known'][sl]; land=arrays['land'][sl]; active=arrays['active'][sl]
            assert not (known&~land).any() and not (known&~active).any() and not (known&~np.isfinite(elev)).any()
            assert active.all(),f'{sid} {n}: selected view extends beyond V6 active footprint'
            write_tif(OUT/'rasters'/f'{sid}_{n}_elevation.tif',elev,tt)
            write_tif(OUT/'rasters'/f'{sid}_{n}_known.tif',known,tt,'uint8',255)
            write_tif(OUT/'rasters'/f'{sid}_{n}_land.tif',land,tt,'uint8',255)
            site['views'].append(dict(pixels=n,km=n*.01,bounds_metric=list(rio.transform.array_bounds(n,n,tt)),
                valid_percent=100*float(np.isfinite(elev).mean()),land_percent=100*float(land.mean()),
                known_of_land_percent=100*float(known.sum()/max(1,land.sum())),
                old_v6_mask_disagreement_percent=100*float((known!=arrays['legacy_known'][sl]).mean())))
        print(sid,'sources',used,'mask',[round(x['known_of_land_percent'],2) for x in site['views']],flush=True)
        render(site,arrays); sites.append(site)
    report=dict(status='complete',created_utc=datetime.now(timezone.utc).isoformat(),purpose='FOV decision preview only',
        planned_dataset='earth_dataset_v7',dataset_created=False,training_started=False,pixel_size_m=10,
        crs='EPSG:3310',vertical_datum='NAVD88 (source unchanged)',halo_pixels=HALO,buffer_m=12000,
        mask_protocol='Same USGS vector source and V6 positive-elevation active land proxy; 12km metric buffers re-rasterized at 10m, not a new coastline',
        source_priority='Publication date descending, then tile id descending; first valid wins; no feathering',
        sites=sites)
    for p in source_paths:
        assert stat_before[str(p)]==(p.stat().st_size,p.stat().st_mtime_ns)
    for p,stat in used_files.items():
        assert stat==(Path(p).stat().st_size,Path(p).stat().st_mtime_ns)
    expected=json.loads((DATA/'qa/validation_report.json').read_text())['dataset_hdf5_sha256']
    report['v6_sha256']=digest(DATA/'faults.h5'); assert report['v6_sha256']==expected
    report['source_files_unchanged']=True
    # Check actual exported rasters, nested-array identity, and image decodability.
    for s in sites:
        with rio.open(OUT/'rasters'/f"{s['id']}_2048_elevation.tif") as full:
            large=full.read(1)
            for n in SIZES:
                with rio.open(OUT/'rasters'/f"{s['id']}_{n}_elevation.tif") as small:
                    st=(2048-n)//2; assert np.array_equal(small.read(1),large[st:st+n,st:st+n])
                    b=small.bounds; assert np.isclose((b.left+b.right)/2,s['cx']) and np.isclose((b.top+b.bottom)/2,s['cy'])
                    assert small.res==(10.,10.) and small.crs.to_epsg()==3310
    for p in OUT.glob('*.png'):
        with Image.open(p) as im: im.verify()
    report['qa']=dict(nested_elevation_arrays_identical=True,same_center=True,metric_extents_exact=True,
                      native_view_count=12,comparison_figures=4,training_imports=False)
    dump(OUT/'report.json',report); html_page(report)
    (OUT/'preview_v7_fov.py').write_text(Path(__file__).read_text(),encoding='utf-8')
    print('COMPLETE',OUT,flush=True)

if __name__=='__main__': main()
