"""Read-only V6 coverage audit and four USGS 10m / GEBCO comparison crops.

Run from this directory with diagnostics/.python-3dep-preview on PYTHONPATH.
Only diagnostics/v6_3dep10m_comparison_v1 is written; no training imports.
"""
from pathlib import Path
import concurrent.futures as cf
import hashlib
import html
import json
import math
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import h5py
import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.features import rasterize
from rasterio.windows import from_bounds as window_bounds, Window
from shapely.geometry import box, shape, mapping
from shapely import union_all
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LightSource
from matplotlib.patches import Patch, Rectangle

CODE=Path(__file__).resolve().parent
OUT=CODE/'diagnostics/v6_3dep10m_comparison_v1'
DATA=CODE/'earth_dataset_V6'
ENV=dict(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',
         GDAL_HTTP_TIMEOUT='45', GDAL_HTTP_MAX_RETRY='2', GDAL_HTTP_RETRY_DELAY='1')

def save_json(p,data):
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def fetch_json(url,path):
    if not path.exists(): path.write_bytes(urllib.request.urlopen(url,timeout=60).read())
    return json.loads(path.read_text())

def write_tif(path,array,transform,crs):
    with rio.open(path,'w',driver='GTiff',width=array.shape[1],height=array.shape[0],count=1,
                  dtype='float32',crs=crs,transform=transform,nodata=-999999.,compress='deflate') as d:
        d.write(np.where(np.isfinite(array),array,-999999.).astype('float32'),1)
        d.update_tags(PURPOSE='Read-only preview; not a training dataset',UNITS='meters')

def main():
    OUT.mkdir(exist_ok=True); (OUT/'rasters').mkdir(exist_ok=True); (OUT/'coverage_cache').mkdir(exist_ok=True)
    source_stat=(DATA/'faults.h5').stat()
    manifest=json.loads((DATA/'manifest.json').read_text())
    p=manifest['provenance']; res=p['res_deg']; size=p['tile']
    splits={s:(DATA/f'key_{s}.lst').read_text().split() for s in ('train','val','test')}
    active_keys=set(sum(splits.values(),[]))
    tiles=manifest['tiles']; keys=[t['key'] for t in tiles]
    i0,j0=map(int,tiles[0]['key'].split('_')[1:])
    x0=tiles[0]['lon_c']-(j0+(size-1)/2)*res
    y0=tiles[0]['lat_c']-(i0+(size-1)/2)*res
    H=max(int(k.split('_')[1])+size for k in keys); W=max(int(k.split('_')[2])+size for k in keys)
    z=np.full((H,W),np.nan,np.float32); active=np.zeros((H,W),bool); stored=active.copy(); known=active.copy(); trace=active.copy()
    print('Read V6, all stored and active footprints',flush=True)
    with h5py.File(DATA/'faults.h5','r') as f:
        for k in keys:
            i,j=map(int,k.split('_')[1:]); region=np.s_[i:i+size,j:j+size]
            a=f[k]['image'][:]; overlap=stored[region]
            assert np.array_equal(z[region][overlap],a[overlap],equal_nan=True)
            z[region]=a; stored[region]=True
            if k in active_keys:
                active[region]=True; known[region]|=f[k]['label_known'][:].astype(bool)
                trace[region]|=f[k]['fault_trace'][:].astype(bool)
    z[z<-1e8]=np.nan
    z,active,stored,known,trace=[np.flipud(a) for a in (z,active,stored,known,trace)]
    bounds=(x0-res/2,y0-res/2,x0+(W-.5)*res,y0+(H-.5)*res)
    tr=from_bounds(*bounds,W,H)
    xs=x0+np.arange(W)*res; ys=y0+np.arange(H-1,-1,-1)*res
    xx,yy=np.meshgrid(xs,ys)
    land=np.isfinite(z)&(z>0)&stored
    area=(6371.0088**2*np.deg2rad(res)*(np.sin(np.deg2rad(ys+res/2))-np.sin(np.deg2rad(ys-res/2))))[:,None]*np.ones((1,W))
    def km2(m): return float(area[m].sum())
    query='https://tnmaccess.nationalmap.gov/api/v1/products?'+urllib.parse.urlencode(dict(
        datasets='National Elevation Dataset (NED) 1/3 arc-second Current',bbox=','.join(map(str,p['bbox'])),max=1000))
    raw=fetch_json(query,OUT/'tnm_catalog_current.json'); assert raw['total']==len(raw['items'])
    items=raw['items']; assert all(x['format']=='GeoTIFF' for x in items)
    # Current status is determined by the API tag, never the URL folder (which may say historical).
    current={}
    for x in items:
        tile=re.search(r'n\d+w\d+',x['downloadURL']).group()
        if tile not in current or x['publicationDate']>current[tile]['publicationDate']: current[tile]=x
    caurl='https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/State_County/MapServer/0/query?'+urllib.parse.urlencode(dict(
        where="STUSAB='CA'",outFields='NAME,STUSAB',returnGeometry='true',outSR=4326,f='geojson'))
    ca=fetch_json(caurl,OUT/'california_boundary.geojson')
    assert ca.get('features'),ca
    ca_geom=union_all([shape(f['geometry']) for f in ca['features']])
    camask=rasterize([(ca_geom,1)],out_shape=(H,W),transform=tr,dtype='uint8').astype(bool)
    selected=[]; catalog=np.zeros((H,W),bool)
    for tile,x in sorted(current.items()):
        b=x['boundingBox']; geom=box(b['minX'],b['minY'],b['maxX'],b['maxY'])
        m=rasterize([(geom,1)],out_shape=(H,W),transform=tr,dtype='uint8').astype(bool)
        if not (m&stored).any(): continue
        catalog|=m
        selected.append(dict(tile=tile,title=x['title'],url=x['downloadURL'],metadata_url=x['vendorMetaUrl'],
            publication_date=x['publicationDate'],bytes=x['sizeInBytes'],bounding_box=b,
            active_land_km2=km2(m&active&land),stored_land_km2=km2(m&land)))
    save_json(OUT/'download_inventory.json',selected)
    (OUT/'download_urls_active_land.txt').write_text('\n'.join(x['url'] for x in selected if x['active_land_km2']>0)+'\n')
    (OUT/'current_query_url.txt').write_text(query)
    print(f'Current query {len(items)}; intersects stored {len(selected)}; active land {sum(x["active_land_km2"]>0 for x in selected)}',flush=True)

    # Read native COG overview for every selected tile, not billions of full-resolution pixels.
    def coarse(x):
        dest=OUT/'coverage_cache'/f'{x["tile"]}.tif'
        if not dest.exists():
            with rio.Env(**ENV),rio.open(x['url']) as src:
                scale=max(src.overviews(1) or [1]); h=math.ceil(src.height/scale); w=math.ceil(src.width/scale)
                a=src.read(1,out_shape=(h,w),masked=True).filled(np.nan)
                t=src.transform*src.transform.scale(src.width/w,src.height/h)
                write_tif(dest,a,t,src.crs)
                save_json(dest.with_suffix('.json'),dict(source=x['url'],native_crs=str(src.crs),native_resolution=list(src.res),
                    native_shape=[src.height,src.width],overview_factor=scale,nodata=src.nodata))
        with rio.open(dest) as src:
            b=transform_bounds(src.crs,'EPSG:4326',*src.bounds)
            c0=max(0,int(math.floor((b[0]-bounds[0])/res))); c1=min(W,int(math.ceil((b[2]-bounds[0])/res)))
            r0=max(0,int(math.floor((bounds[3]-b[3])/res))); r1=min(H,int(math.ceil((bounds[3]-b[1])/res)))
            a=np.full((r1-r0,c1-c0),np.nan,np.float32)
            reproject(rio.band(src,1),a,src_transform=src.transform,src_crs=src.crs,src_nodata=src.nodata,
                dst_transform=tr*rio.Affine.translation(c0,r0),dst_crs='EPSG:4326',dst_nodata=np.nan,resampling=Resampling.nearest)
        return (r0,r1,c0,c1,a)
    dep=np.full_like(z,np.nan)
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        for n,(r0,r1,c0,c1,a) in enumerate(pool.map(coarse,selected),1):
            view=dep[r0:r1,c0:c1]; good=np.isfinite(a); view[good]=a[good]
            print(f'Coverage COG read {n}/{len(selected)}',flush=True)
    valid=np.isfinite(dep)
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='building',
        query=query,california_boundary_url=caurl,nominal_bbox=p['bbox'],stored_pixel_bounds=bounds,
        stored_tiles=len(keys),active_tiles=len(active_keys),split_counts={s:len(k) for s,k in splits.items()},
        current_query_products=len(items),intersecting_stored_products=len(selected),
        active_land_products=sum(x['active_land_km2']>0 for x in selected),
        active_land_full_download_GB=sum(x['bytes'] for x in selected if x['active_land_km2']>0)/1e9,
        coverage_method='V6 positive-elevation land proxy; spherical cell areas; product bbox + COG 32x overview validity warped to V6 grid. Not a full native-pixel audit.',
        coverage={})
    for name,m in dict(stored_positive_land=land,active_positive_land=land&active,
                       active_california_positive_land=land&active&camask,active_supervision=known&active).items():
        report['coverage'][name]=dict(area_km2=km2(m),catalog_covered_percent=100*km2(m&catalog)/km2(m),
            overview_valid_percent=100*km2(m&valid)/km2(m),overview_missing_km2=km2(m&~valid))
    print(json.dumps(report['coverage']),flush=True)
    save_json(OUT/'report.json',report)

    # Choose small regions around existing active USGS fault pixels, wholly within the active footprint
    # and within one current 1-degree DEM tile (leaving a margin around each crop).
    targets=[('point_reyes','Point Reyes / San Andreas',-122.78,38.06),
             ('santa_cruz','Santa Cruz Mountains',-121.78,37.10),
             ('carrizo','Carrizo Plain / San Andreas',-119.78,35.15),
             ('san_bernardino','San Bernardino / San Andreas',-117.18,34.20)]
    faults=gpd.read_file(CODE.parents[1]/'Qfaults_GIS/SHP/Qfaults_US_Database.shp',bbox=bounds).to_crs(4326)
    samples=[]; N=36; half=N//2
    for name,label,tx,ty in targets:
        rr,cc=np.where(trace&active&valid&camask)
        order=np.argsort(((xs[cc]-tx)*np.cos(np.deg2rad(ty)))**2+(ys[rr]-ty)**2)
        chosen=None
        for idx in order[:15000]:
            r,c=rr[idx],cc[idx]; r0=r-half; c0=c-half; r1=r0+N; c1=c0+N
            if r0<0 or c0<0 or r1>H or c1>W or not active[r0:r1,c0:c1].all(): continue
            t=tr*rio.Affine.translation(c0,r0); b=rio.transform.array_bounds(N,N,t)
            source=next((x for x in selected if x['bounding_box']['minX']<b[0]-.001 and x['bounding_box']['maxX']>b[2]+.001
                         and x['bounding_box']['minY']<b[1]-.001 and x['bounding_box']['maxY']>b[3]+.001),None)
            if source: chosen=(r0,r1,c0,c1,t,b,source); break
        assert chosen,name
        r0,r1,c0,c1,t,b,source=chosen
        print('Preview',name,'bounds',b,'source',source['tile'],flush=True)
        native=OUT/'rasters'/f'{name}_3dep_native.tif'
        if not native.exists():
            with rio.Env(**ENV),rio.open(source['url']) as src:
                sb=transform_bounds('EPSG:4326',src.crs,*b)
                win=window_bounds(*sb,src.transform)
                # Small margin gives area averaging valid support at target borders.
                win=Window(math.floor(win.col_off)-2,math.floor(win.row_off)-2,math.ceil(win.width)+5,math.ceil(win.height)+5)
                a=src.read(1,window=win,masked=True).filled(np.nan)
                write_tif(native,a,src.window_transform(win),src.crs)
        fine=np.full((N*45,N*45),np.nan,np.float32); ft=from_bounds(*b,fine.shape[1],fine.shape[0])
        avg=np.full((N,N),np.nan,np.float32)
        with rio.open(native) as src:
            for dst,dtr,method in [(fine,ft,Resampling.bilinear),(avg,t,Resampling.average)]:
                reproject(rio.band(src,1),dst,src_transform=src.transform,src_crs=src.crs,src_nodata=src.nodata,
                    dst_transform=dtr,dst_crs='EPSG:4326',dst_nodata=np.nan,resampling=method)
        coarse_z=z[r0:r1,c0:c1].copy(); coarse_z[coarse_z<=0]=np.nan
        display_fine=fine.copy(); display_fine[display_fine<=0]=np.nan
        display_avg=avg.copy(); display_avg[display_avg<=0]=np.nan
        # Positive elevation here is only a preview mask matching V6, not a new coastline.
        write_tif(OUT/'rasters'/f'{name}_gebco_v6.tif',z[r0:r1,c0:c1],t,'EPSG:4326')
        write_tif(OUT/'rasters'/f'{name}_3dep_on_v6_grid.tif',avg,t,'EPSG:4326')
        vals=np.concatenate([a[np.isfinite(a)] for a in (coarse_z,display_fine,display_avg)])
        vmin,vmax=float(vals.min()),float(vals.max())
        ext=(b[0],b[2],b[1],b[3]); lat=(b[1]+b[3])/2
        fig,axes=plt.subplots(2,3,figsize=(15,10),layout='constrained')
        cmap=plt.get_cmap('terrain').copy(); cmap.set_bad('#dce5eb')
        for ax,a,title in zip(axes[0],(coarse_z,display_fine,display_avg),
                ('GEBCO / V6 native 15 arc-sec','USGS 3DEP native ~10 m','3DEP averaged to V6 grid')):
            im=ax.imshow(a,origin='upper',extent=ext,cmap=cmap,vmin=vmin,vmax=vmax,interpolation='nearest')
            ax.set_title(title,fontsize=11)
        fig.colorbar(im,ax=list(axes[0]),label='Elevation (m); shared scale; native vertical datums',shrink=.85,pad=.02)
        def hill(a,step):
            dzdy,dzdx=np.gradient(a,step*111320,step*111320*np.cos(np.deg2rad(lat)))
            norm=np.sqrt(1+dzdx**2+dzdy**2)
            # East / north light components for azimuth 315 degrees, altitude 45 degrees.
            return np.clip((.5*dzdx+.5*dzdy+np.sqrt(.5))/norm,0,1)
        gray=plt.get_cmap('gray').copy(); gray.set_bad('#dce5eb')
        for ax,a,step,title in [(axes[1,0],coarse_z,res,'GEBCO hillshade'),(axes[1,1],display_fine,res/45,'3DEP hillshade')]:
            ax.imshow(hill(a,step),origin='upper',extent=ext,cmap=gray,vmin=0,vmax=1,interpolation='nearest'); ax.set_title(title,fontsize=11)
        ax=axes[1,2]; ax.imshow(display_fine,origin='upper',extent=ext,cmap=cmap,vmin=vmin,vmax=vmax,interpolation='nearest')
        local=faults[faults.intersects(box(*b))].copy(); local.geometry=local.geometry.intersection(box(*b))
        if len(local): local.plot(ax=ax,color='#df1747',linewidth=.8)
        m=known[r0:r1,c0:c1]
        if m.any() and not m.all(): ax.contour(xs[c0:c1],ys[r0:r1],m,levels=[.5],colors=['#00d5e5'],linewidths=.9)
        ax.set_title('Existing USGS lines (red) / V6 mask edge (cyan)',fontsize=10)
        for ax in axes.flat:
            ax.set(xlim=ext[:2],ylim=ext[2:],xlabel='Longitude',ylabel='Latitude'); ax.set_aspect(1/np.cos(np.deg2rad(lat))); ax.tick_params(labelsize=8)
        fig.suptitle(f'{label} | same geographic extent, north-up\nElevation-only comparison; hillshade derived from height (not imagery)',fontsize=14)
        fig.savefig(OUT/f'{name}.png',dpi=130); plt.close(fig)
        sample=dict(id=name,label=label,bbox=list(b),source=source,figure=f'{name}.png',
            coarse_shape=[N,N],fine_display_shape=list(fine.shape),fine_valid_percent=100*float(np.isfinite(fine).mean()),
            native_tif=f'rasters/{name}_3dep_native.tif',gebco_tif=f'rasters/{name}_gebco_v6.tif',
            averaged_tif=f'rasters/{name}_3dep_on_v6_grid.tif',fault_features=len(local),v6_known_fraction=float(m.mean()))
        samples.append(sample)

    # Two panels clearly distinguish catalog footprints from sampled raster validity.
    fig,axs=plt.subplots(1,2,figsize=(14,9),layout='constrained')
    ext=(bounds[0],bounds[2],bounds[1],bounds[3])
    colors=['#ffffff','#c3dce9','#e3ded2','#47a787','#db673e']
    for ax,cover,title in zip(axs,[catalog,valid],['Current product footprints','Sampled raster validity (COG overview)']):
        a=np.zeros((H,W),np.uint8); a[stored&~land]=1; a[land&~active]=2
        a[land&active&cover]=3; a[land&active&~cover]=4
        ax.imshow(a,origin='upper',extent=ext,cmap=ListedColormap(colors),vmin=0,vmax=4,interpolation='nearest')
        gpd.GeoSeries([ca_geom],crs=4326).boundary.plot(ax=ax,color='#444',linewidth=.6)
        ax.contour(xs,ys,active,levels=[.5],colors=['#666'],linewidths=.35)
        for s in samples:
            b=s['bbox']; ax.add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,color='#8b176a',linewidth=1.2))
            ax.text(b[0],b[3]+.04,s['id'].replace('_',' '),fontsize=7,color='#6a1252',ha='right' if b[0]>-118 else 'left')
        ax.set(xlim=ext[:2],ylim=ext[2:],title=title,xlabel='Longitude',ylabel='Latitude'); ax.set_aspect(1.25)
    fig.legend(handles=[Patch(color=c,label=l) for c,l in zip(colors[1:],['V6 non-positive elevation / water proxy','Stored land outside active splits','Active land covered','Active land missing'])],loc='outside lower center',ncol=2)
    fig.suptitle('V6 / USGS 3DEP ~10 m coverage audit\nLand proxy = V6 GEBCO elevation > 0; boundaries are not native 10m pixel QA')
    fig.savefig(OUT/'coverage.png',dpi=140); plt.close(fig)
    report['samples']=samples
    after=(DATA/'faults.h5').stat()
    assert (source_stat.st_size,source_stat.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    expected=json.loads((DATA/'qa/validation_report.json').read_text())['dataset_hdf5_sha256']
    report['v6_sha256']=sha(DATA/'faults.h5'); assert report['v6_sha256']==expected
    from PIL import Image
    for f in [OUT/'coverage.png']+[OUT/s['figure'] for s in samples]:
        with Image.open(f) as im: im.verify()
    for f in (OUT/'rasters').glob('*.tif'):
        with rio.open(f) as src:
            a=src.read(1,masked=True); assert src.crs and a.count()>0 and np.isfinite(a.compressed()).all()
    report['status']='complete'; report['training_started']=False
    report['artifacts_sha256']={str(f.relative_to(OUT)):sha(f) for f in list((OUT/'rasters').glob('*.tif'))+list(OUT.glob('*.png'))}
    save_json(OUT/'report.json',report)
    build_html(report)
    print('COMPLETE',OUT,flush=True)

def build_html(r):
    rows=''.join(f'<tr><td>{label}</td><td>{r["coverage"][key]["area_km2"]:,.0f}</td><td>{r["coverage"][key]["catalog_covered_percent"]:.2f}%</td><td>{r["coverage"][key]["overview_valid_percent"]:.2f}%</td></tr>' for key,label in [
        ('stored_positive_land','全部 339 图块中的陆地'),('active_positive_land','216 个有效图块中的陆地'),('active_california_positive_land','有效图块中的加州陆地'),('active_supervision','V6 人工监督 mask')])
    cards=''.join(f'<section><h2>{html.escape(s["label"])}</h2><p>范围：{s["bbox"]}；原生窗口有效数值覆盖 {s["fine_valid_percent"]:.2f}%。</p><a href="{s["figure"]}"><img src="{s["figure"]}"></a><p><a href="{s["native_tif"]}">下载原始 3DEP 高程裁剪 GeoTIFF</a> · <a href="{s["gebco_tif"]}">GEBCO 裁剪</a> · <a href="{s["averaged_tif"]}">3DEP 聚合至 V6 网格</a> · <a href="{s["source"]["url"]}">USGS 原始整瓦片</a></p><p>产品：{html.escape(s["source"]["title"])}</p></section>' for s in r['samples'])
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V6 · USGS 3DEP 10米与GEBCO对比</title><style>body{{font-family:system-ui,"Microsoft YaHei",sans-serif;max-width:1400px;margin:auto;padding:28px;background:#f3f5f7;color:#182c39;line-height:1.7}}section{{background:white;padding:24px;margin:24px 0;border-radius:12px}}img{{width:100%;height:auto}}a{{color:#146c91}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #ddd}}code,pre{{background:#edf1f4;padding:12px;white-space:pre-wrap;overflow-wrap:anywhere}}.note{{border-left:4px solid #b97920;padding:12px;background:#fff7e9}}</style>
    <h1>V6 研究区：USGS 3DEP 约10米 / GEBCO 对比预览</h1><p>只读检查与下载预览 · 未修改 V6、标签、模型或训练配置 · 未启动训练</p>
    <section><h2>覆盖检查</h2><p>V6 名义范围：125°W–116.5°W，31.5°N–41°N。共 339 个存储图块，其中 216 个有效图块（训练161 / 验证17 / 测试38）。统计按去重后的实际像元区域计算，不是把整个外接矩形当作研究区。</p>
    <table><tr><th>范围</th><th>面积约 km²</th><th>当前产品外框覆盖</th><th>概览栅格有效覆盖</th></tr>{rows}</table>
    <p>查询返回 {r['current_query_products']} 个当前产品，{r['intersecting_stored_products']} 个与已存储图块相交；{r['active_land_products']} 个与有效陆地相交，完整下载这些整瓦片约 {r['active_land_full_download_GB']:.2f} GB（目录文件大小估算）。本次仅读取概览和四处高分辨率窗口。</p>
    <p class="note">覆盖统计以 V6 的高程&gt;0作为原陆地定义。产品外框可能包含海洋或 NoData；概览有效覆盖只是在 V6 网格上的初步检查，并非整区逐个10米像元的完整审计。蓝色不表示海底数据；墨西哥部分和岛屿需单独检查。加州范围由 Census TIGERweb 州界裁切。</p><a href="coverage.png"><img src="coverage.png"></a></section>
    <section><h2>如何看对比图</h2><p>每张图上排为：原 GEBCO、约10米 3DEP、聚合回原 V6 网格的 3DEP。同一张图共用高程色标和地理范围。下排为两者相同光照参数的地形阴影，以及 USGS 断层矢量和 V6 mask 边界。</p><p>原 V6 为15角秒；3DEP为1/3角秒，每方向约细45倍。预览为北向上。原生高程 GeoTIFF 每个像元只有一个以米为单位的高度数值；阴影图由高程计算，并非卫星影像。</p><p class="note">3DEP 美国本土高程使用 NAVD88；GEBCO沿用原数据基准。此次未做垂直基准转换，所以不把两者差值解读成测量误差。灰蓝色为预览中非正高程或缺失区域，不等于新增精确海岸线。红线只用于定位，不代表10米精度的人工真值。</p></section>
    {cards}
    <section><h2>下载方法</h2><ol><li>打开 <a href="https://apps.nationalmap.gov/downloader/">The National Map Downloader</a>，圈定研究区。</li><li>选择 Elevation Products → 1/3 arc-second DEM → GeoTIFF，并使用 Current 当前产品。</li><li>按下方清单下载；文件名中的日期是产品日期，不能直接当作原始采集日期。Current API 返回的现行产品链接也可能位于 historical 文件夹，因此不要仅凭路径排除它们。</li></ol>
    <p><a href="download_urls_active_land.txt">有效陆地整瓦片下载链接清单</a> · <a href="download_inventory.json">带大小、日期、范围的产品清单</a> · <a href="current_query_url.txt">本次API查询地址</a> · <a href="report.json">覆盖统计与校验记录</a></p>
    <p>单个文件可用：</p><pre>curl.exe -L --fail --retry 3 -o USGS_tile.tif "从链接清单复制的完整URL"</pre><p>读取四处样本时已直接按 COG 窗口下载所需数据，无需下载全部整瓦片。复现脚本：项目根代码目录中的 preview_v6_3dep.py。</p>
    <p><a href="https://www.usgs.gov/3d-elevation-program/about-3dep-products-services">USGS 产品说明</a> · <a href="https://www.usgs.gov/faqs/there-api-accessing-national-map-data">官方API说明</a></p></section></html>'''
    (OUT/'index.html').write_text(page,encoding='utf-8')

if __name__=='__main__': main()
