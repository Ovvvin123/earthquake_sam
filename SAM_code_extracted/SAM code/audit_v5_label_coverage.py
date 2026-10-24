"""Read-only source audit; outputs review evidence, never a trusted supervision mask."""
from pathlib import Path
import csv
import json
import hashlib
import html
import numpy as np
import h5py
import geopandas as gpd
from shapely.geometry import box, mapping
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
DATA = CODE / 'earth_dataset_v3'
OUT = CODE / 'diagnostics/v5_label_coverage_audit'
OUT.mkdir(parents=True, exist_ok=True)
man = json.loads((DATA/'manifest.json').read_text())
meta = {t['key']: t for t in man['tiles']}
lists = {s: (DATA/f'key_{s}.lst').read_text().split() for s in ['train','val','test']}
split = {k: s for s, ks in lists.items() for k in ks}
res = man['provenance']['res_deg']; T = man['provenance']['tile']
first = man['tiles'][0]; i, j = map(int, first['key'].split('_')[1:])
x0 = first['lon_c'] - ((T-1)/2+j)*res
y0 = first['lat_c'] - ((T-1)/2+i)*res
H = max(int(k.split('_')[1])+T for k in meta)
W = max(int(k.split('_')[2])+T for k in meta)
z = np.full((H,W), np.nan, np.float32)
used = np.zeros((H,W),bool); positive = used.copy()
fields = {}; rows = []; mask_count = 0
with h5py.File(DATA/'faults.h5','r') as f:
    for key in meta:
        g=f[key]; i,j=map(int,key.split('_')[1:]); a=g['image'][:]
        z[i:i+T,j:j+T]=a
        if key not in split: continue
        fields[key]=list(g.keys()); mask_count += 'label_known' in g
        p=g['k0'][:]<.01
        used[i:i+T,j:j+T]=True; positive[i:i+T,j:j+T] |= p
        valid=np.isfinite(a)&(a>-2147483647)
        bins={'land_proxy':valid&(a>=0), 'shallow_proxy':valid&(a<0)&(a>=-200), 'deep_proxy':valid&(a<-200)}
        row={'key':key,'split':split[key], 'lon':meta[key]['lon_c'],'lat':meta[key]['lat_c'],
             'label_positive_pixels':int(p.sum()), 'label_known_present':'label_known' in g}
        for name,m in bins.items():
            row[name+'_fraction']=float(m.sum()/max(valid.sum(),1))
            row[name+'_label_fraction']=float((p&m).sum()/max(m.sum(),1))
        rows.append(row)

bounds=box(x0-res/2,y0-res/2,x0+(W-.5)*res,y0+(H-.5)*res)
catalogs={}; frames={}
for name in ['ca_offshore','Qfaults_US_Database','fault_areas']:
    path=ROOT/f'Qfaults_GIS/SHP/{name}.shp'
    g=gpd.read_file(path).to_crs(4326)
    g=g[g.geometry.intersects(bounds)].copy()
    frames[name]=g
    catalogs[name]={'features_in_grid_bbox':len(g),'geometry_types':g.geom_type.value_counts().to_dict(),
       'fields':list(g.columns),'attribute_counts':{}}
    for col in ['LINE_TYPE','EXPRESSION','MAPPED_SCA','FLT_AGE','FLT_SOURCE','Location','linetype','certainty','class','name']:
        if col in g:
            catalogs[name]['attribute_counts'][col]=g[col].fillna('MISSING').astype(str).value_counts().head(30).to_dict()
    g.to_file(OUT/f'{name}_bbox.geojson',driver='GeoJSON')

def raster_points(geom):
    """Dense samples mapped to the same geographic pixel grid; screening only."""
    geom=geom.intersection(bounds)
    lines=list(geom.geoms) if hasattr(geom,'geoms') else [geom]
    pts=[]
    for ln in lines:
        if ln.geom_type != 'LineString': continue
        xy=np.asarray(ln.coords)
        for a,b in zip(xy[:-1],xy[1:]):
            n=max(2,int(np.linalg.norm(b-a)/(res*.5))+1)
            pts.append(np.linspace(a,b,n))
    if not pts: return np.array([],int),np.array([],int)
    p=np.vstack(pts); jj=np.rint((p[:,0]-x0)/res).astype(int); ii=np.rint((p[:,1]-y0)/res).astype(int)
    good=(ii>=0)&(ii<H)&(jj>=0)&(jj<W)
    return ii[good],jj[good]

external=np.zeros_like(used); feature_evidence=[]
for idx, row in frames['Qfaults_US_Database'].iterrows():
    ii,jj=raster_points(row.geometry)
    if len(ii)==0: continue
    hit=used[ii,jj]&~positive[ii,jj]
    if not hit.any(): continue
    external[ii[hit],jj[hit]]=True
    feature_evidence.append({'source_index':int(idx), 'fault_name':str(row.get('fault_name','')),
        'fault_id':str(row.get('fault_id','')), 'fault_url':str(row.get('fault_url','')),
        'linetype':str(row.get('linetype','')), 'certainty':str(row.get('certainty','')),
        'class':str(row.get('class','')), 'ref_id':str(row.get('ref_id',''))})

offshore_raster=np.zeros_like(used)
for geom in frames['ca_offshore'].geometry:
    ii,jj=raster_points(geom); offshore_raster[ii,jj]=True

features=[]; source_rows=[]
for row in rows:
    k=row['key']; i,j=map(int,k.split('_')[1:]); e=external[i:i+T,j:j+T]
    row['external_catalog_uncovered_trace_pixels']=int(e.sum())
    reasons=[]
    if e.any(): reasons.append('external_catalog_trace_outside_V5_positive_buffer')
    if row['land_proxy_fraction']>.01: reasons.append('land_without_documented_annotation_coverage')
    if row['deep_proxy_fraction']>.01: reasons.append('deep_water_without_documented_annotation_coverage')
    if row['shallow_proxy_fraction']>.01: reasons.append('shallow_water_completeness_unverified')
    if row['label_positive_pixels']==0: reasons.append('empty_catalog_label_is_not_verified_negative')
    row['review_reasons']=';'.join(reasons)
    row['review_status']='UNREVIEWED'; row['trusted_negative_coverage']='NOT_ESTABLISHED'
    footprint=box(x0+(j-.5)*res,y0+(i-.5)*res,x0+(j+T-.5)*res,y0+(i+T-.5)*res)
    features.append({'type':'Feature','geometry':mapping(footprint),'properties':row.copy()})
    src=frames['ca_offshore'][frames['ca_offshore'].intersects(footprint)]
    source_rows.append(dict(key=k,split=row['split'],lon=row['lon'],lat=row['lat'],source_feature_count=len(src),
        line_types='; '.join(sorted(set(src.LINE_TYPE.dropna()))),
        source_refs=' | '.join(sorted(set(src.FLT_SOURCE.dropna()))),
        evidence_types=' | '.join(sorted(set(src.EXPRESSION.dropna()))),
        scales='; '.join(sorted(set(src.MAPPED_SCA.dropna())))))
rows.sort(key=lambda r:(-r['external_catalog_uncovered_trace_pixels'],-r['land_proxy_fraction']))
with (OUT/'tile_review_queue.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (OUT/'source_review_by_tile.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(source_rows[0]));w.writeheader();w.writerows(source_rows)
(OUT/'tile_review_queue.geojson').write_text(json.dumps({'type':'FeatureCollection','features':features}),encoding='utf-8')
(OUT/'external_catalog_review.json').write_text(json.dumps(feature_evidence,indent=2),encoding='utf-8')
bins={'land_proxy':used&(z>=0),'shallow_proxy':used&(z<0)&(z>=-200),'deep_proxy':used&(z<-200)}
summary={name:{'unique_pixels':int(m.sum()),'fraction_of_used_pixels':float(m.sum()/used.sum()),
    'V5_label_positive_pixels':int((m&positive).sum()),'V5_positive_fraction_in_zone':float((m&positive).sum()/max(m.sum(),1)),
    'external_uncovered_trace_pixels':int((m&external).sum()),'offshore_source_trace_pixels':int((m&offshore_raster).sum())} for name,m in bins.items()}
report={'status':'Coverage not established; no label_known generated',
    'used_tiles':len(rows),'label_known_present_tiles':mask_count,'split_counts':{s:len(v) for s,v in lists.items()},
    'zone_definition':'Descriptive elevation bins only: land >=0 m, shallow [-200,0), deep <-200; NOT annotation confidence or coastline truth.',
    'pixel_statistics':'Union of selected tile footprints, overlap counted once; existing k0<0.01 buffers unioned.',
    'zones':summary,'catalogs':catalogs,'tiles_with_external_uncovered_trace':sum(r['external_catalog_uncovered_trace_pixels']>0 for r in rows),
    'external_features_for_review':len(feature_evidence),'h5_fields':sorted(set(v for names in fields.values() for v in names)),
    'caveats':['External line samples outside old positive buffers are candidates for review, not automatically validated missing ground truth.',
       'fault_areas polygons have fault identifiers and ages; no evidence they represent surveyed negative coverage.',
       'No trustworthy background can be derived from water depth, catalogue envelope or absence of lines alone.',
       'This audit does not determine completeness of any individual source publication.'],
    'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [DATA/'manifest.json',*[DATA/f'key_{s}.lst' for s in lists]]},
    'h5_stat':{'bytes':(DATA/'faults.h5').stat().st_size,'mtime_ns':(DATA/'faults.h5').stat().st_mtime_ns}}
(OUT/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

extent=[x0-res/2,x0+(W-.5)*res,y0-res/2,y0+(H-.5)*res]
zones=np.full((H,W),np.nan)
for n,m in enumerate(bins.values()): zones[m]=n
fig,axs=plt.subplots(1,2,figsize=(13,8),constrained_layout=True)
axs[0].imshow(zones,origin='lower',extent=extent,cmap=ListedColormap(['#d6b488','#8bcbd4','#537ba4']),vmin=0,vmax=2)
frames['ca_offshore'].plot(ax=axs[0],color='#a11536',linewidth=.55)
axs[0].set_title('V5 footprint: land / 0-200 m / >200 m\nRed: source fault lines; colors are NOT known-label masks')
axs[1].imshow(np.ma.masked_where(~used,z),origin='lower',extent=extent,cmap='terrain',vmin=-4000,vmax=2000)
yy,xx=np.where(external)
axs[1].scatter(x0+xx*res,y0+yy*res,s=.2,c='#c400db',rasterized=True)
axs[1].set_title('Magenta: other catalog traces outside V5 labels\nReview evidence only; no automatic relabeling')
for ax in axs: ax.set(xlim=extent[:2],ylim=extent[2:],xlabel='Longitude',ylabel='Latitude');ax.set_aspect(1/np.cos(np.deg2rad(36)))
fig.savefig(OUT/'coverage_review.png',dpi=160);plt.close(fig)

# A small, spatially varied set of review panels. Full queue remains available.
examples=[]
for row in rows:
    if row['external_catalog_uncovered_trace_pixels']==0: continue
    if any(abs(row['lat']-r['lat'])<.8 and abs(row['lon']-r['lon'])<.8 for r in examples):continue
    examples.append(row)
    if len(examples)==6:break
with h5py.File(DATA/'faults.h5','r') as f:
    for row in examples:
        key=row['key'];g=f[key];i,j=map(int,key.split('_')[1:]);a=g['image'][:];p=g['k0'][:]<.01
        ex=[x0+(j-.5)*res,x0+(j+T-.5)*res,y0+(i-.5)*res,y0+(i+T-.5)*res]
        fig,ax=plt.subplots(figsize=(7,6),constrained_layout=True)
        ax.imshow(g['TW'][0],origin='lower',extent=ex,cmap='gray')
        rgba=np.zeros((T,T,4));rgba[p]=[1,.1,.1,.5]
        ax.imshow(rgba,origin='lower',extent=ex)
        yy,xx=np.where(external[i:i+T,j:j+T]);ax.scatter(x0+(j+xx)*res,y0+(i+yy)*res,s=2,c='#00dbff')
        if a.min()<0<a.max():ax.contour(a,levels=[0],origin='lower',extent=ex,colors=['#fff000'],linewidths=.7)
        ax.set_title(f'{key} ({row["split"]})\nRed: V5 labels; cyan: external review candidates; yellow: elevation 0 m')
        ax.set(xlabel='Longitude',ylabel='Latitude');fig.savefig(OUT/f'{key}.png',dpi=140);plt.close(fig)
cards=''.join(f'<section><h2>{html.escape(r["key"])} · {r["split"]}</h2><img src="{r["key"]}.png"></section>' for r in examples)
(OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>V5 标注覆盖审查</title><style>body{font:17px sans-serif;max-width:1200px;margin:30px auto;padding:20px}img{max-width:100%}section{display:inline-block;vertical-align:top;width:49%}</style><h1>V5 标注覆盖审查</h1><p>本页不是 label_known 掩膜。红色为现有标签，青色为另一目录提示需核查的线段，黄色为高程零米线。目录差异不直接作为真值。</p><p>浅海采用 0–200 米仅用于统计；水深不能证明标注完整性。未修改训练数据。</p><a href="tile_review_queue.csv">全部图块核查清单 CSV</a> · <a href="tile_review_queue.geojson">图块位置 GeoJSON</a> · <a href="audit.json">审查统计</a><img src="coverage_review.png">'+cards,encoding='utf-8')
print(json.dumps({k:report[k] for k in ['used_tiles','label_known_present_tiles','zones','tiles_with_external_uncovered_trace','external_features_for_review']},indent=2))
print('Examples:',[r['key'] for r in examples])
print('Output:',OUT)
