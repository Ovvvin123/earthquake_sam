"""Generate a review-only coastal envelope from source geometry; never write label_known."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import geopandas as gpd
import h5py
from shapely import contains_xy
from shapely.ops import unary_union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--buffer-km',type=float,default=8)
ap.add_argument('--closing-km',type=float,default=4)
args=ap.parse_args()
if not (0<args.buffer_km<=30 and 0<=args.closing_km<=15): ap.error('Invalid preview distance')
CODE=Path(__file__).resolve().parent; ROOT=CODE.parents[1]
DATA=CODE/'earth_dataset_v3'
OUT=CODE/'diagnostics/v5_coastal_envelope_preview_v1'
OUT.mkdir(parents=True,exist_ok=True)
source_h5_stat=(DATA/'faults.h5').stat()
man=json.loads((DATA/'manifest.json').read_text())
keys=set().union(*[(DATA/f'key_{s}.lst').read_text().split() for s in ['train','val','test']])
res=man['provenance']['res_deg'];T=man['provenance']['tile']
first=man['tiles'][0];ii,jj=map(int,first['key'].split('_')[1:])
x0=first['lon_c']-((T-1)/2+jj)*res;y0=first['lat_c']-((T-1)/2+ii)*res
H=max(int(t['key'].split('_')[1])+T for t in man['tiles'])
W=max(int(t['key'].split('_')[2])+T for t in man['tiles'])
z=np.full((H,W),np.nan,np.float32);used=np.zeros((H,W),bool);positive=used.copy()
with h5py.File(DATA/'faults.h5','r') as f:
    for key in keys:
        i,j=map(int,key.split('_')[1:]);g=f[key]
        z[i:i+T,j:j+T]=g['image'][:]
        used[i:i+T,j:j+T]=True
        positive[i:i+T,j:j+T] |= g['k0'][:]<.01
valid=used&np.isfinite(z)&(z>-2147483647)
g=gpd.read_file(ROOT/'Qfaults_GIS/SHP/ca_offshore.shp').to_crs(4326)
metric=g.to_crs(3310)
envelope=unary_union(metric.geometry.buffer(args.buffer_km*1000))
if args.closing_km:
    envelope=envelope.buffer(args.closing_km*1000).buffer(-args.closing_km*1000)
geographic=gpd.GeoSeries([envelope],crs=3310).to_crs(4326).iloc[0]
xs=x0+np.arange(W)*res;ys=y0+np.arange(H)*res
inside=np.zeros((H,W),bool)
for start in range(0,H,128):
    xx,yy=np.meshgrid(xs,ys[start:start+128]);inside[start:start+128]=contains_xy(geographic,xx,yy)
# In-memory preview classification only. No raster mask or HDF5 field is exported.
candidate=inside&valid&(z<0)
assert not (candidate & (z>=0)).any()
assert not (candidate & ~used).any()
zones=np.full((H,W),np.nan)
zones[valid&(z>=0)]=0;zones[valid&(z<0)&(z>=-200)]=1;zones[valid&(z<-200)]=2
extent=[x0-res/2,x0+(W-.5)*res,y0-res/2,y0+(H-.5)*res]
font=FontProperties(fname='/mnt/c/Windows/Fonts/msyh.ttc')
plt.rcParams['axes.unicode_minus']=False
rgba=np.zeros((H,W,4),np.float32);rgba[candidate]=[.75,1.,.05,.42]
cmap=ListedColormap(['#d6b488','#8bcbd4','#537ba4'])
def draw(ax,overlay,xlim,ylim):
    ax.imshow(zones,origin='lower',extent=extent,cmap=cmap,vmin=0,vmax=2,interpolation='nearest')
    if overlay:
        ax.imshow(rgba,origin='lower',extent=extent,interpolation='nearest')
        ax.contour(xs,ys,candidate.astype(float),levels=[.5],colors=['#007e3e'],linewidths=1.05)
    g.plot(ax=ax,color='#b11736',linewidth=.6,zorder=4)
    ax.set(xlim=xlim,ylim=ylim,xlabel='Longitude',ylabel='Latitude')
    ax.set_aspect(1/np.cos(np.deg2rad(36)))
    ax.tick_params(labelsize=9)
    ax.grid(alpha=.13)
fig,axs=plt.subplots(1,2,figsize=(14,10))
for ax,over in zip(axs,[False,True]):draw(ax,over,extent[:2],extent[2:])
axs[0].set_title('现有 V5 覆盖与原始断层线',fontproperties=font,fontsize=15)
axs[1].set_title('候选包围范围 v1（待确认）',fontproperties=font,fontsize=15)
handles=[Patch(color='#d6b488',label='高程 ≥0 m'),Patch(color='#8bcbd4',label='−200 至 0 m'),
         Patch(color='#537ba4',label='低于 −200 m'),Line2D([0],[0],color='#b11736',label='目录断层线'),
         Patch(facecolor='#c4e879',edgecolor='#007e3e',label='候选范围：浅绿色填充／深绿色边界')]
fig.legend(handles=handles,loc='lower center',ncol=3,prop=font,bbox_to_anchor=(.5,.035))
fig.suptitle('沿岸断层候选监督范围 · 仅预览，不是已核查的标注覆盖面',fontproperties=font,fontsize=18,y=.98)
fig.text(.5,.015,f'断层线外扩 {args.buffer_km:g} km，合并近邻空隙；剔除高程非负区域及 V5 覆盖外区域。没有按 200 m 水深截断。',fontproperties=font,ha='center',fontsize=10)
fig.subplots_adjust(top=.92,bottom=.13,left=.065,right=.98,wspace=.16)
fig.savefig(OUT/'coastal_envelope_overview.png',dpi=180);plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(16,7))
views=[('北段：海岸附近',[-124.7,-122.4],[38.8,40.5]),('中段：湾区至中部海岸',[-123.6,-120.2],[35.5,38.6]),
       ('南段：海湾与近海断层群',[-121.1,-117.0],[32.0,35.1])]
for ax,(title,xlim,ylim) in zip(axs,views):
    draw(ax,True,xlim,ylim);ax.set_title(title,fontproperties=font,fontsize=13)
fig.suptitle('候选范围局部放大：红线为已有目录，绿色为待确认范围',fontproperties=font,fontsize=17)
fig.text(.5,.035,'保留不连续区域和空洞；白色表示当前 V5 未使用的范围。浅绿色区域仍未经过标注完整性核查。',fontproperties=font,ha='center',fontsize=11)
fig.subplots_adjust(top=.87,bottom=.12,left=.04,right=.99,wspace=.23)
fig.savefig(OUT/'coastal_envelope_details.png',dpi=180);plt.close(fig)
stat={'status':'PREVIEW_ONLY_NOT_APPROVED','recipe':{'buffer_km':args.buffer_km,'closing_km':args.closing_km,'metric_crs':'EPSG:3310',
      'exclude':'nonfinite/no-data, elevation>=0, outside current train/val/test footprint','depth_cutoff':None},
      'used_pixels':int(valid.sum()),'candidate_pixels':int(candidate.sum()),'candidate_fraction':float(candidate.sum()/valid.sum()),
      'candidate_nonnegative_elevation_pixels':int((candidate&(z>=0)).sum()),
      'label_positive_pixels':int((positive&valid).sum()),'positive_pixels_inside_candidate':int((positive&candidate).sum()),
      'positive_pixels_excluded_nonnegative_elevation':int((positive&valid&(z>=0)).sum()),
      'marine_positive_pixels_outside_candidate':int((positive&valid&(z<0)&~candidate).sum()),
      'candidate_deeper_than_200m_pixels':int((candidate&(z<-200)).sum()),
      'coverage_note':'Pixel counts deduplicate overlapping tiles; not equal-area statistics. Sea/land filtering is only an elevation proxy.',
      'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'Qfaults_GIS/SHP/ca_offshore.shp',DATA/'manifest.json',*[DATA/f'key_{s}.lst' for s in ['train','val','test']]]},
      'writes_training_mask':False}
after=(DATA/'faults.h5').stat()
assert (source_h5_stat.st_size,source_h5_stat.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
(OUT/'preview_recipe.json').write_text(json.dumps(stat,indent=2),encoding='utf-8')
(OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>V5 沿岸候选范围</title><style>body{font:17px sans-serif;margin:30px auto;max-width:1500px;padding:20px}img{width:100%}</style><h1>V5 沿岸候选范围 v1 · 待确认</h1><p>浅绿色填充、深绿色边界是候选范围，红线是已有断层。范围由断层线外扩 8 km 后合并近邻空隙，再排除高程非负区域和当前 V5 覆盖以外区域得到。它不是调查完整性证明，尚未制作 label_known。</p><p>保留已有断层附近的深水区域；没有用 200 m 水深作为监督边界。白色空缺来自当前 V5 图块覆盖。</p><img src="coastal_envelope_overview.png"><img src="coastal_envelope_details.png"><p><a href="preview_recipe.json">生成参数与统计</a></p>',encoding='utf-8')
print(json.dumps(stat,indent=2));print(OUT)
