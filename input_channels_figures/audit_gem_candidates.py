"""Audit local GEM candidate windows; approximate coast classification, not coverage."""
from pathlib import Path
import json, urllib.request, hashlib
import geopandas as gpd
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union, transform
from pyproj import Transformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'input_channels_figures/gem_candidates_20260930'
OUT.mkdir(exist_ok=True)
src=ROOT/'GEBCO_Global/GEM_label/gem-global-active-faults-master/geojson/gem_active_faults_harmonized.geojson'
gem=gpd.read_file(src).to_crs(4326)
archive=OUT/'ne_10m_land.zip'
url='https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_land.zip'
if not archive.exists():
    with urllib.request.urlopen(url,timeout=60) as response:
        archive.write_bytes(response.read())
land=gpd.read_file('zip://'+str(archive)).to_crs(4326)
cfg=json.loads((ROOT/'SAM_code_extracted/SAM code/regions_global.json').read_text())['regions']
names={'turkey_naf':'土耳其 / 北安纳托利亚','himalaya_tibet':'喜马拉雅—青藏','new_zealand':'新西兰','japan':'日本','italy':'意大利','greece_aegean':'希腊—爱琴海'}
report={'source':str(src),'sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'total_features':len(gem),'fields':list(gem.columns),'coast_source':url,'method':'Clip traces to configured bbox. Project each region to local AEQD. Partition trace length into land >5 km from coast, coastal band +/-5 km, and sea >5 km from mapped land. Natural Earth 1:10 million is generalized, not 10 m resolution. Length fractions describe catalog traces, NOT survey completeness, pixel fractions or GEBCO accuracy.','regions':{}}
font_manager.fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font_manager.FontProperties(fname='/mnt/c/Windows/Fonts/msyh.ttc').get_name(),'axes.unicode_minus':False,'font.size':10})
fig,axes=plt.subplots(2,3,figsize=(17,11))
for ax,(name,title) in zip(axes.ravel(),names.items()):
    r=next(r for r in cfg if r['name']==name);b=r['bbox'];window=box(*b)
    subset=gem[gem.intersects(window)].copy()
    subset.geometry=subset.geometry.intersection(window)
    lat0=(b[1]+b[3])/2;lon0=(b[0]+b[2])/2
    crs=f'+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m'
    project=Transformer.from_crs(4326,crs,always_xy=True).transform
    inverse=Transformer.from_crs(crs,4326,always_xy=True).transform
    big=box(b[0]-2,b[1]-2,b[2]+2,b[3]+2)
    ld=land[land.intersects(big)].copy();ld.geometry=ld.geometry.intersection(big)
    lp=unary_union([transform(project,g) for g in ld.geometry])
    inside=lp.buffer(-5000);expanded=lp.buffer(5000)
    inland=[];coastal=[];offshore=[]
    for geom in subset.geometry:
        p=transform(project,geom)
        a=p.intersection(inside);o=p.difference(expanded);c=p.difference(inside).intersection(expanded)
        inland.append(a);offshore.append(o);coastal.append(c)
    lens=[sum(g.length for g in arr) for arr in [inland,coastal,offshore]]
    total=sum(lens)
    fields={}
    for f in ['activity_confidence','exposure_quality','epistemic_quality','accuracy']:
        if f in subset:
            vals=subset[f].replace('',np.nan)
            fields[f]={'nonmissing':int(vals.notna().sum()),'counts':{str(k):int(v) for k,v in vals.value_counts(dropna=False).items()}}
        else:fields[f]={'nonmissing':0,'counts':{'FIELD_ABSENT':len(subset)}}
    item={'bbox':b,'features':len(subset),'trace_length_km':round(total/1000,2),'length_percent':dict(zip(['inland_gt5km','coastal_within5km','offshore_gt5km'],[round(x/total*100,2) for x in lens])),'catalog_names':{str(k):int(v) for k,v in subset['catalog_name'].value_counts(dropna=False).items()},'quality_fields':fields,'slip_types':{str(k):int(v) for k,v in subset['slip_type'].value_counts(dropna=False).items()}}
    eligible=(subset['catalog_name'].fillna('')!='Bird 2003') & ~subset['slip_type'].fillna('').isin(['Anticline','Syncline'])
    filtered_lens=[sum(g.length for g,ok in zip(arr,eligible) if ok) for arr in [inland,coastal,offshore]]
    filtered_total=sum(filtered_lens)
    item['excluding_Bird2003_and_folds']={'features':int(eligible.sum()),'length_percent':dict(zip(['inland_gt5km','coastal_within5km','offshore_gt5km'],[round(x/filtered_total*100,2) for x in filtered_lens]))}
    item['offshore_km_by_catalog']={str(c):round(sum(g.length for g,cat in zip(offshore,subset['catalog_name']) if cat==c)/1000,2) for c in subset['catalog_name'].dropna().unique()}
    report['regions'][name]=item
    ax.set_facecolor('#e9f3fa');ld.plot(ax=ax,color='#eeeade',edgecolor='#909f9e',linewidth=.35)
    for arr,color in zip([inland,coastal,offshore],['#c54d36','#d3a114','#236bb0']):
        gs=[transform(inverse,g) for g in arr if not g.is_empty]
        if gs:gpd.GeoSeries(gs,crs=4326).plot(ax=ax,color=color,linewidth=.7)
    ax.set_xlim(b[0],b[2]);ax.set_ylim(b[1],b[3]);ax.set_aspect(1/np.cos(np.radians(lat0)))
    ax.set_title(f'{title} | {len(subset)} 条要素\n内陆 {lens[0]/total:.0%} · 近岸 {lens[1]/total:.0%} · 外海 {lens[2]/total:.0%}',fontsize=11)
    ax.set_xlabel('经度');ax.set_ylabel('纬度');ax.grid(alpha=.2)
    print(json.dumps({name:item},ensure_ascii=False),flush=True)
fig.suptitle('本地 GEM 候选区域：断层迹线的海陆分布',fontsize=20,y=.985)
fig.legend(handles=[Line2D([0],[0],color=c,label=l,lw=2) for c,l in zip(['#c54d36','#d3a114','#236bb0'],['陆地且距岸 >5 km','海岸两侧 5 km 范围','海侧且距岸 >5 km'])],loc='lower center',bbox_to_anchor=(.5,.058),ncol=3)
fig.text(.04,.036,'比例按窗口内迹线长度计算；不表示区域标注完整率。近岸带用于降低概化海岸线误差，不能作为精确海陆标签。',fontsize=10)
fig.text(.04,.014,'来源：本地 GEM GAF-DB harmonized（CC BY-SA 4.0）；Natural Earth 1:1000万陆地底图。统计日期：2026-09-30。',fontsize=9,color='#536773')
fig.subplots_adjust(left=.055,right=.98,top=.91,bottom=.145,hspace=.38,wspace=.28)
fig.savefig(OUT/'gem_candidate_land_sea.png',dpi=160)
fig.savefig(OUT/'gem_candidate_land_sea.svg')
(OUT/'candidate_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print('DONE',flush=True)
