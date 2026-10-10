"""Map actual stride-64 train/validation tile footprints from the final manifest."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
from shapely.geometry import box, mapping
from shapely.ops import unary_union

ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'SAM_code_extracted/SAM code'
sys.path.insert(0,str(CODE))
from prepare_greece_nz import DEFAULT_DATA_DIR, footprint, overlap_audit

OUT=ROOT/'input_channels_figures/greece_nz_actual_selection'
OUT.mkdir(exist_ok=True)
manifest=DEFAULT_DATA_DIR/'manifest.json'
man=json.loads(manifest.read_text());rows=man['tiles'];prov=man['provenance']
assert overlap_audit(rows)['train_val_overlap_pairs']==0
for split in ['train','val']:
    assert set((DEFAULT_DATA_DIR/f'key_{split}.lst').read_text().split())=={t['key'] for t in rows if t['split']==split}

land_path=ROOT/'input_channels_figures/gem_candidates_20260930/ne_10m_land.zip'
land=gpd.read_file('zip://'+str(land_path)).to_crs(4326)
font_manager.fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font_manager.FontProperties(fname='/mnt/c/Windows/Fonts/msyh.ttc').get_name(),
                     'axes.unicode_minus':False,'font.size':11,'savefig.facecolor':'white',
                     'svg.fonttype':'path'})
COLORS={'train':'#1976b9','val':'#ef8c24'}
TITLES={'greece_aegean':'希腊—爱琴海窗口','new_zealand':'新西兰窗口'}
DESCRIPTIONS={'greece_aegean':'选取范围：19–29°E，34–42°N',
              'new_zealand':'选取范围：166–179°E，47–34°S'}


def label(ax,x,y,text):
    ax.text(x,y,text,ha='center',va='center',fontsize=11,color='#394c59',zorder=10,
            path_effects=[patheffects.withStroke(linewidth=3,foreground='white',alpha=.85)])


def draw(ax,region):
    b=prov['regions'][region]['bbox']
    padding=.55 if region=='greece_aegean' else .7
    view=box(b[0]-padding,b[1]-padding,b[2]+padding,b[3]+padding)
    local=land[land.intersects(view)].copy();local.geometry=local.geometry.intersection(view)
    ax.set_facecolor('#edf5fb')
    local.plot(ax=ax,facecolor='#eeece5',edgecolor='#82939a',linewidth=.55,zorder=1)
    subset=[t for t in rows if t['region']==region]
    counts=Counter(t['split'] for t in subset)
    for split in ['train','val']:
        tiles=[box(*footprint(t)) for t in subset if t['split']==split]
        union=unary_union(tiles)
        gpd.GeoSeries([union],crs=4326).plot(ax=ax,color=COLORS[split],alpha=.36,edgecolor='none',zorder=2)
        for t in subset:
            if t['split']==split:
                x,y,X,Y=footprint(t)
                ax.add_patch(Rectangle((x,y),X-x,Y-y,facecolor='none',edgecolor=COLORS[split],
                                       lw=.28,alpha=.22,zorder=3))
        gpd.GeoSeries([union],crs=4326).boundary.plot(ax=ax,color=COLORS[split],linewidth=1.5,zorder=4)
    faults=gpd.read_file(DEFAULT_DATA_DIR/f'{region}_labels.geojson').to_crs(4326)
    faults.geometry=faults.geometry.intersection(box(*b))
    faults[~faults.geometry.is_empty].plot(ax=ax,color='#a14740',linewidth=.5,alpha=.72,zorder=5)
    ax.add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],facecolor='none',
                           edgecolor='#253c4c',linewidth=1.4,linestyle=(0,(5,3)),zorder=6))
    ax.set_xlim(b[0]-padding,b[2]+padding);ax.set_ylim(b[1]-padding,b[3]+padding)
    ax.set_aspect(1/np.cos(np.radians((b[1]+b[3])/2)))
    ax.xaxis.set_major_locator(MultipleLocator(2));ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}°E'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{abs(v):g}°'+('N' if v>=0 else 'S')))
    ax.grid(alpha=.22,linewidth=.5,color='#71818d',zorder=0)
    ax.set_xlabel('经度');ax.set_ylabel('纬度')
    for spine in ax.spines.values(): spine.set_color('#a1aeb6')
    ax.set_title(f'{TITLES[region]}\n训练 {counts["train"]} 块  ·  验证 {counts["val"]} 块\n{DESCRIPTIONS[region]}',fontsize=13,pad=13)
    if region=='greece_aegean':
        label(ax,22.0,38.7,'希腊');label(ax,25.2,37.1,'爱琴海')
        label(ax,27.9,39.0,'土耳其西部');label(ax,24.9,35.2,'克里特岛')
    else:
        label(ax,176.8,-37.0,'北岛');label(ax,169.9,-44.8,'南岛')
    ax.annotate('N',xy=(.94,.94),xytext=(.94,.85),xycoords='axes fraction',
                ha='center',va='center',color='#263c4a',fontsize=12,
                arrowprops={'arrowstyle':'-|>','color':'#263c4a','lw':1.2},zorder=12)


def legend(fig):
    fig.legend(handles=[Patch(facecolor=COLORS['train'],alpha=.55,label='实际训练切块覆盖区'),
                        Patch(facecolor=COLORS['val'],alpha=.55,label='实际验证切块覆盖区'),
                        Line2D([0],[0],color='#a14740',lw=1,label='筛选后 GEM 断层迹线'),
                        Line2D([0],[0],color='#253c4c',ls='--',lw=1.4,label='原始选取窗口')],
               loc='lower center',bbox_to_anchor=(.5,.093),ncol=4,frameon=False,fontsize=10)


def footer(fig):
    fig.text(.5,.067,'着色仅表示最终进入数据集的 384×384 切块足迹，并非整个国家；细线为单块边界。步长 64，同集合内部有重叠。',
             ha='center',fontsize=10,color='#475965')
    fig.text(.5,.040,'按完整 4°×4°地理块分配训练／验证，跨块及不满足标签筛选的切片已剔除；训练与验证足迹重叠为零。',
             ha='center',fontsize=10,color='#475965')
    fig.text(.5,.014,'来源：本次最终 manifest 与 train/val 清单；GEM 本地筛选标签；Natural Earth 1:10m 陆地底图。经纬度 WGS84。',
             ha='center',fontsize=9,color='#64747e')


def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=200)
    fig.savefig(OUT/f'{name}.svg')
    plt.close(fig)


fig,axes=plt.subplots(1,2,figsize=(16,10.5))
for ax,region in zip(axes,['greece_aegean','new_zealand']): draw(ax,region)
fig.suptitle('本次训练实际选择了哪些地区？',fontsize=22,y=.97)
fig.subplots_adjust(left=.055,right=.975,top=.82,bottom=.18,wspace=.21)
legend(fig);footer(fig);save(fig,'selected_regions_overview')
for region in TITLES:
    fig,ax=plt.subplots(figsize=(12,11))
    draw(ax,region)
    fig.subplots_adjust(left=.09,right=.95,top=.86,bottom=.18)
    legend(fig);footer(fig);save(fig,region+'_selection')

features=[{'type':'Feature','properties':{k:t[k] for k in ['key','region','split','spatial_cell']},
           'geometry':mapping(box(*footprint(t)))} for t in rows]
(OUT/'actual_tile_footprints.geojson').write_text(json.dumps({'type':'FeatureCollection','features':features},ensure_ascii=False),encoding='utf-8')
report={'source_manifest':str(manifest),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'source_land':str(land_path),'counts':dict(Counter(t['split'] for t in rows)),
        'train_val_overlap_pairs':0,'regions':{}}
for region in TITLES:
    subset=[t for t in rows if t['region']==region]
    report['regions'][region]={'selection_bbox':prov['regions'][region]['bbox'],
                               'counts':dict(Counter(t['split'] for t in subset)),
                               'split_union_bounds':{split:list(unary_union([box(*footprint(t)) for t in subset if t['split']==split]).bounds)
                                                     for split in ['train','val']}}
(OUT/'map_provenance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
print('Saved:',OUT,flush=True)
