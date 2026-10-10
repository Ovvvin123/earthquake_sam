"""Geographic candidate guide; does NOT represent verified GMRT coverage."""
from pathlib import Path
import json
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, patheffects
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'input_channels_figures/gmrt_candidates'
OUT.mkdir(exist_ok=True)
LAND = ROOT/'input_channels_figures/gem_candidates_20260930/ne_10m_land.zip'
LABELS = ROOT/'SAM_code_extracted/SAM code/earth_dataset_greece_nz_v1'
font_manager.fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font_manager.FontProperties(fname='/mnt/c/Windows/Fonts/msyh.ttc').get_name(),
    'axes.unicode_minus':False,'font.size':11,'svg.fonttype':'path','savefig.facecolor':'#fafbf9'})
land = gpd.read_file('zip://'+str(LAND)).to_crs(4326)
BLUE, RED, ORANGE, INK = '#176c8e','#b14649','#da8a26','#213c4b'
REGIONS = [
    dict(key='california',title='加州外海',subtitle='已有：USGS 海上断层目录',bounds=(-124.1,34.8,-119.7,39.1),
         path=ROOT/'Qfaults_GIS/SHP/ca_offshore.shp',
         areas=[('1',(-123.1,36.35,-121.65,38.05)),('2',(-121.75,35.25,-120.65,36.15))],
         card='① 旧金山—蒙特雷近海\n② San Simeon—Hosgri 一带\n可补充：USGS 详细测区断层 Shapefile\n当前实验角色：加州为外部测试区'),
    dict(key='greece',title='希腊 · 科林斯湾',subtitle='已有：GEM 的 EMME／SHARE 目录线',bounds=(20.5,36.5,24.8,40.0),
         path=LABELS/'greece_aegean_labels.geojson',
         areas=[('3',(21.76,37.93,23.21,38.44))],
         card='③ 科林斯湾 Corinth Rift\n可补充：BGS 海上断层位置与活动速率数据\n橙框为 BGS 元数据范围；BGS 断层线未叠加\n需区分海底表面表现与地下地震反射解释'),
    dict(key='new_zealand',title='新西兰近海',subtitle='已有：GEM 的 Litchfield_NZ_2013 目录线',bounds=(165.8,-47.3,179.5,-33.5),
         path=LABELS/'new_zealand_labels.geojson',
         areas=[('4',(173.25,-42.75,174.75,-41.45))],
         card='④ Kaikōura 近海：可查 NZAFD 海上破裂迹线\n⑤ 其他近海：可查 NZ CFM，具体范围待筛选\nNZAFD 主要在陆地；NZ CFM 为概化断层带\n图上红线来自本地 GEM，并非新下载的标签')]

def text(ax,x,y,s,size=10,color=INK,ha='center'):
    ax.text(x,y,s,ha=ha,va='center',fontsize=size,color=color,zorder=8,
      path_effects=[patheffects.withStroke(linewidth=3,foreground='white',alpha=.9)])

def draw(ax,r):
    bounds=r['bounds']; view=box(*bounds)
    local=land[land.intersects(view)].copy();local.geometry=local.geometry.intersection(view)
    ax.set_facecolor('#e8f3f7')
    local.plot(ax=ax,color='#e6e5dc',edgecolor='#899c9c',linewidth=.55,zorder=1)
    faults=gpd.read_file(r['path']).to_crs(4326)
    localfault=faults[faults.intersects(view)].copy()
    localfault.geometry=localfault.geometry.intersection(view)
    localfault.plot(ax=ax,color=RED,linewidth=.65,alpha=.85,zorder=3)
    for number,(w,s,e,n) in r['areas']:
        ax.add_patch(Rectangle((w,s),e-w,n-s,facecolor=ORANGE,alpha=.10,zorder=4))
        ax.add_patch(Rectangle((w,s),e-w,n-s,facecolor='none',edgecolor=ORANGE,lw=2,ls='--',zorder=5))
        ax.annotate(number,(w,n),xytext=(0,4),textcoords='offset points',ha='center',va='bottom',color='white',fontsize=12,weight='bold',
                    bbox=dict(boxstyle='circle,pad=.25',facecolor=ORANGE,edgecolor='white'),zorder=9)
    if r['key']=='california':
        for x,y,name,dx,dy in [(-122.42,37.77,'旧金山',30,12),(-121.89,36.60,'蒙特雷',28,10),(-121.19,35.64,'San Simeon',33,4)]:
            ax.plot(x,y,'o',ms=3.5,color=INK,zorder=7)
            ax.annotate(name,(x,y),xytext=(dx,dy),textcoords='offset points',fontsize=10,color=INK,
                        arrowprops=dict(arrowstyle='-',lw=.6,color=INK),zorder=8)
        text(ax,-123.35,35.85,'太平洋',13,BLUE)
        text(ax,-120.45,38.35,'加州',13)
    elif r['key']=='greece':
        text(ax,22.20,39.35,'希腊',13)
        text(ax,22.02,37.15,'伯罗奔尼撒',11)
        text(ax,24.15,37.30,'爱琴海',12,BLUE)
        ax.plot(23.73,37.98,'o',ms=3.5,color=INK,zorder=7)
        text(ax,24.08,38.00,'雅典',10)
        ax.annotate('科林斯湾',(22.5,38.2),xytext=(23.1,38.8),fontsize=11,color=INK,
                    arrowprops=dict(arrowstyle='-',color=INK,lw=.8),zorder=8)
    else:
        text(ax,175.65,-38.8,'北岛',12);text(ax,170.25,-44.6,'南岛',12)
        text(ax,168.2,-37.6,'塔斯曼海',12,BLUE)
        text(ax,177.7,-44.7,'太平洋',12,BLUE)
        ax.plot(173.68,-42.40,'o',ms=3.5,color=INK,zorder=7)
        ax.annotate('Kaikōura',(173.68,-42.4),xytext=(175.65,-42.8),fontsize=10,color=INK,
                    arrowprops=dict(arrowstyle='-',color=INK,lw=.8),zorder=8)
        text(ax,175.8,-35.2,'⑤ 其他近海待筛选',10,BLUE)
    ax.set_xlim(bounds[0],bounds[2]);ax.set_ylim(bounds[1],bounds[3])
    ax.set_aspect(1/np.cos(np.radians((bounds[1]+bounds[3])/2)))
    ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{abs(v):g}°'+('E' if v>=0 else 'W')))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{abs(v):g}°'+('N' if v>=0 else 'S')))
    ax.tick_params(labelsize=9)
    ax.grid(lw=.5,alpha=.24,color='#66889b')
    for spine in ax.spines.values():spine.set_color('#a1b4ba')
    ax.set_title(r['title']+'\n'+r['subtitle'],fontsize=12,color=INK,pad=16,linespacing=1.8)
    ax.annotate('N',xy=(.94,.96),xytext=(.94,.85),xycoords='axes fraction',ha='center',fontsize=11,color=INK,
                arrowprops=dict(arrowstyle='-|>',color=INK,lw=1.1))
    return len(localfault)

def footer(fig):
    fig.legend(handles=[Line2D([0],[0],color=RED,lw=1.4,label='本地已有断层目录线（来源见各面板）'),
                        Line2D([0],[0],color=ORANGE,lw=2,ls='--',label='候选定位框／数据集范围，不是 GMRT 覆盖')],
               loc='lower center',bbox_to_anchor=(.5,.078),ncol=2,frameon=False,fontsize=11)
    fig.text(.5,.058,'本图未叠加 GMRT 实测覆盖掩膜，不能据此认定框内均有高分辨率数据；各面板比例尺不同。',ha='center',fontsize=11,color=RED)
    fig.text(.5,.030,'底图：Natural Earth 1:10m 陆地｜坐标：WGS84｜候选框 ①②④ 为示意定位；③ 为 BGS 发布范围｜2026-10-01',ha='center',fontsize=9,color='#617785')

fig=plt.figure(figsize=(19,11.7),facecolor='#fafbf9')
fig.text(.055,.955,'GMRT ＋ 断层标签：候选区域在哪里？',fontsize=24,weight='bold',color=INK)
fig.text(.055,.918,'地理位置与已有标签概览  /  用于选区讨论，尚未完成 GMRT 覆盖交集核查',fontsize=12,color='#617785')
counts={}
for i,r in enumerate(REGIONS):
    x=.052+i*.321
    ax=fig.add_axes([x,.335,.285,.49]);counts[r['key']]=draw(ax,r)
    fig.text(x,.265,r['card'],fontsize=11,color=INK,va='top',linespacing=1.85)
footer(fig)
fig.savefig(OUT/'candidate_regions_overview.png',dpi=190)
fig.savefig(OUT/'candidate_regions_overview.svg')
plt.close(fig)
for r in REGIONS:
    fig=plt.figure(figsize=(10,11.5),facecolor='#fafbf9')
    ax=fig.add_axes([.10,.31,.8,.59]);draw(ax,r)
    fig.text(.10,.24,r['card'],fontsize=11,color=INK,va='top',linespacing=1.65)
    fig.text(.5,.071,'橙框为候选定位范围，非 GMRT 覆盖；红线为本地已有目录。',ha='center',fontsize=10,color=RED)
    fig.text(.5,.04,'Natural Earth 陆地底图 · WGS84 · 2026-10-01',ha='center',fontsize=9,color='#617785')
    fig.savefig(OUT/(r['key']+'_detail.png'),dpi=180)
    plt.close(fig)
sources={
 'gmrt_coverage':'https://www.gmrt.org/services/',
 'california_faults':'https://cmgds.marine.usgs.gov/catalog/pcmsc/DataReleases/ScienceBase/DR_P91RYEZ4/qFaults_offshore_california_2020_metadata.html',
 'corinth_bgs':'https://www.data.gov.uk/dataset/5513fa80-319a-4b38-8469-1a191e7709bb/corinth-rift-greece-fault-location-and-activity-rate-data-nerc-grant-ne-r016550-1',
 'nzafd':'https://www.gns.cri.nz/data-and-resources/new-zealand-active-faults-database/',
 'nzcfm':'https://www.gns.cri.nz/data-and-resources/new-zealand-community-fault-model/'}
(OUT/'provenance.json').write_text(json.dumps(dict(gmrt_coverage_verified=False,land=str(LAND),
    plotted_sources={r['key']:str(r['path']) for r in REGIONS},local_features_intersecting_panels=counts,
    sources=sources,candidate_bounds={r['key']:r['areas'] for r in REGIONS},
    note='Boxes 1,2,4 are illustrative candidate extents; box 3 is published BGS metadata extent. BGS, NZAFD and NZ CFM geometries were not downloaded or plotted.'),ensure_ascii=False,indent=2),encoding='utf-8')
print('Saved',OUT)
print(counts)
