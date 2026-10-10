from pathlib import Path
import json, zipfile, urllib.request
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from shapely.geometry import box

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
plt.rcParams['font.family'] = FontProperties(fname='/mnt/c/Windows/Fonts/msyh.ttc').get_name()
from matplotlib import font_manager
font_manager.fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 11, 'savefig.facecolor': 'white'})
archive = OUT / 'ne_110m_land.zip'
if not archive.exists():
    urllib.request.urlretrieve('https://naturalearth.s3.amazonaws.com/110m_physical/ne_110m_land.zip', archive)
land = gpd.read_file('zip://' + str(archive)).to_crs(4326)
gem_path = ROOT / 'GEBCO_Global/GEM_label/gem-global-active-faults-master/shapefile/gem_active_faults_harmonized.shp'
gem = gpd.read_file(gem_path).to_crs(4326)
us = gpd.read_file(ROOT / 'Qfaults_GIS/SHP/Qfaults_US_Database.shp').to_crs(4326)
ca = gpd.read_file(ROOT / 'Qfaults_GIS/SHP/ca_offshore.shp').to_crs(4326)

def background(ax, bounds):
    ax.set_facecolor('#eaf3f8')
    land.plot(ax=ax, color='#f1ede4', edgecolor='#9caeb5', linewidth=.35)
    ax.set_xlim(bounds[0], bounds[2]); ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect(1 / np.cos(np.radians((bounds[1]+bounds[3])/2)))
    ax.grid(color='#9babb3', alpha=.3, linewidth=.5)
    ax.set_xlabel('经度（°）'); ax.set_ylabel('纬度（°）')
    for spine in ax.spines.values(): spine.set_color('#b9c7ce')

def save(fig, name):
    fig.savefig(OUT / (name+'.png'), dpi=190)
    fig.savefig(OUT / (name+'.svg'))
    plt.close(fig)

fig, ax = plt.subplots(figsize=(15,7.5))
background(ax, (-180,-65,180,85)); ax.set_aspect('auto')
# Split date-line crossing segments so no spurious horizontal lines appear.
from matplotlib.collections import LineCollection
segments=[]
for geom in gem.geometry:
    if geom is None: continue
    for line in (geom.geoms if geom.geom_type == 'MultiLineString' else [geom]):
        xy=np.asarray(line.coords)[:,:2]
        for part in np.split(xy, np.where(np.abs(np.diff(xy[:,0]))>180)[0]+1):
            if len(part)>1: segments.append(part)
ax.add_collection(LineCollection(segments, colors='#d63c43', linewidths=.6))
fig.suptitle('GEM 全球活动断层目录：本地下载数据的迹线分布', fontsize=19, x=.08, ha='left', y=.97)
ax.set_title(f'harmonized 版本 | {len(gem):,} 条 GIS 要素（不等于独立断层数量）', loc='left', fontsize=11, pad=12)
fig.text(.08,.055,'红线＝目录中的断层迹线；空白不代表没有断层，也不能据此量化漏标率。显示纬度：65°S–85°N。',fontsize=11)
fig.text(.08,.022,'来源：GEM GAF-DB，本地 harmonized shapefile；Styron & Pagani (2020)，CC BY-SA 4.0。底图：Natural Earth。',fontsize=9,color='#536773')
fig.subplots_adjust(left=.08,right=.975,top=.86,bottom=.15)
save(fig,'01_GEM_global_distribution')

fig=plt.figure(figsize=(15,8))
gs=fig.add_gridspec(2,2,width_ratios=[2,1],hspace=.38,wspace=.2)
axes=[fig.add_subplot(gs[:,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,1])]
for ax,bounds,title in zip(axes,[(-126,24,-66,50),(-180,50,-128,73),(-161,18,-154,23)],['美国本土','阿拉斯加','夏威夷']):
    background(ax,bounds)
    subset=us[us.intersects(box(*bounds))]
    if len(subset): subset.plot(ax=ax,color='#2565b5',linewidth=.6)
    ax.set_title(title,loc='left',fontsize=13)
ca.plot(ax=axes[0],color='#df7130',linewidth=.8)
axes[0].legend(handles=[Line2D([0],[0],color='#2565b5',label='Qfaults 全美目录'),Line2D([0],[0],color='#df7130',label='ca_offshore 加州外海目录')],loc='lower left')
fig.suptitle('USGS 本地标签库：全美目录与加州外海目录',fontsize=19,x=.07,ha='left',y=.97)
fig.text(.07,.075,f'本地 GIS 要素数：Qfaults {len(us):,}；ca_offshore {len(ca):,}。本图表示已收录迹线，非完整调查覆盖或精度地图。',fontsize=11)
fig.text(.07,.035,'来源：本地 Qfaults_US_Database.shp、ca_offshore.shp；USGS 加州外海数据 DOI: 10.5066/P91RYEZ4。底图：Natural Earth。',fontsize=9,color='#536773')
fig.subplots_adjust(left=.07,right=.975,top=.87,bottom=.17)
save(fig,'02_USGS_catalog_distribution')

bounds=(-125,31.5,-116.5,41)
gem_ca=gem[gem.intersects(box(*bounds))].copy()
gem_ca.geometry=gem_ca.geometry.intersection(box(*bounds))
fig,axes=plt.subplots(1,2,figsize=(13,9))
for ax,data,color,title in zip(axes,[ca,gem_ca],['#d66d25','#d63c43'],['USGS：ca_offshore（当前标签来源）','GEM：harmonized（同一经纬度窗口）']):
    background(ax,bounds)
    if len(data): data.plot(ax=ax,color=color,linewidth=.85)
    ax.set_title(title,fontsize=13,pad=12)
fig.suptitle('加州同范围比较：两套目录收录的断层迹线',fontsize=19,x=.07,ha='left',y=.965)
fig.text(.07,.065,'窗口：125°W–116.5°W，31.5°N–41°N；包含陆地和海域，并非模型实际测试切片的精确足迹。',fontsize=10)
fig.text(.07,.035,'只比较目录分布，不判定谁对谁错；GEM 与 USGS 的来源可能重叠，空白不等于可靠负样本。底图：Natural Earth。',fontsize=10)
fig.subplots_adjust(left=.07,right=.975,top=.89,bottom=.15,wspace=.17)
save(fig,'03_California_USGS_vs_GEM')

summary={'gem_features':len(gem),'usgs_national_features':len(us),'usgs_ca_offshore_features':len(ca),'gem_features_intersecting_ca_window':len(gem_ca),'gem_bounds':gem.total_bounds.tolist(),'usgs_bounds':us.total_bounds.tolist(),'ca_bounds':ca.total_bounds.tolist(),'gem_source':str(gem_path),'note':'Feature counts are not unique fault counts; no quantitative completeness or missed-label rate assessed.'}
(OUT/'map_provenance.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False))
