from pathlib import Path
import json
import numpy as np
import h5py
import geopandas as gpd
import netCDF4
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
font=FontProperties(fname='/mnt/c/Windows/Fonts/msyh.ttc')
plt.rcParams['axes.unicode_minus']=False
provenance=json.loads((OUT/'provenance.json').read_text(encoding='utf-8'))
key=provenance['tile']
with h5py.File(ROOT/'SAM_code_extracted/SAM code/earth_dataset_v3/faults.h5','r') as f:
    g=f[key]
    terrain=g['TW'][0,64:320,64:320].astype(float)/255
    latc=float(g.attrs['lat_c']); lonc=float(g.attrs['lon_c'])
with netCDF4.Dataset(ROOT/'GEBCO_California/gebco_2026_n50.0_s19.0_w-129.0_e-110.0.nc') as f:
    resolution=abs(float(f.variables['lat'][1])-float(f.variables['lat'][0]))
extent=[lonc-128*resolution,lonc+128*resolution,latc-128*resolution,latc+128*resolution]
faults=gpd.read_file(ROOT/'Qfaults_GIS/SHP/ca_offshore.shp').to_crs(4326)
from shapely.geometry import box
visible=faults[faults.intersects(box(*[extent[0],extent[2],extent[1],extent[3]]))].copy()
visible.geometry=visible.geometry.intersection(box(extent[0],extent[2],extent[1],extent[3]))
def background(ax,overlay=False):
    ax.imshow(terrain,cmap='gray',vmin=-.15,vmax=1,origin='lower',extent=extent,interpolation='nearest')
    if overlay:
        visible.plot(ax=ax,color='#e63d32',linewidth=1.45,zorder=3)
    ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:])
    ax.set_aspect(1/np.cos(np.deg2rad(latc)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{abs(x):.1f}°W'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:.1f}°N'))
    ax.tick_params(labelsize=11,color='#94a3b8')
    for s in ax.spines.values():s.set_color('#94a3b8')
fig,axs=plt.subplots(1,2,figsize=(14,8),facecolor='white')
background(axs[0]);background(axs[1],True)
for ax,title in zip(axs,['地形背景','USGS 断层标注叠加']):
    ax.set_title(title,fontproperties=font,fontsize=18,pad=16,loc='left')
axs[1].legend(handles=[Line2D([0],[0],color='#e63d32',lw=2,label='USGS 目录断层线')],prop=font,loc='lower left',framealpha=.95)
fig.suptitle('USGS 断层标注',fontproperties=font,fontsize=28,x=.08,ha='left',y=.96)
fig.text(.08,.895,'同一地理区域的地形输入与断层目录；红线表示已有断层记录。',fontproperties=font,fontsize=14,color='#475569')
fig.text(.08,.075,'地形：GEBCO 衍生晕渲（315°）  |  标注：项目所用 USGS ca_offshore.shp',fontproperties=font,fontsize=11,color='#475569')
fig.text(.08,.038,'未画出断层线的区域不等于已确认无断层；图中展示目录矢量线，训练标签由其另行生成。',fontproperties=font,fontsize=11,color='#475569')
fig.subplots_adjust(left=.08,right=.96,top=.80,bottom=.15,wspace=.22)
fig.savefig(OUT/'usgs_fault_annotation.png',dpi=260)
fig.savefig(OUT/'usgs_fault_annotation.svg')
plt.close(fig)
fig,ax=plt.subplots(figsize=(7,8))
background(ax,True)
ax.set_title('USGS 断层标注',fontproperties=font,fontsize=22,pad=18)
ax.legend(handles=[Line2D([0],[0],color='#e63d32',lw=2,label='USGS 目录断层线')],prop=font,loc='lower left')
fig.text(.5,.02,'GEBCO 地形背景 · USGS ca_offshore 断层目录',fontproperties=font,fontsize=10,ha='center')
fig.tight_layout(rect=[0,.05,1,1])
fig.savefig(OUT/'usgs_fault_overlay.png',dpi=300)
plt.close(fig)
(OUT/'usgs_annotation_provenance.json').write_text(json.dumps({'tile':key,'extent_west_east_south_north':extent,'intersecting_catalog_features':len(visible),'source_shapefile':'Qfaults_GIS/SHP/ca_offshore.shp','source_crs':str(faults.crs),'background':'TW[0], 315-degree hillshade; center crop 256x256, origin lower','note':'Original catalog vectors, not a buffered binary training mask'},indent=2),encoding='utf-8')
print(f'Saved annotation figures; {len(visible)} catalog features intersect sample {key}.')
