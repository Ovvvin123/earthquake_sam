"""Map the saved USGS-trained UniMatch d0=0.01 predictions in geographic space."""
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
from pyproj import Geod
from shapely.geometry import box

from count_fault_segments import make_mosaic


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    code = Path(__file__).resolve().parent
    root = code.parents[1]
    run = code/'train_v5_unimatch_d001'
    output = run/'geographic_maps'
    output.mkdir(exist_ok=True)
    export = run/'predictions_seen_unseen'
    manifest = json.loads((export/'manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['full_export']
    assert manifest['d0'] == .01 and manifest['threshold'] == .5 and manifest['epoch'] == 174
    config = json.loads((run/'ema_config.json').read_text())
    assert config['unimatch'] and config['d0'] == .01
    model = run/'best_student/model.pth'
    assert digest(model) == manifest['model_sha256']
    data = code/'earth_dataset_v3'
    dataset_meta = json.loads((data/'manifest.json').read_text())
    resolution = float(dataset_meta['provenance']['res_deg'])
    bounds = tuple(dataset_meta['provenance']['bbox'])
    assert np.isclose(resolution, 1/240)
    assert manifest['counts'] == {'train_seen': 161, 'test_unseen': 38}
    with h5py.File(data/'faults.h5', 'r') as h5:
        offsets = []
        for tile in manifest['images']:
            row, col = map(int, tile['key'].split('_')[1:])
            h, w = tile['shape']
            group = h5[tile['key']]
            offsets.append((float(group.attrs['lon_c'])-(col+(w-1)/2)*resolution,
                            float(group.attrs['lat_c'])-(row+(h-1)/2)*resolution))
        offsets = np.array(offsets)
        assert np.max(np.ptp(offsets, axis=0)) < 1e-8
        lon0, lat0 = np.median(offsets, axis=0)
        # Rebuild once from every exported tile to verify saved mosaic and labels.
        p, valid, truth, origin, audit = make_mosaic(export, manifest['images'], .5, .01, h5)
    mosaic = run/'segment_counts/all_exported_mosaic.npz'
    with np.load(mosaic) as z:
        assert np.array_equal(p, z['probability'])
        assert np.array_equal(valid, z['valid']) and np.array_equal(truth, z['truth'])
        assert origin == (int(z['row_origin']), int(z['col_origin']))
    h, w = p.shape
    lon = lon0+(origin[1]+np.arange(w))*resolution
    lat = lat0+(origin[0]+np.arange(h))*resolution
    extent = [lon[0]-resolution/2, lon[-1]+resolution/2,
              lat[0]-resolution/2, lat[-1]+resolution/2]
    prediction = (p >= .5) & valid
    print('Mosaic verified:', p.shape, 'valid:', int(valid.sum()),
          'predicted:', int(prediction.sum()), 'extent:', extent, flush=True)

    land_path = root/'input_channels_figures/gem_candidates_20260930/ne_10m_land.zip'
    fault_path = root/'Qfaults_GIS/SHP/ca_offshore.shp'
    frame = box(bounds[0]-.1, bounds[1]-.1, bounds[2]+.1, bounds[3]+.1)
    land = gpd.read_file('zip://'+str(land_path)).to_crs(4326)
    land = land[land.intersects(frame)].copy()
    land.geometry = land.geometry.intersection(frame)
    faults = gpd.read_file(fault_path).to_crs(4326)
    faults = faults[faults.intersects(frame)].copy()
    faults.geometry = faults.geometry.intersection(frame)
    font_manager.fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname='/mnt/c/Windows/Fonts/msyh.ttc').get_name(),
                         'axes.unicode_minus': False, 'font.size': 11, 'svg.fonttype': 'path'})
    geod = Geod(ellps='WGS84')
    cx = np.r_[lon[0]-resolution, lon, lon[-1]+resolution]
    cy = np.r_[lat[0]-resolution, lat, lat[-1]+resolution]

    def map_panel(ax, mode):
        ax.set_facecolor('#e9f3f7')
        land.plot(ax=ax, facecolor='#e9e3d6', edgecolor='#586b70', linewidth=.65, zorder=1)
        coverage = np.ma.masked_where(~valid, np.ones_like(p))
        ax.imshow(coverage, extent=extent, origin='lower', interpolation='nearest',
                  cmap=ListedColormap(['#74879b']), alpha=.085, zorder=2)
        ax.contour(cx, cy, np.pad(valid.astype(np.uint8), 1), levels=[.5], colors=['#758694'],
                   linewidths=.65, linestyles='--', zorder=3)
        if mode in ('predictions', 'overlay'):
            ax.imshow(np.ma.masked_where(~prediction, p), extent=extent, origin='lower',
                      interpolation='nearest', cmap=ListedColormap(['#d63c65']), alpha=.86, zorder=4)
        if mode in ('labels', 'overlay'):
            faults.plot(ax=ax, color='#007f8b', linewidth=.7, alpha=.95, zorder=5)
        ax.set_xlim(bounds[0], bounds[2]); ax.set_ylim(bounds[1], bounds[3])
        ax.set_aspect(1/np.cos(np.deg2rad((bounds[1]+bounds[3])/2)))
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{abs(v):g}°W'))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}°N'))
        ax.grid(color='#637f8c', alpha=.18, lw=.5, zorder=0)
        ax.set_xlabel('经度'); ax.set_ylabel('纬度')
        ax.text(-119.1,38.7,'加利福尼亚',color='#66685e',fontsize=13,rotation=-28,ha='center')
        ax.text(-123.5,34.6,'太 平 洋',color='#7692a0',fontsize=15,rotation=-25,ha='center')
        ax.annotate('N',xy=(.94,.94),xytext=(.94,.86),xycoords='axes fraction',
                    ha='center',va='center',fontsize=12,
                    arrowprops={'arrowstyle':'-|>','color':'#334c59','lw':1.2})
        x0,y0=-124.5,32.05
        x1,y1,_=geod.fwd(x0,y0,90,100000)
        ax.plot([x0,x1],[y0,y0],color='#334c59',lw=2.2,zorder=6)
        ax.plot([x0,x0],[y0-.035,y0+.035],color='#334c59',lw=1)
        ax.plot([x1,x1],[y0-.035,y0+.035],color='#334c59',lw=1)
        ax.text((x0+x1)/2,y0+.08,'100 km',ha='center',fontsize=9,color='#334c59')
        handles=[]
        if mode != 'labels': handles.append(Patch(facecolor='#d63c65',label='模型预测区域（概率 ≥ 0.5）'))
        if mode != 'predictions': handles.append(Line2D([0],[0],color='#007f8b',lw=1.5,label='USGS 原始断层线'))
        handles += [Line2D([0],[0],color='#758694',ls='--',lw=1,label='已预测区域边界'),
                    Patch(facecolor='#e9e3d6',edgecolor='#586b70',label='陆地／海岸线')]
        ax.legend(handles=handles,loc='lower right',fontsize=9,framealpha=.97)
        for spine in ax.spines.values(): spine.set_color('#8599a2')

    subtitle='USGS 训练标签 · 最佳 Student 第174轮 · d0=0.01° · 预测阈值0.5'
    footer=('199个图块：训练161个＋未见测试38个；重叠位置平均概率后再取阈值。\n'
            '虚线外未做本次预测；空白不代表无断层。预测区域未做去碎点或人为连线。\n'
            '底图：Natural Earth 1:10m land；断层：本地 USGS ca_offshore；本图混合训练区与测试区，不用于整体精度评分。')
    outputs=[]
    for mode,name,title in [('predictions','prediction_full_map','UniMatch V2 适配版｜加州断层预测全幅拼接'),
                             ('overlay','prediction_usgs_overlay','UniMatch V2 适配版｜预测与 USGS 原始标注')]:
        fig,ax=plt.subplots(figsize=(10.5,12.5))
        map_panel(ax,mode)
        fig.suptitle(title,fontsize=18,y=.965)
        fig.text(.5,.935,subtitle,ha='center',fontsize=10,color='#526574')
        fig.text(.09,.026,footer,fontsize=9,color='#526574',linespacing=1.65)
        fig.subplots_adjust(left=.10,right=.97,top=.90,bottom=.14)
        for ext in ['png','svg']:
            path=output/(name+'.'+ext);fig.savefig(path,dpi=260,facecolor='white');outputs.append(path.name)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(17,11.5))
    for ax,mode,title in zip(axes,['labels','predictions'],['USGS 原始断层目录','同范围模型预测（概率 ≥ 0.5）']):
        map_panel(ax,mode);ax.set_title(title,fontsize=14,pad=12)
    fig.suptitle('加州全幅对照｜原始目录与 UniMatch V2 d0=0.01 预测',fontsize=19,y=.965)
    fig.text(.5,.935,subtitle,ha='center',fontsize=10,color='#526574')
    fig.text(.075,.025,footer,fontsize=10,color='#526574',linespacing=1.5)
    fig.subplots_adjust(left=.07,right=.97,top=.89,bottom=.15,wspace=.15)
    fig.savefig(output/'usgs_vs_prediction_full.png',dpi=230,facecolor='white')
    outputs.append('usgs_vs_prediction_full.png');plt.close(fig)
    report=dict(model=str(model),model_sha256=manifest['model_sha256'],epoch=174,
                training_label_source=str(fault_path),label_channel='k0 (any fault)',d0_degrees=.01,
                threshold=.5,smoothing=False,small_component_removal=False,
                map_bounds_w_s_e_n=bounds,raster_extent_w_e_s_n=extent,
                grid_resolution_degrees=resolution,grid_origin_pixel_center_lon_lat=[float(lon0),float(lat0)],
                max_tile_georeference_spread_degrees=float(np.max(np.ptp(offsets,axis=0))),
                counts=manifest['counts'],unique_valid_pixels=int(valid.sum()),
                predicted_pixels=int(prediction.sum()),original_label_pixels=int(truth.sum()),
                coverage_fraction_of_study_bbox=float(valid.sum()*resolution**2/((bounds[2]-bounds[0])*(bounds[3]-bounds[1]))),
                mosaic_sha256=digest(mosaic),manifest_sha256=digest(export/'manifest.json'),
                land_path=str(land_path),land_sha256=digest(land_path),
                fault_shapefile_sha256=digest(fault_path),script_sha256=digest(Path(__file__)),
                checks=['199 tile georeferences agree', 'all predictions checked against original HDF5 labels',
                        'rebuilt all-tile mosaic equals saved probability, valid and truth arrays exactly',
                        'model checksum matches exported prediction manifest'],outputs=outputs)
    (output/'map_provenance.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__': main()
