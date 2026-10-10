"""Rebuild the requested run's saved prediction mosaic without training or new inference."""
import hashlib
import importlib.util
import json
from pathlib import Path
import h5py
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from grid_core import BASE


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def main():
    code=BASE.parent; root=BASE.parents[2]
    run=code/'train_v5_unimatch_d001'
    export=run/'predictions_seen_unseen'
    out=BASE/'reports/unimatch_d001_mosaic'; out.mkdir(parents=True,exist_ok=True)
    man=json.loads((export/'manifest.json').read_text())
    assert man['status']=='complete' and man['full_export']
    model=run/'best_student/model.pth'
    assert sha(model)==man['model_sha256']
    spec=importlib.util.spec_from_file_location('original_mosaic',code/'count_fault_segments.py')
    original=importlib.util.module_from_spec(spec); spec.loader.exec_module(original)
    meta=json.loads((code/'earth_dataset_v3/manifest.json').read_text())
    res=meta['provenance']['res_deg']; bounds=meta['provenance']['bbox']
    with h5py.File(code/'earth_dataset_v3/faults.h5','r') as f:
        p,v,t,origin,audit=original.make_mosaic(export,man['images'],man['threshold'],man['d0'],f)
        origins=[]
        for row in man['images']:
            i,j=map(int,row['key'].split('_')[1:]); h,w=row['shape']
            g=f[row['key']]
            origins.append((float(g.attrs['lon_c'])-(j+(w-1)/2)*res,
                            float(g.attrs['lat_c'])-(i+(h-1)/2)*res))
    assert np.max(np.ptp(origins,axis=0))<1e-8
    lon0,lat0=np.median(origins,axis=0)
    with np.load(run/'segment_counts/all_exported_mosaic.npz') as old:
        assert np.array_equal(p,old['probability']) and np.array_equal(v,old['valid']) and np.array_equal(t,old['truth'])
    lon=lon0+(origin[1]+np.arange(p.shape[1]))*res
    lat=lat0+(origin[0]+np.arange(p.shape[0]))*res
    extent=[lon[0]-res/2,lon[-1]+res/2,lat[0]-res/2,lat[-1]+res/2]
    pred=(p>=man['threshold'])&v
    np.savez_compressed(out/'stitched_predictions.npz',probability=p,valid=v,truth=t,prediction=pred,
                        longitude=lon,latitude=lat,threshold=man['threshold'],d0=man['d0'])
    land=gpd.read_file('zip://'+str(root/'input_channels_figures/gem_candidates_20260930/ne_10m_land.zip')).to_crs(4326)
    land=land.cx[bounds[0]:bounds[2],bounds[1]:bounds[3]]
    faults=gpd.read_file(root/'Qfaults_GIS/SHP/ca_offshore.shp').to_crs(4326)
    faults=faults.cx[bounds[0]:bounds[2],bounds[1]:bounds[3]]
    font=Path('/mnt/c/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    def panel(ax,kind):
        ax.set_facecolor('#edf5f8')
        land.plot(ax=ax,facecolor='#ede8df',edgecolor='#8b969c',linewidth=.5,zorder=0)
        if kind=='probability':
            artist=ax.imshow(np.ma.masked_where(~v,p),extent=extent,origin='lower',vmin=0,vmax=1,
                cmap='magma',interpolation='nearest',zorder=1)
        else:
            ax.imshow(np.ma.masked_where(~v,np.ones(v.shape)),extent=extent,origin='lower',
                cmap=ListedColormap(['#8595a4']),alpha=.08,zorder=1)
            artist=ax.imshow(np.ma.masked_where(~pred,p),extent=extent,origin='lower',
                cmap=ListedColormap(['#d63b6a']),interpolation='nearest',zorder=2)
        if kind=='overlay': faults.plot(ax=ax,color='#16798a',linewidth=.75,zorder=3)
        ax.contour(np.r_[lon[0]-res,lon,lon[-1]+res],np.r_[lat[0]-res,lat,lat[-1]+res],
            np.pad(v.astype(np.uint8),1),levels=[.5],colors='#6e8395',linestyles='--',linewidths=.7,zorder=4)
        ax.set_xlim(bounds[0],bounds[2]); ax.set_ylim(bounds[1],bounds[3])
        ax.set_aspect(1/np.cos(np.deg2rad(36.25)))
        ax.set_xlabel('经度'); ax.set_ylabel('纬度'); ax.grid(alpha=.13)
        return artist
    handles=[Patch(facecolor='#d63b6a',label='预测断层区域：概率 ≥ 0.5'),
             Line2D([0],[0],color='#6e8395',linestyle='--',label='已预测覆盖边界')]
    note='161块训练区 + 38块测试区；切片重叠处先平均概率，再取阈值。虚线外未预测，不能解释为无断层。'
    fig,ax=plt.subplots(figsize=(10,12))
    panel(ax,'prediction'); ax.legend(handles=handles,loc='lower right',fontsize=9)
    ax.set_title(f'UniMatch d0=0.01°：完整区域预测拼图\n最佳学生模型，第{man["epoch"]}轮',fontsize=16,pad=15)
    fig.text(.5,.025,note,ha='center',fontsize=9,color='#526474')
    fig.tight_layout(rect=(0,.05,1,1)); fig.savefig(out/'prediction_full_map.png',dpi=240); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(19,8.5))
    prob=panel(axes[0],'probability'); axes[0].set_title('断层概率（0—1）')
    fig.colorbar(prob,ax=axes[0],orientation='horizontal',fraction=.035,pad=.09,label='预测概率')
    panel(axes[1],'prediction'); axes[1].set_title('预测区域（阈值0.5）')
    panel(axes[2],'overlay'); axes[2].set_title('预测与USGS断层目录叠加')
    axes[2].legend(handles=handles+[Line2D([0],[0],color='#16798a',label='USGS目录断层线')],fontsize=8,loc='lower right')
    fig.suptitle(f'train_v5_unimatch_d001 · 最佳学生第{man["epoch"]}轮 · 地理拼接结果',fontsize=16)
    fig.text(.5,.018,note+' 此图混合训练与测试区域，仅用于观察。',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.94)); fig.savefig(out/'probability_prediction_comparison.png',dpi=180); plt.close(fig)
    report=dict(model=str(model),epoch=man['epoch'],model_sha256=man['model_sha256'],
        source_manifest_sha256=sha(export/'manifest.json'),d0=man['d0'],threshold=man['threshold'],
        counts=man['counts'],shape=list(p.shape),extent_w_e_s_n=extent,
        unique_valid_pixels=int(v.sum()),predicted_pixels=int(pred.sum()),
        missing_validation_tiles=17,coverage_fraction_of_bbox=float(v.sum()/((bounds[2]-bounds[0])*(bounds[3]-bounds[1])/res**2)),
        method='Rebuild 199 saved tile predictions. Equal valid-tile probability averaging. No smoothing, dilation, filtering, new inference or training.',
        matches_previous_mosaic_exactly=True,audit=audit)
    (out/'provenance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:x for k,x in report.items() if k!='audit'},indent=2))

if __name__=='__main__': main()
