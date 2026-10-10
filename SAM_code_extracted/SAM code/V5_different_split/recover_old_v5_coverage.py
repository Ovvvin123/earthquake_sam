"""Recover old V5 footprint polygons from manifest geometry and the actual key lists."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import netCDF4
import geopandas as gpd
from shapely.geometry import box, mapping
from shapely.ops import unary_union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from grid_core import BASE


def main():
    old = BASE.parent/'earth_dataset_v3'
    root = BASE.parents[2]
    out = BASE/'reports/old_v5_coverage'
    out.mkdir(parents=True, exist_ok=True)
    man = json.loads((old/'manifest.json').read_text())
    info = json.loads((old/'blocked_split_info.json').read_text())
    lists = {s:(old/f'key_{s}.lst').read_text().split() for s in ('train','val','test')}
    tiles = {t['key']:t for t in man['tiles']}
    selected = set(sum(lists.values(), []))
    assert len(selected) == sum(map(len, lists.values()))
    assert set(tiles)-selected == set(info['dropped'])
    assert {s:len(k) for s,k in lists.items()} == info['counts']
    T = man['provenance']['tile']; res = man['provenance']['res_deg']
    xmin, ymin, xmax, ymax = man['provenance']['bbox']
    height, width = round((ymax-ymin)/res), round((xmax-xmin)/res)
    masks = {s:np.zeros((height,width),bool) for s in ('all','used','train','val','test')}
    polygons, rows, features = {}, [], []
    for key, t in tiles.items():
        _, i, j = key.split('_'); i,j = int(i),int(j)
        # Bounds are pixel-footprint edges, not first/last pixel centres.
        bounds = [t['lon_c']-T*res/2, t['lat_c']-T*res/2,
                  t['lon_c']+T*res/2, t['lat_c']+T*res/2]
        assert abs(bounds[0]-(xmin+j*res)) < 1e-7
        assert abs(bounds[1]-(ymin+i*res)) < 1e-7
        assert i+T <= height and j+T <= width
        polygons[key] = box(*bounds)
        split = next((s for s,ks in lists.items() if key in ks), 'excluded_buffer')
        masks['all'][i:i+T,j:j+T] = True
        if key in selected:
            masks['used'][i:i+T,j:j+T] = True
            masks[split][i:i+T,j:j+T] = True
        row = dict(key=key, current_v5_split=split, old_manifest_split=t['split'],
                   west=bounds[0],south=bounds[1],east=bounds[2],north=bounds[3],
                   lat_c=t['lat_c'],lon_c=t['lon_c'],is_positive=t['is_positive'])
        rows.append(row)
        features.append(dict(type='Feature',properties=row,geometry=mapping(polygons[key])))
    with (out/'old_v5_tile_bounds.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (out/'old_v5_tiles.geojson').write_text(json.dumps(dict(type='FeatureCollection',features=features)))
    (out/'old_v5_used_tiles.geojson').write_text(json.dumps(dict(type='FeatureCollection',features=[f for f in features if f['properties']['key'] in selected])))
    key_groups = dict(all=list(tiles),used=sorted(selected),**lists)
    unions = {s:unary_union([polygons[k] for k in ks]) for s,ks in key_groups.items()}
    (out/'old_v5_union.geojson').write_text(json.dumps(dict(type='FeatureCollection',features=[
        dict(type='Feature',properties=dict(group=s,tile_count=len(key_groups[s])),geometry=mapping(p))
        for s,p in unions.items()])))
    (out/'old_v5_used_coverage.geojson').write_text(json.dumps(dict(type='FeatureCollection',features=[dict(type='Feature',properties=dict(group='used',tile_count=len(selected)),geometry=mapping(unions['used']))])))
    report=dict(source_manifest=str(old/'manifest.json'),
        source_lists={s:str(old/f'key_{s}.lst') for s in lists},
        source_sha256={n:hashlib.sha256((old/n).read_bytes()).hexdigest() for n in
            ['manifest.json','key_train.lst','key_val.lst','key_test.lst','blocked_split_info.json']},
        extraction_bbox=[xmin,ymin,xmax,ymax],stored_tiles=len(tiles),used_tiles=len(selected),
        excluded_tiles=len(info['dropped']),counts={s:len(k) for s,k in lists.items()},
        footprint_bboxes={s:list(p.bounds) for s,p in unions.items()},
        coverage_pixels={s:int(m.sum()) for s,m in masks.items()},
        bbox_coverage_percent={s:100*float(m.mean()) for s,m in masks.items()},
        shared_pixels_train_val=int((masks['train']&masks['val']).sum()),
        shared_pixels_train_test=int((masks['train']&masks['test']).sum()),
        shared_pixels_val_test=int((masks['val']&masks['test']).sum()),
        note='Used footprint is reconstructed from current V5 key lists. Manifest split field is historical. Bounds alone do not represent the irregular union.')
    assert report['shared_pixels_train_test']==report['shared_pixels_val_test']==0
    (out/'coverage_summary.json').write_text(json.dumps(report,indent=2))
    # Read original terrain only to draw the 0 m contour and distinguish sea/land.
    with netCDF4.Dataset(root/'GEBCO_Global/GEBCO_2026_global/GEBCO_2026.nc') as f:
        la=np.asarray(f['lat'][:]); lo=np.asarray(f['lon'][:])
        iy=np.flatnonzero((la>=ymin)&(la<=ymax)); ix=np.flatnonzero((lo>=xmin)&(lo<=xmax))
        z=np.ma.asarray(f['elevation'][iy[0]:iy[-1]+1:4,ix[0]:ix[-1]+1:4],dtype=float).filled(np.nan)
        la=la[iy][::4]; lo=lo[ix][::4]
    faults=gpd.read_file(root/'Qfaults_GIS/SHP/ca_offshore.shp').to_crs(4326).cx[xmin:xmax,ymin:ymax]
    fig,axes=plt.subplots(1,3,figsize=(18,8),sharex=True,sharey=True)
    extent=[xmin,xmax,ymin,ymax]
    for ax in axes:
        ax.imshow((z>=0).astype(float),origin='lower',extent=extent,cmap=ListedColormap(['#e8f3f7','#ede9df']),vmin=0,vmax=1)
        ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax); ax.set_aspect('equal'); ax.set_xlabel('Longitude')
    def overlay(ax,mask,color,alpha=.5):
        ax.imshow(np.ma.masked_where(~mask,np.ones(mask.shape)),origin='lower',extent=extent,
                  cmap=ListedColormap([color]),vmin=0,vmax=1,alpha=alpha,interpolation='nearest')
    overlay(axes[0],masks['all'],'#9472bb')
    axes[0].set_title('A. Original stored tiles: 339\nUnion of footprints before V5 re-split')
    overlay(axes[1],masks['train'],'#2878ba')
    overlay(axes[1],masks['val'],'#f6b83f')
    overlay(axes[1],masks['train']&masks['val'],'#319c91',.8)
    overlay(axes[1],masks['test'],'#e66a65')
    axes[1].set_title('B. Actual V5 tiles: 216\nTrain 161 / val 17 / test 38')
    new=json.loads((BASE/'datasets/ca_stride384/manifest.json').read_text())
    for t in new['tiles']:
        x0,y0,x1,y1=t['bbox']
        axes[2].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor='#8397af',edgecolor='white',alpha=.3,lw=1))
    overlay(axes[2],masks['used'],'#8061aa',.6)
    axes[2].set_title('C. New grid: 36 non-overlapping tiles\nPurple = actual old V5 footprint')
    for ax in axes:
        ax.contour(lo,la,z,levels=[0],colors=['#655b4e'],linewidths=.8)
        faults.plot(ax=ax,color='#17252c',linewidth=.65)
    axes[0].set_ylabel('Latitude')
    axes[0].legend(handles=[Patch(facecolor='#9472bb',alpha=.5,label='Stored footprint'),
        Line2D([0],[0],color='#17252c',label='USGS fault traces'),
        Line2D([0],[0],color='#655b4e',label='GEBCO 0 m contour')],loc='lower left',fontsize=8)
    axes[1].legend(handles=[Patch(facecolor=c,label=s) for s,c in
        [('Train','#2878ba'),('Validation','#f6b83f'),('Train + validation overlap','#319c91'),('Test','#e66a65')]],loc='lower left',fontsize=8)
    axes[2].legend(handles=[Patch(facecolor='#8061aa',label='Old V5 used footprint'),
        Patch(facecolor='#8397af',alpha=.3,label='New full-region grid')],loc='lower left',fontsize=8)
    fig.suptitle('Recovered old V5 coverage: tile footprints, not tile centres',fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig(out/'old_v5_coverage_comparison.png',dpi=180); plt.close(fig)
    (out/'说明.txt').write_text('旧V5覆盖恢复说明\n\nA图是旧H5全部339块的覆盖并集。\nB图是旧V5实际名单中216块的覆盖，123块缓冲排除样本不计入。\nC图将实际旧V5覆盖叠到新的36块非重叠网格上。\n\n旧V5名单是归属依据，manifest的split是更早的划分，不能用于确定当前V5归属。\nGeoJSON中的used要素就是实际旧V5覆盖范围，可作为后续沿旧范围裁剪的参考。\n覆盖并集不是严格的海岸带：旧384像素切片横跨约1.6度，会延伸到远海与陆地。\n原目录未修改；本次只恢复和导出范围，没有改变新数据集或训练名单。\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if not k.startswith('source')},indent=2))
    print(out/'old_v5_coverage_comparison.png')

if __name__=='__main__': main()
