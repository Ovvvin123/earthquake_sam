from pathlib import Path
import json
import zipfile
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
DATA = ROOT / 'SAM_code_extracted/SAM code/earth_dataset_v3/faults.h5'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12})
with h5py.File(DATA, 'r') as f:
    key = 't_128_1408' if 't_128_1408' in f else max(f, key=lambda k: f[k].attrs.get('n_fault_px', 0))
    g = f[key]
    ctx = g['CTX'][:].astype('float32')
    depth = g['image'][:].astype('float32')
    tw = g['TW'][:].astype('float32')
    attrs = {k: v.item() if hasattr(v, 'item') else v for k, v in g.attrs.items()}
v0 = np.isfinite(ctx) & (ctx > -2147483647)
v1 = np.isfinite(depth) & (depth > -2147483647)
ctx = (np.where(v0, ctx, 0)/255 - .5)/.5
depth = np.where(v1, depth, np.nan)
depth = np.nan_to_num((depth-np.nanmean(depth))/(np.nanstd(depth)+1e-6))
tw = (np.where(np.isfinite(tw) & (tw > -2147483647), tw, 0)/255-.5)/.5
arrays = [ctx, depth, v0, v1, *tw]
# Match training patch dimensions; normalize on the stored tile before cropping.
arrays = [a[64:320, 64:320] for a in arrays]
names = ['CTX: grayscale bathymetry', 'Standardized elevation / depth',
         'CTX validity mask', 'Depth validity mask',
         'Hillshade: azimuth 315 deg', 'Hillshade: azimuth 75 deg', 'Hillshade: azimuth 195 deg']
slugs = ['01_ctx', '02_depth', '03_valid_ctx', '04_valid_depth', '05_hillshade_315', '06_hillshade_075', '07_hillshade_195']
cmaps = ['gray', 'viridis', 'gray', 'gray', 'gray', 'gray', 'gray']
lims = [(-1,1), (float(arrays[1].min()),float(arrays[1].max())), (0,1), (0,1), (-1,1), (-1,1), (-1,1)]
def draw(ax, i):
    a = arrays[i]
    lo, hi = lims[i]
    im = ax.imshow(a, cmap=cmaps[i], vmin=lo, vmax=hi, interpolation='nearest', origin='upper')
    ax.set_title(f'{i+1:02d}  {names[i]}', fontsize=13, loc='left', pad=12)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#cbd5e1')
    if i in (2,3):
        ax.set_xlabel(f'1 = valid, 0 = missing | {a.mean():.0%} valid', fontsize=11)
    elif i == 1:
        ax.set_xlabel('Per-tile z-score; color is for display', fontsize=10)
    else:
        ax.set_xlabel('Model input range: -1 to +1', fontsize=11)
    return im
fig, axs = plt.subplots(2,4,figsize=(17,9), facecolor='white')
for i in range(7):
    draw(axs.flat[i], i)
axs.flat[7].axis('off')
axs.flat[7].text(.02,.9,'7-channel model input', fontsize=20, weight='bold', va='top')
axs.flat[7].text(.02,.72,'2 terrain representations\n2 validity masks\n3 directional hillshades\n\nSame location, aligned pixels\n256 x 256 center crop\n\nFault labels are separate.', fontsize=15, linespacing=1.6, va='top')
fig.suptitle('Earthquake-SAM | Seven input channels', fontsize=24, weight='bold', y=.98)
fig.text(.5,.025,f'Source: local GEBCO-derived training data | Tile {key} | No augmentation | Channel order follows tool.py', ha='center', fontsize=11, color='#475569')
fig.subplots_adjust(left=.035,right=.98,top=.88,bottom=.1,wspace=.18,hspace=.3)
fig.savefig(OUT/'seven_input_channels.png',dpi=200)
plt.close(fig)
for i,slug in enumerate(slugs):
    fig,ax=plt.subplots(figsize=(6,6))
    draw(ax,i)
    fig.tight_layout()
    fig.savefig(OUT/f'{slug}.png',dpi=220)
    plt.close(fig)
    plt.imsave(OUT/f'{slug}_plain.png',arrays[i],cmap=cmaps[i],vmin=lims[i][0],vmax=lims[i][1],origin='upper')
(OUT/'provenance.json').write_text(json.dumps({'data':str(DATA),'tile':key,'tile_attributes':attrs,'crop':[64,320,64,320],'normalization':'Same as evaluate_cv.inputs_for_tile, then center crop; no augmentation','channels':names,'display_limits':lims},indent=2),encoding='utf-8')
(OUT/'图片说明.md').write_text('''# 七通道输入图

总览：seven_input_channels.png。01–07 为带标题单图，_plain 为无标题原尺寸图。

按代码顺序：1 灰度地形 CTX；2 按整块均值、标准差归一化的水深/高程；3 CTX 有效掩膜；4 水深有效掩膜；5–7 方位角 315°、75°、195° 的晕渲。

这些图来自同一个真实 HDF5 样本；先按 evaluate_cv.py 的规则归一化，再取中心 256×256 区域，无随机增强。水深使用伪彩色方便展示，实际仍为单通道数值；灰度和晕渲按 [-1,1] 显示。掩膜白色为1，黑色为0，全白表示全有效，两个掩膜可能相同。

CTX、image、TW 是沿用的字段名。在当前地球数据中，它们均来源于 GEBCO 地形及其处理，不代表七种独立传感器，也不是火星 CTX 影像。断层目录标签 k0/k1/k2 不包含在七通道输入里。七通道是完整模型的输入，不表示原版 SAM-2 编码器直接接收七通道。

来源与样本坐标见 provenance.json；脚本 export_channels.py 可复现。
''',encoding='utf-8')
with zipfile.ZipFile(OUT/'seven_input_channels_pack.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.iterdir():
        if p.suffix in ('.png','.md','.json','.py'):
            z.write(p,p.name)
print(json.dumps({'tile':key,'attrs':attrs,'output':str(OUT)},ensure_ascii=False))
