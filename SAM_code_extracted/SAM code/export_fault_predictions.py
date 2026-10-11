"""Export full-tile seen-training vs spatially held-out predictions; no training."""
import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np

from evaluate_cv import inputs_for_tile, positions, split_audit


def predict_tile(model, image, crop, device):
    import torch
    height, width = image.shape[-2:]
    summed = np.zeros((height, width), np.float32)
    count = np.zeros((height, width), np.uint8)
    with torch.inference_mode():
        for y in positions(height, crop):
            for x in positions(width, crop):
                tensor = torch.from_numpy(image[None, :, y:y+crop, x:x+crop]).to(device)
                probability = model(tensor)[0, 0].cpu().numpy()
                if not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
                    raise ValueError('Invalid model probabilities.')
                summed[y:y+crop, x:x+crop] += probability
                count[y:y+crop, x:x+crop] += 1
    if not count.all():
        raise ValueError('Inference left uncovered pixels.')
    return summed / count


def save_panel(path, image, truth, valid, probability, threshold, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    terrain = np.clip(image[0] / 2 + .5, 0, 1)
    prediction = (probability >= threshold) & valid
    terrain_rgb = np.repeat(terrain[..., None], 3, axis=-1)
    terrain_rgb[~valid] = .25

    def overlay(mask, color):
        rgb = terrain_rgb.copy()
        rgb[mask] = rgb[mask] * .35 + np.asarray(color) * .65
        return rgb

    annotations = overlay(truth, [.95, .2, .75])
    predictions = overlay(prediction, [.05, .8, 1.])
    agreement = terrain_rgb.copy() * .55
    agreement[truth & prediction] = [.15, .85, .4]
    agreement[truth & ~prediction] = [1., .7, .05]
    agreement[prediction & ~truth] = [1., .2, .65]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    axes[0].imshow(terrain_rgb)
    axes[0].set_title('Terrain / CTX')
    axes[1].imshow(annotations)
    axes[1].set_title('Original annotated faults')
    prob = axes[2].imshow(np.ma.array(probability, mask=~valid), cmap='viridis', vmin=0, vmax=1)
    axes[2].set_title('Student fault probability')
    fig.colorbar(prob, ax=axes[2], fraction=.046, pad=.03)
    axes[3].imshow(predictions)
    axes[3].set_title(f'Predicted faults (threshold={threshold:g})')
    axes[4].imshow(agreement)
    axes[4].set_title('Prediction vs original annotations')
    depth = axes[5].imshow(np.ma.array(image[1], mask=~valid), cmap='terrain', vmin=-3, vmax=3)
    axes[5].set_title('Normalized depth (input)')
    fig.colorbar(depth, ax=axes[5], fraction=.046, pad=.03)
    for ax in axes:
        ax.axis('off')
    fig.suptitle(title, fontsize=12)
    legend = [Patch(color=[.15, .85, .4], label='Prediction overlaps annotation'),
              Patch(color=[1., .7, .05], label='Annotated but not predicted'),
              Patch(color=[1., .2, .65], label='Predicted outside annotations (unverified)')]
    fig.legend(handles=legend, loc='lower center', ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, .96))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_index(output, report):
    cards = []
    labels = {'train_seen': '训练区域', 'test_unseen': '未参与训练的测试区域'}
    for row in report['images']:
        split, key = row['split'], row['key']
        relative = f'{split}/{key}.png'
        cards.append(f'''<article data-split="{split}" data-key="{html.escape(key)}">
<h3>{html.escape(key)} <small>{labels[split]}</small></h3>
<a href="{relative}" target="_blank"><img loading="lazy" src="{relative}" alt="{html.escape(key)}"></a>
<p><a href="{relative}" target="_blank">打开完整对比图</a> ·
<a href="{split}/{key}.npz">概率与掩膜数据</a></p></article>''')
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>断层预测：训练区域与独立测试区域</title>
<style>
body{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#f3f5f8;color:#1d2939;margin:0}
main{max-width:1380px;margin:auto;padding:26px}
h1{font-size:27px}p{line-height:1.7}header{background:white;padding:22px;border-radius:12px}
nav{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0}
button,input{padding:10px 14px;border:1px solid #bcc8d6;border-radius:6px;background:white;font:inherit}
button{cursor:pointer}button.active{background:#155e75;color:white}
section{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:18px}
article{background:white;padding:12px;border-radius:10px}article[hidden]{display:none}
h3{font-size:16px;margin:8px}small{font-weight:normal;color:#596579;margin-left:10px}
img{width:100%;height:auto}a{color:#126d87}article p{margin:8px;font-size:13px}
.note{color:#536277}#count{align-self:center}
</style><main><header>
<h1>d0=__D0__：训练区域与独立测试区域预测</h1>
<p>模型：__MODEL__。模型由验证 loss 选定，__EPOCH__。所有图使用同一模型、同一阈值 __THRESHOLD__。</p>
<p>训练区域共 __TRAIN__ 张；未参与训练的测试区域共 __TEST__ 张。每张图以滑窗覆盖完整图块。</p>
<p class="note">绿色：预测与原标注重合；黄色：原标注中未被预测的部分；粉色：预测超出原标注的部分。
原标注有漏标，粉色不能直接认定为错误或新发现的真实断层。训练图块之间有重叠，张数不是独立地理样本数。</p>
<p class="note">“看过”指参与训练的区域，不表示模型见过每一张完全相同的裁剪。测试区经空间检查不与训练/验证区重叠。
当前验证区与训练区有重叠，因此此处没有把验证图标为“未见区域”。本图集用于定性检查，不代替 AP 评估。</p>
</header><nav>
<button class="active" data-filter="all">全部图像</button>
<button data-filter="train_seen">训练区域</button>
<button data-filter="test_unseen">未见测试区域</button>
<input id="search" placeholder="搜索图块编号，例如 t_512"><span id="count"></span>
</nav><section>__CARDS__</section></main>
<script>
let selected='all';
const articles=[...document.querySelectorAll('article')];
function update(){const q=document.querySelector('#search').value.trim().toLowerCase();let n=0;
for(const a of articles){a.hidden=!((selected==='all'||a.dataset.split===selected)&&a.dataset.key.toLowerCase().includes(q));if(!a.hidden)n++;}
document.querySelector('#count').textContent='显示 '+n+' 张';}
document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{selected=b.dataset.filter;
document.querySelectorAll('[data-filter]').forEach(x=>x.classList.toggle('active',x===b));update();});
document.querySelector('#search').oninput=update;update();
</script></html>'''
    values = dict(D0=report['d0'], MODEL=html.escape(Path(report['model']).parent.name),
                  EPOCH=f"epoch {report['epoch']}" if report['epoch'] is not None else 'epoch 未记录',
                  THRESHOLD=report['threshold'], TRAIN=report['counts']['train_seen'],
                  TEST=report['counts']['test_unseen'], CARDS='\n'.join(cards))
    for key, value in values.items():
        document = document.replace('__'+key+'__', str(value))
    (output/'index.html').write_text(document, encoding='utf-8')


def main():
    import torch
    from h5py import File
    from torchUnet import conv_sam
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--outDir', required=True)
    parser.add_argument('--dataDir', default='earth_dataset_v3')
    parser.add_argument('--d0', type=float, default=.01)
    parser.add_argument('--threshold', type=float, default=.5)
    parser.add_argument('--crop', type=int, default=256)
    parser.add_argument('--limit', type=int, default=0, help='0 exports all tiles; otherwise first N per split, for testing only')
    args = parser.parse_args()
    if not 0 < args.threshold < 1 or not 0 < args.d0 or args.limit < 0:
        parser.error('Invalid threshold, distance threshold or limit.')
    output, data, model_path = Path(args.outDir), Path(args.dataDir), Path(args.model)
    if output.exists() and any(output.iterdir()):
        parser.error('Output is not empty; choose a new --outDir.')
    metadata_file = model_path.parent/'metrics.json'
    metadata = json.loads(metadata_file.read_text()) if metadata_file.exists() else {}
    config_file = model_path.parent.parent/'ema_config.json'
    if config_file.exists():
        config = json.loads(config_file.read_text())
        if config['d0'] != args.d0:
            parser.error('d0 differs from this training run; use its original label definition.')
        if config['modelType'] != 'mars_sam' or config.get('isalign') or config.get('datasetAdd'):
            parser.error('This exporter supports unaligned mars_sam without extra training data.')
    splits = {s:(data/f'key_{s}.lst').read_text().split() for s in ('train', 'val', 'test')}
    if any(not keys or len(set(keys)) != len(keys) for keys in splits.values()):
        raise ValueError('Empty split or duplicate keys.')
    torch.set_num_threads(4)
    torch.manual_seed(0)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    from sam2_lora import load_run_model
    model = load_run_model(model_path, device)
    model.eval()
    output.mkdir(parents=True, exist_ok=True)
    report = dict(model=str(model_path.resolve()), epoch=metadata.get('epoch'),
                  model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                  d0=args.d0, threshold=args.threshold, threshold_selection='Predefined 0.5 unless explicitly supplied; not tuned on test',
                  crop=args.crop, device=device, full_export=args.limit==0, images=[], counts={},
                  protocol='Full-tile nine-window average; panels are per tile, NOT unique-pixel aggregate metrics.')
    with File(data/'faults.h5', 'r') as h5:
        report['split_audit'] = split_audit(h5, splits)
        for name, split in [('train_seen', 'train'), ('test_unseen', 'test')]:
            keys = splits[split][:args.limit] if args.limit else splits[split]
            directory = output/name
            directory.mkdir()
            report['counts'][name] = len(keys)
            for index, key in enumerate(keys):
                image, distances, valid = inputs_for_tile(h5[key])
                truth = (distances < args.d0) & valid
                probability = predict_tile(model, image, args.crop, device)
                prediction = (probability >= args.threshold) & valid
                np.savez_compressed(directory/f'{key}.npz', probability=probability,
                                    truth=truth, valid=valid, prediction=prediction,
                                    threshold=args.threshold, d0=args.d0)
                title = f"{name} | {key} | {model_path.parent.name}, epoch={report['epoch']} | d0={args.d0}"
                save_panel(directory/f'{key}.png', image, truth, valid, probability,
                           args.threshold, title)
                report['images'].append(dict(split=name, key=key, shape=list(valid.shape),
                    valid_pixels=int(valid.sum()), annotated_pixels=int(truth.sum()),
                    predicted_pixels=int(prediction.sum())))
                if (index+1) % 10 == 0 or index+1 == len(keys):
                    print(f'{name}: {index+1}/{len(keys)} exported', flush=True)
    report['status'] = 'complete'
    (output/'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    save_index(output, report)
    print('Saved:', output/'index.html', flush=True)


if __name__ == '__main__':
    main()
