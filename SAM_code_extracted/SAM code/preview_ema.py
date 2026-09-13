"""Preview an existing EMA checkpoint without training or modifying weights."""
import argparse
from pathlib import Path
import torch

from aio2_ema import EMATeacher
from aio2_preview import FixedPreview

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--outDir', default='')
    parser.add_argument('--count', type=int, default=6)
    parser.add_argument('--threshold', type=float, default=.7)
    parser.add_argument('--buffer', type=int, default=2)
    parser.add_argument('--minPixels', type=int, default=8)
    options = parser.parse_args()
    state = torch.load(options.checkpoint, map_location='cpu', weights_only=True)
    if state.get('format') != 'aio2_ema_v1':
        parser.error('Expected ema_checkpoint.pth.')
    config = state['config']
    if config['modelType'] not in ('mars_sam', 'mars_sam_vit'):
        parser.error('Standalone preview currently supports mars_sam and mars_sam_vit.')
    dataset_path = Path(config['dataset'])
    key_path = dataset_path.parent / 'key_train.lst'
    if not key_path.exists():
        parser.error('Standalone preview requires the spatial key_train.lst.')
    filename = dataset_path.parent / 'faults.h5'
    if not filename.exists():
        filename = Path('earth_dataset_v3/faults.h5')
    keys = [key for key in key_path.read_text().splitlines() if key]
    if config.get('datasetAdd'):
        keys += [key for key in Path(config['datasetAdd']).read_text().splitlines() if key]
    from tool import Generator_new
    from torchUnet import conv_sam
    device = (f'cuda:{config.get("gpuIndex", 0)}' if torch.cuda.is_available()
              else 'mps' if torch.backends.mps.is_available() else 'cpu')
    student = conv_sam(32, dropout=config['dropout'], isvit=config['modelType']=='mars_sam_vit',
                       isalign=config['isalign']).to(device)
    student.load_state_dict(state['student'])
    teacher = EMATeacher(student, config['emaDecay'])
    teacher.load_state_dict(state['teacher'])
    epoch = int(state['next_epoch'])
    del state
    sizes = str(config['mn']).split(',')
    output = options.outDir or str(Path(options.checkpoint).parent / 'preview')
    preview = FixedPreview(Generator_new, str(filename), keys, output, int(sizes[0]), int(sizes[-1]),
                           config['d0'], options.count, options.threshold, options.buffer, options.minPixels)
    result = preview.write(student, teacher, epoch)
    print(f'Preview saved: {result}; epoch={epoch}; EMA updates={teacher.num_updates}; no training performed.')

if __name__ == '__main__':
    main()
