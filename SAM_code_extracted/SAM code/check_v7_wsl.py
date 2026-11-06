"""No optimizer or training: data adapter/preview and optional GPU inference checks."""
import argparse
import json
from pathlib import Path
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-inference', action='store_true')
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    import numpy as np
    import torch
    from v7_dataset import FullTileGenerator
    from tool import Generator_new
    from aio2_preview import FixedPreview
    from run_experiment import read_profile
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parent
    data = root/'Earth_dataset_v7'
    keys = {split: (data/f'key_{split}.lst').read_text().split() for split in ('train','val','test')}
    assert not set(keys['train']) & (set(keys['val']) | set(keys['test']))
    assert not set(keys['val']) & set(keys['test'])
    report = dict(training_started=False, optimizer_steps=0, splits={k:len(v) for k,v in keys.items()})
    for split, listed in keys.items():
        reader = FullTileGenerator(data/'faults.h5', listed[:1])
        x, y = reader[0]
        legacy = Generator_new(str(data/'faults.h5'), listed[:1], m=1024,n=1024,
                               training=False,fixed_crop=True,supervision_key='label_known')
        a, b = legacy[0]
        np.testing.assert_allclose(x, a, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(y, b)
        del legacy
    reader = FullTileGenerator(data/'faults.h5', keys['train'][:1], training=True)
    x, y = FullTileGenerator(data/'faults.h5', keys['train'][:1])[0]
    from v7_dataset import d4
    from unittest.mock import patch
    for transform in range(8):
        with patch('numpy.random.randint', return_value=transform):
            a,b = reader[0]
        np.testing.assert_array_equal(a, d4(x, transform))
        np.testing.assert_array_equal(b, d4(y, transform))
    with tempfile.TemporaryDirectory() as folder:
        preview = FixedPreview(FullTileGenerator, data/'faults.h5', keys['train'][:2],
                               folder,1024,1024,.01,count=1,dataset_options={'supervision_key':'label_known'})
        assert preview.samples[0][1].shape == (1,7,1024,1024)
    report['reader_preview'] = 'PASS'
    if args.gpu_inference:
        from torchUnet import conv_sam
        cfg = read_profile(root/'experiments/california_v7_30m_1024_wsl.json')
        if not torch.cuda.is_available():
            report['gpu_inference'] = 'UNAVAILABLE'
        else:
            report['gpu'] = torch.cuda.get_device_name(0)
            try:
                model = conv_sam(32,isvit=False,adaptation=cfg,n=1024,m=1024).cuda().eval()
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():
                    output = model(torch.from_numpy(x[None]).cuda(), return_logits=True)
                torch.cuda.synchronize()
                assert tuple(output.shape) == (1,3,1024,1024) and torch.isfinite(output).all()
                report.update(gpu_inference='PASS',output_shape=list(output.shape),peak_cuda_bytes=torch.cuda.max_memory_allocated())
            except torch.cuda.OutOfMemoryError as error:
                report.update(gpu_inference='OUT_OF_MEMORY',error=str(error))
            report['training_memory_validated'] = False
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
