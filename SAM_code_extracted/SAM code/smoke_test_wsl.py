"""Validate real v5 data, CUDA forward/backward and one optimizer step.

Runs in memory only; never overwrites a trained checkpoint.
"""
import argparse
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import torch
from tool import Generator_new, collate_function
from torchUnet import conv_sam, FocalTversky


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", type=int, default=2)
    parser.add_argument("--dataDir", default="earth_dataset_v3")
    parser.add_argument("--output", default="wsl_validation/smoke.json")
    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA unavailable in this Python environment"
    torch.set_num_threads(4)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    data = Path(args.dataDir)
    keys = (data / "key_train.lst").read_text().splitlines()
    ds = Generator_new(str(data / "faults.h5"), keys, m=256, n=256, training=True)
    batch = collate_function([ds[i] for i in range(args.batchsize)])
    image = torch.from_numpy(batch[0]).float().cuda()
    distance = torch.from_numpy(batch[1]).float().cuda()[:, :1]
    assert torch.isfinite(image).all() and torch.isfinite(distance).all()
    weights = torch.exp(-distance.square() / 0.08**2) * 0.9 + 0.1
    weights = torch.where(image[:, 2:4].amin(dim=1, keepdim=True) < 1e-9, 1., weights)
    weights = weights / weights.mean()
    truth = (distance < 0.02).float()
    model = conv_sam(32, dropout=0., isvit=False, isalign=False).cuda()
    model.train()
    model.in_sam.eval()
    # Match the actual training script's optimizer parameter groups.
    optimizer = torch.optim.AdamW([
        {"params": model.not_sam.parameters(), "lr": 1e-4},
        {"params": model.in_sam.parameters(), "lr": 1e-4 / 8},
    ])
    before = model.head[0].weight.detach().clone()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    prediction = model(image)
    assert prediction.shape == (args.batchsize, 3, 256, 256)
    assert torch.isfinite(prediction).all()
    loss = FocalTversky(truth, prediction, sw=weights, pu=0.)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(torch.count_nonzero(g).item() for g in gradients)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    optimizer.step()
    torch.cuda.synchronize()
    assert not torch.equal(before, model.head[0].weight), "Optimizer did not update parameters"
    report = {
        "status": "passed", "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "batchsize": args.batchsize,
        "input_shape": list(image.shape), "output_shape": list(prediction.shape),
        "loss": loss.item(), "step_seconds": time.perf_counter() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "finite_gradients": True, "optimizer_updated_parameters": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("WSL GPU SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
