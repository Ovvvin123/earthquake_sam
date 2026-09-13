"""
smoke_test_mac.py  —  RUN THIS FIRST on the Mac, before the long training.
------------------------------------------------------------------
It does ONE real training step with the actual SAM model + your dataset:
  Generator_new -> conv_sam (SAM-2) -> Dice loss -> backward -> optimizer step
If it prints "SMOKE TEST PASSED", the full run in run.sh will work too.
Takes <1-2 min on CPU/MPS. Place this file inside the "SAM code" folder and run:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python smoke_test_mac.py
"""
import os, numpy as np, torch
from tool import Generator_new, collate_function
from torchUnet import conv_sam, Dice

H5  = 'earth_dataset/ca_offshore_faults.h5'
KEY = 'earth_dataset/key.lst'
D0, DW, ALPHA = 0.02, 0.08, 0.05      # match recommended training settings

# ---- device (CUDA / Apple MPS / CPU) ----
if torch.cuda.is_available(): device='cuda:0'
elif getattr(torch.backends,'mps',None) and torch.backends.mps.is_available(): device='mps'
else: device='cpu'
print('device =', device)

# ---- data: pull one real batch through the REAL Generator_new ----
keys = open(KEY).read().split('\n')[:-1]
ds = Generator_new(H5, keys, m=256, n=256, training=True)
batch = collate_function([ds[0], ds[1]])          # list: [image(2,7,256,256), label(2,3,256,256)]
image  = torch.from_numpy(batch[0]).float().to(device)
target = torch.from_numpy(batch[1]).float().to(device)
print('input  :', tuple(image.shape), '(expect 2,7,256,256)')
print('label  :', tuple(target.shape), '(expect 2,3,256,256)')

# ---- model: real SAM-2 wrapper (mars_sam = frozen backbone) ----
net = conv_sam(32, dropout=0., isvit=False, isalign=False).to(device)
opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
n_all = sum(p.numel() for p in net.parameters())
n_tr  = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f'model params: total {n_all:,}  trainable {n_tr:,}')

# ---- replicate trainV10.py train step exactly ----
target, others = target[:, :2], target[:, 2:]
tMin = target + 0
sw = torch.exp(-tMin**2/DW**2)*(1-ALPHA)+ALPHA
target = torch.where(target < D0, 1., 0.)
valid_01 = image[:, 2:4].amin(dim=1, keepdim=True)
sw = torch.where(valid_01 < 1e-9, torch.ones_like(sw), sw)
sw = sw/sw.mean()

net.train(); net.in_sam.eval()
opt.zero_grad()
predict = net(image)
print('predict:', tuple(predict.shape))
loss = Dice(target, predict, sw=sw)
loss.backward()
torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
opt.step()

val = float(loss.item())
assert np.isfinite(val), 'loss is NaN/Inf'
assert predict.shape[-2:] == (256, 256), 'wrong output size'
print(f'loss = {val:.4f}')
print('\n=== SMOKE TEST PASSED — the full run.sh command will work. ===')
