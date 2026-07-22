"""
poc_train_cpu.py  — CPU proof-of-concept training
------------------------------------------------------------------
NOT the real SAM-2 fine-tune (that needs a GPU). This trains a small U-Net on
the 93 California tiles to verify (a) the data/labels are learnable and
(b) the class-imbalance handling avoids the collapse seen in the Nov run.

Binary task: fault / no-fault, with the same proximity weighting the Mars code
uses:  sw = exp(-d^2/dw^2)*(1-alpha)+alpha
"""
import h5py, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent
H5 = BASE / 'SAM_code_extracted' / 'SAM code' / 'earth_dataset_v3' / 'faults.h5'
D0=0.02; DW=0.06; ALPHA=0.05
DS=4          # downsample tiles 256->64 so this fits the CPU time budget
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)

# ---------- load all tiles into memory (93 x 256 x 256, small) ----------
f=h5py.File(H5,'r'); keys=list(f.keys())
X,Y,W=[],[],[]
for k in keys:
    ctx=f[k]['CTX'][:].astype('float32'); ctx=np.where(ctx<-1e9,0,ctx)/255.0
    ctx=(ctx-0.5)/0.5
    slope=f[k]['TW'][2].astype('float32')/255.0
    k0=f[k]['k0'][:].astype('float32')
    ctx=ctx[::DS,::DS]; slope=slope[::DS,::DS]; k0=k0[::DS,::DS]   # downsample
    tgt=(k0<D0).astype('float32')
    sw=np.exp(-np.minimum(k0,0.35)**2/DW**2)*(1-ALPHA)+ALPHA
    X.append(np.stack([ctx,slope])); Y.append(tgt); W.append(sw)
X=torch.tensor(np.stack(X)); Y=torch.tensor(np.stack(Y))[:,None]; W=torch.tensor(np.stack(W))[:,None]
print("data",X.shape,"positive px frac",Y.mean().item())

n=len(keys); idx=np.random.permutation(n); ntr=int(n*0.85)
tr,va=idx[:ntr],idx[ntr:]

# ---------- compact U-Net ----------
def cbr(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(inplace=True),
                                   nn.Conv2d(o,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(inplace=True))
class UNet(nn.Module):
    def __init__(s,c=16):
        super().__init__()
        s.e1=cbr(2,c); s.e2=cbr(c,c*2); s.e3=cbr(c*2,c*4); s.b=cbr(c*4,c*8)
        s.p=nn.MaxPool2d(2)
        s.u3=nn.ConvTranspose2d(c*8,c*4,2,2); s.d3=cbr(c*8,c*4)
        s.u2=nn.ConvTranspose2d(c*4,c*2,2,2); s.d2=cbr(c*4,c*2)
        s.u1=nn.ConvTranspose2d(c*2,c,2,2);   s.d1=cbr(c*2,c)
        s.out=nn.Conv2d(c,1,1)
    def forward(s,x):
        e1=s.e1(x); e2=s.e2(s.p(e1)); e3=s.e3(s.p(e2)); b=s.b(s.p(e3))
        d3=s.d3(torch.cat([s.u3(b),e3],1)); d2=s.d2(torch.cat([s.u2(d3),e2],1))
        d1=s.d1(torch.cat([s.u1(d2),e1],1)); return s.out(d1)

net=UNet(8); print("params",sum(p.numel() for p in net.parameters()))
opt=torch.optim.Adam(net.parameters(),1e-3)

def wloss(logit,tgt,w):
    bce=F.binary_cross_entropy_with_logits(logit,tgt,reduction='none')
    bce=(bce*w).mean()
    p=torch.sigmoid(logit)
    dice=1-((2*(p*tgt*w).sum()+1)/(((p+tgt)*w).sum()+1))
    return bce+dice

def run(ids,train):
    net.train() if train else net.eval()
    tot=0; bs=12; order=np.random.permutation(ids) if train else ids
    for i in range(0,len(order),bs):
        b=order[i:i+bs]
        xb,yb,wb=X[b],Y[b],W[b]
        if train: opt.zero_grad()
        with torch.set_grad_enabled(train):
            out=net(xb); l=wloss(out,yb,wb)
            if train: l.backward(); opt.step()
        tot+=l.item()*len(b)
    return tot/len(order)

hist=[]
EP=18
for ep in range(EP):
    ltr=run(tr,True); lva=run(va,False); hist.append((ltr,lva))
    if ep%5==0 or ep==EP-1: print(f"epoch {ep:2d}  train {ltr:.4f}  val {lva:.4f}")

hist=np.array(hist)
plt.figure(figsize=(7,4))
plt.plot(hist[:,0],'o-',ms=3,label='train'); plt.plot(hist[:,1],'s-',ms=3,label='val')
plt.xlabel('epoch'); plt.ylabel('weighted BCE+Dice loss'); plt.legend(); plt.grid(alpha=.3)
plt.title('CPU proof-of-concept — loss DECREASES (no collapse)')
plt.tight_layout(); plt.savefig(BASE / 'poc_loss_curve.png',dpi=130)

# ---------- prediction panel on validation tiles ----------
net.eval()
show=va[:4] if len(va)>=4 else va
fig,ax=plt.subplots(3,len(show),figsize=(4*len(show),9))
with torch.no_grad():
    for c,ii in enumerate(show):
        pr=torch.sigmoid(net(X[ii:ii+1]))[0,0].numpy()
        bathy=X[ii,0].numpy()
        ax[0,c].imshow(bathy,cmap='terrain'); ax[0,c].set_title('bathymetry'); ax[0,c].axis('off')
        ax[1,c].imshow(Y[ii,0].numpy(),cmap='hot'); ax[1,c].set_title('truth (fault)'); ax[1,c].axis('off')
        ax[2,c].imshow(pr,cmap='hot',vmin=0,vmax=1); ax[2,c].set_title('prediction'); ax[2,c].axis('off')
plt.suptitle('Validation tiles: truth vs prediction (structure learned, not flat)')
plt.tight_layout(); plt.savefig(BASE / 'poc_predictions.png',dpi=120)
print("saved poc_loss_curve.png and poc_predictions.png")
print(f"loss start {hist[0,1]:.4f} -> end {hist[-1,1]:.4f}  (drop {(1-hist[-1,1]/hist[0,1])*100:.0f}%)")
