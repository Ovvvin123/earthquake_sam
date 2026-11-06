"""
evaluate.py  —  publication-grade evaluation on the spatial TEST split
======================================================================
Run AFTER training, on the Mac/GPU machine, from inside the "SAM code" folder:

    PYTORCH_ENABLE_MPS_FALLBACK=1 python evaluate.py --resDir train_ca_offshore_v1/

Loads model.pth, runs deterministic CENTRE crops over every key in
earth_dataset/key_test.lst (a spatially-separated region the model never saw),
and reports, for k0 (any fault) and k1 (dip-slip fault):

  * Average Precision (area under precision-recall curve) — threshold-free
  * Best-F1 operating point: precision / recall / F1 / IoU
  * Buffered detection metrics (a hit counts if within TOL_PX of a true fault),
    which is the fair way to score thin curvilinear targets
Outputs: <resDir>/test_metrics.json  and  <resDir>/test_qualitative.png
"""
import os, json, argparse, numpy as np, torch
from h5py import File
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, binary_dilation
from torchUnet import conv_sam

ap = argparse.ArgumentParser()
ap.add_argument('--resDir', default='train_ca_offshore_v1/')
ap.add_argument('--dataDir', default='earth_dataset')
ap.add_argument('--modelType', default='mars_sam')
ap.add_argument('--crop', type=int, default=256)
ap.add_argument('--d0', type=float, default=0.02)     # truth threshold (deg)
ap.add_argument('--tol_px', type=int, default=3)      # buffer for detection metrics
ap.add_argument('--keyFile', default='key_test.lst',  # e.g. key_test_unseen.lst for fair old-vs-new comparison
                help='key list inside dataDir to evaluate on')
args = ap.parse_args()

H5 = f"{args.dataDir}/faults.h5"
keys = [k for k in open(f"{args.dataDir}/{args.keyFile}").read().split('\n') if k]
print(f"test tiles: {len(keys)}")

dev = 'cuda:0' if torch.cuda.is_available() else ('mps' if getattr(torch.backends,'mps',None)
       and torch.backends.mps.is_available() else 'cpu')
from sam2_lora import load_run_model, read_model_config
cfg = read_model_config(args.resDir)
cfg.setdefault('modelType', args.modelType)
net = load_run_model(args.resDir, dev, cfg)
net.eval()

def build_input_full(g):
    """Normalise the FULL stored tile (no crop)."""
    CTX = g['CTX'][:].astype('float32'); mole = g['image'][:].astype('float32'); TW = g['TW'][:].astype('float32')
    k0=g['k0'][:]; k1=g['k1'][:]
    valid_0=np.where(CTX>-2147483647,1,0).astype('float32')
    valid_1=np.where(mole>-2147483647,1,0).astype('float32')
    CTX=CTX.copy(); CTX[CTX<-2147483647]=0; CTX=(CTX/255-0.5)/0.5
    mole=mole.copy(); mole[mole<-2147483647]=np.nan
    mole=(mole-np.nanmean(mole))/(np.nanstd(mole)+1e-6); mole[np.isnan(mole)]=0
    TW=TW.copy(); TW[TW<-2147483647]=0; TW=(TW/255-0.5)/0.5
    inp=np.concatenate([CTX[None],mole[None],valid_0[None],valid_1[None],TW]).astype('float32')
    return inp, k0, k1

# PATCH (v5): sliding-window over the whole 384px tile instead of a single
# centre crop — the old version silently ignored the outer 1/3 of every test
# tile. Overlapping window predictions are averaged.
P0,P1,T0,T1=[],[],[],[]
with torch.no_grad():
    for k in keys:
        inp,k0,k1=build_input_full(File(H5,'r')[k])
        C=args.crop; H,W=inp.shape[1],inp.shape[2]
        acc=np.zeros((2,H,W),'float32'); wgt=np.zeros((H,W),'float32')
        ii=sorted({0,(H-C)//2,H-C}); jj=sorted({0,(W-C)//2,W-C})
        for i0 in ii:
            for j0 in jj:
                out=net(torch.from_numpy(inp[None,:,i0:i0+C,j0:j0+C]).to(dev)).cpu().numpy()[0]
                acc[:,i0:i0+C,j0:j0+C]+=out[:2]; wgt[i0:i0+C,j0:j0+C]+=1
        acc/=wgt[None]
        P0.append(acc[0]); P1.append(acc[1])
        T0.append((k0<args.d0).astype('uint8')); T1.append((k1<args.d0).astype('uint8'))
P0,P1=np.array(P0),np.array(P1); T0,T1=np.array(T0),np.array(T1)

def pr_metrics(prob, truth, tol):
    p=prob.ravel(); t=truth.ravel()
    # Average Precision via threshold sweep
    ths=np.linspace(0,1,101); prec=[]; rec=[]
    for th in ths:
        pred=p>=th; tp=(pred&(t>0)).sum(); fp=(pred&(t==0)).sum(); fn=((~pred)&(t>0)).sum()
        prec.append(tp/(tp+fp+1e-9)); rec.append(tp/(tp+fn+1e-9))
    prec=np.array(prec); rec=np.array(rec)
    order=np.argsort(rec); ap=np.trapezoid(prec[order],rec[order])
    f1=2*prec*rec/(prec+rec+1e-9); bi=int(np.nanargmax(f1)); bth=ths[bi]
    predb=p>=bth; tp=(predb&(t>0)).sum(); fp=(predb&(t==0)).sum(); fn=((~predb)&(t>0)).sum()
    iou=tp/(tp+fp+fn+1e-9)
    # buffered detection metrics (per-tile, tolerance tol px)
    bp=bn=brp=brn=0
    for pr2d,tr2d in zip(prob,truth):
        pm=pr2d>=bth
        if tr2d.any():
            tr_buf=binary_dilation(tr2d>0,iterations=tol)
            brp+=(pm&tr_buf).sum(); brn+=(pm).sum()
        if pm.any():
            pm_buf=binary_dilation(pm,iterations=tol)
            bp+=((tr2d>0)&pm_buf).sum(); bn+=(tr2d>0).sum()
    buf_prec=brp/(brn+1e-9); buf_rec=bp/(bn+1e-9)
    buf_f1=2*buf_prec*buf_rec/(buf_prec+buf_rec+1e-9)
    return dict(AP=float(ap), best_threshold=float(bth),
                precision=float(prec[bi]), recall=float(rec[bi]),
                F1=float(f1[bi]), IoU=float(iou),
                buffered_precision=float(buf_prec), buffered_recall=float(buf_rec),
                buffered_F1=float(buf_f1), positive_pixel_frac=float((t>0).mean()))

res={'n_test_tiles':len(keys),'d0_deg':args.d0,'tol_px':args.tol_px,
     'k0_any_fault':pr_metrics(P0,T0,args.tol_px),
     'k1_dip_slip' :pr_metrics(P1,T1,args.tol_px)}
_stem = args.keyFile.replace('key_','').replace('.lst','')
json.dump(res, open(f"{args.resDir}/{_stem}_metrics.json",'w'), indent=2)
print(json.dumps(res, indent=2))

# qualitative panel
n=min(5,len(keys)); idx=np.linspace(0,len(keys)-1,n).astype(int)
fig,ax=plt.subplots(3,n,figsize=(4*n,9))
for c,ii in enumerate(idx):
    ax[0,c].imshow(T0[ii],cmap='hot'); ax[0,c].set_title('truth (any fault)'); ax[0,c].axis('off')
    ax[1,c].imshow(P0[ii],cmap='hot',vmin=0,vmax=1); ax[1,c].set_title('pred k0'); ax[1,c].axis('off')
    ax[2,c].imshow(P1[ii],cmap='hot',vmin=0,vmax=1); ax[2,c].set_title('pred k1 (dip-slip)'); ax[2,c].axis('off')
plt.suptitle('TEST split — truth vs prediction'); plt.tight_layout()
plt.savefig(f"{args.resDir}/{_stem}_qualitative.png",dpi=120)
print(f"saved {_stem}_metrics.json and {_stem}_qualitative.png")
