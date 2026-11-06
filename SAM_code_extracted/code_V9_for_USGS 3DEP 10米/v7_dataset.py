"""Opt-in full-tile V7 reader. Does not change any existing training loader.

Returns numpy (7,1024,1024) inputs and (4,1024,1024) distances/known mask.
Optional D4 augmentation preserves 30 m pixels and the entire field of view.
Selected by --fullTile; legacy experiments retain Generator_new.
"""
from pathlib import Path
from types import SimpleNamespace
import h5py
import numpy as np


def d4(a,k):
    if k&1: a=a[:,::-1]
    if k&2: a=a[:,:,::-1]
    if k&4: a=a.transpose(0,2,1)
    return np.ascontiguousarray(a)


class FullTileDataset:
    def __init__(self,filename,keys,augment=False):
        self.filename=str(filename)
        self.keys=list(keys)
        self.augment=augment

    def __len__(self):
        return len(self.keys)

    def __getitem__(self,index):
        with h5py.File(self.filename,'r') as f:
            g=f[self.keys[index]]
            ctx=g['CTX'][:].astype('float32')
            z=g['image'][:].astype('float32')
            tw=g['TW'][:].astype('float32')
            valid=g['valid_input'][:].astype('float32')
            known=g['label_known'][:].astype('float32')
            labels=np.stack([g[n][:] for n in ('k0','k1','k2')]).astype('float32')
        assert ctx.shape==(1024,1024) and known.shape==ctx.shape
        assert np.array_equal(valid>0,ctx>-2147483647)
        assert np.array_equal(valid>0,z>-2147483647)
        ctx[ctx<-2147483647]=0
        tw[tw<-2147483647]=0
        ctx=(ctx/255-.5)/.5
        tw=(tw/255-.5)/.5
        z[z<-2147483647]=np.nan
        z=(z-np.nanmean(z))/(np.nanstd(z)+1e-6)
        z[np.isnan(z)]=0
        x=np.concatenate([ctx[None],z[None],valid[None],valid[None],tw]).astype('float32')
        labels[(labels<0)|(labels>=.35)]=999
        y=np.concatenate([labels,known[None]])
        if self.augment:
            k=int(np.random.randint(8)); x,y=d4(x,k),d4(y,k)
        return x,y


class FullTileGenerator(FullTileDataset):
    """Adapter for the existing trainer, fixed validation and preview interfaces."""
    def __init__(self, filename, keys, m=1024, n=1024, training=False,
                 fixed_crop=False, supervision_key='label_known', known_crop=False,
                 min_known_pixels=1, known_crop_stride=32):
        if (m, n) != (1024, 1024) or supervision_key != 'label_known' or known_crop:
            raise ValueError('FullTileGenerator requires 1024px label_known tiles without crops')
        super().__init__(filename, keys, augment=training and not fixed_crop)
        self.list = self.keys
        self.excluded_keys = []
        self.dataD = {}  # compatibility only: full-tile reads are never cached


def validate_readers(path,records):
    import torch
    from tool import Generator_new
    from fault_losses import supervision
    from aio2_online import temporary_targets
    from unimatch_fault import reliable_weights,positive_consistency
    torch.set_num_threads(1)
    chosen=set()
    for split in ('train','val','test'):
        rows=[t for t in records if t['split']==split]
        chosen.add(max(rows,key=lambda t:t['known_pixels'])['key'])
        chosen.add(max(rows,key=lambda t:t['unknown_valid_pixels'])['key'])
        chosen.add(min(rows,key=lambda t:t['valid_pixels'])['key'])
    keys=sorted(chosen)
    legacy=Generator_new(str(path/'faults.h5'),keys,training=False,
        m=1024,n=1024,fixed_crop=True,supervision_key='label_known',known_crop=False)
    reader=FullTileDataset(path/'faults.h5',keys)
    args=SimpleNamespace(k0Only=True,swMode='single',isOther=False,supervisionKey='label_known',
        onlineSupport=.5,onlineThreshold=.9,onlineBuffer=0,onlineMaxFraction=1.,
        onlineMinPixels=1,collapseGuard=True)
    seen_unknown=0; pseudo_checked=False
    for i,key in enumerate(keys):
        x,y=reader[i]; ox,oy=legacy[i]; legacy.dataD.clear()
        assert x.shape==(7,1024,1024) and y.shape==(4,1024,1024)
        np.testing.assert_allclose(x,ox,rtol=1e-6,atol=1e-6)
        np.testing.assert_array_equal(y,oy)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        vv=x[2]>.5
        assert abs(float(x[1][vv].mean()))<1e-4
        if np.std(x[1][vv])>0:
            assert abs(float(x[1][vv].std())-1)<1e-4
        assert not x[1][~vv].any()
        tx,ty=torch.from_numpy(x[None]),torch.from_numpy(y[None])
        target,weights,valid=supervision(tx,ty,args,.01,.15,.1)
        known=ty[:,3:4]>.5
        assert torch.equal(weights>0,valid&known)
        assert not target[~(valid&known)].any()
        unknown=valid&~known
        seen_unknown+=int(unknown.sum())
        if not pseudo_checked and int(unknown.sum())>=16:
            probability=torch.zeros_like(target)
            indices=torch.nonzero(unknown.flatten(),as_tuple=False)[:16,0]
            probability.flatten()[indices]=.99
            corrected,pw,selected,_=temporary_targets(target,weights,probability,tx,.4,args)
            assert int(selected.sum())==16 and (pw[selected]>0).all()
            assert not selected[~valid].any() and not selected[known].any()
            cw=reliable_weights(target,selected,tx,.4)
            loss=positive_consistency(torch.full_like(target,.8),cw)
            assert torch.isfinite(loss) and loss>0
            pseudo_checked=True
        for k in range(8):
            xx,yy=d4(x,k),d4(y,k)
            assert np.isin(yy[3],[0,1]).all()
            assert not ((yy[3]>.5)&~(xx[2]>.5)).any()
            assert xx.shape==x.shape and yy.shape==y.shape
    assert pseudo_checked and seen_unknown>0
    return dict(status='PASS',full_size=[1024,1024],keys_checked=keys,
        new_reader_matches_legacy_fixed_full_crop=True,d4_transforms_checked=8,
        valid_unknown_land_pixels=seen_unknown,synthetic_pseudo_and_consistency_eligibility=True,
        invalid_pixels_excluded=True,training_code_modified=False,training_started=False,
        model_constructed=False,backward_or_optimizer_called=False,
        training_loader_integration='pending future training task; old random-scale crop is not used')
