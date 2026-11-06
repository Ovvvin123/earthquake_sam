"""Deterministic crop inventory for incomplete labels; never writes HDF5 data."""
import h5py
import numpy as np


def crop_inventory(filename, keys, supervision_key, height=256, width=256,
                   minimum=1, stride=32):
    if not supervision_key or minimum < 1 or stride < 1:
        raise ValueError('Known crops require a mask, positive minimum and stride')
    windows, excluded = {}, []
    with h5py.File(filename, 'r') as f:
        for key in keys:
            g=f[key]
            if supervision_key not in g:
                raise ValueError(f'{key}: missing {supervision_key}')
            mask=g[supervision_key][:]
            ctx=g['CTX'][:]; depth=g['image'][:]; tw=g['TW'][:]
            if mask.shape != ctx.shape or not np.isin(mask,[0,1]).all():
                raise ValueError(f'{key}: malformed supervision mask')
            valid=(np.isfinite(ctx)&np.isfinite(depth)&np.isfinite(tw).all(axis=0)
                   &(ctx > -2147483647)&(depth > -2147483647))
            known=(mask > .5)&valid
            h,w=known.shape
            if h < height or w < width:
                raise ValueError(f'{key}: tile smaller than crop')
            ys=sorted(set(range(0,h-height+1,stride))|{h-height})
            xs=sorted(set(range(0,w-width+1,stride))|{w-width})
            integral=np.pad(known.astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1)
            choices=[]
            for y in ys:
                for x in xs:
                    count=int(integral[y+height,x+width]-integral[y,x+width]
                              -integral[y+height,x]+integral[y,x])
                    if count >= minimum: choices.append((y,x,count))
            if choices: windows[key]=choices
            else: excluded.append(key)
    return windows, excluded


def choose_crop(choices, training):
    if training:
        # Favor larger known areas without discarding thin coastal-edge samples.
        weights=np.asarray([v[2] for v in choices],dtype=np.float64)
        return choices[np.random.choice(len(choices),p=weights/weights.sum())][:2]
    # Fixed validation: largest known area, stable top/left tie break.
    return max(choices,key=lambda v:(v[2],-v[0],-v[1]))[:2]
