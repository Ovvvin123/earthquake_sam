"""Shared masked input construction for training and whole-tile validation."""
import h5py
import numpy as np
import torch
from scipy.ndimage import zoom


def load_tile(group):
    ctx = group['CTX'][:].astype(np.float32)
    depth = group['image'][:].astype(np.float32)
    tw = group['TW'][:].astype(np.float32)
    valid = group['valid'][:].astype(bool) if 'valid' in group else (
        (ctx > -2147483647)&(depth > -2147483647))
    valid &= np.isfinite(ctx)&np.isfinite(depth)&np.isfinite(tw).all(axis=0)
    if not valid.any():
        raise ValueError(f'No valid terrain in {group.name}')
    z = np.zeros_like(depth)
    z[valid] = (depth[valid]-depth[valid].mean())/(depth[valid].std()+1e-6)
    ctx = np.where(valid, ctx/127.5-1, 0)
    tw = np.where(valid[None], tw/127.5-1, 0)
    inputs = np.concatenate([ctx[None], z[None], valid[None], valid[None], tw]).astype(np.float32)
    labels = np.stack([group[k][:] if k in group else np.full(ctx.shape, 999)
                       for k in ('k0', 'k1', 'k2')]).astype(np.float32)
    labels[:, ~valid] = 999
    if not np.isfinite(labels).all():
        raise ValueError(f'Non-finite labels in {group.name}')
    return inputs, labels


class GridDataset(torch.utils.data.Dataset):
    def __init__(self, filename, key_list, m=256, n=256, training=False,
                 isOthers=False, fixed_crop=False, full=False):
        self.filename, self.list = str(filename), list(key_list)
        self.m, self.n, self.training, self.fixed_crop = m, n, training, fixed_crop
        self.full = full
        self.dataD = {}
        if not self.list:
            raise ValueError('Empty dataset')
        if training and (fixed_crop or full):
            raise ValueError('Training requires random crops')

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        key = self.list[index]
        if key not in self.dataD:
            with h5py.File(self.filename, 'r') as f:
                pair = load_tile(f[key])
            if len(self.dataD) < 64:
                self.dataD[key] = pair
        else:
            pair = self.dataD[key]
        x, y = (a.copy() for a in pair)
        if self.full:
            return x, y
        if self.training:
            s = np.random.uniform(.75, 1.25)
            # Distances remain in geographic degrees under image scaling.
            y = zoom(np.minimum(y, .35001), (1, s, s), order=1)
            masks = zoom(x[2:4], (1, s, s), order=0)
            x = zoom(x, (1, s, s), order=1)
            x[2:4] = masks
            v = masks.min(axis=0) > .5
            x[[0, 1, 4, 5, 6]] *= v
            y[:, ~v] = 999
        h, w = x.shape[-2:]
        if h < self.n or w < self.m:
            raise ValueError('Stored/scaled tile smaller than crop')
        v = x[2:4].min(axis=0) > .5
        if self.fixed_crop:
            i, j = (h-self.n)//2, (w-self.m)//2
        else:
            i, j = np.random.randint(h-self.n+1), np.random.randint(w-self.m+1)
        for attempt in range(32):
            if v[i:i+self.n, j:j+self.m].any():
                break
            if self.fixed_crop or attempt == 31:
                yy, xx = np.argwhere(v)[len(np.argwhere(v))//2]
                i, j = int(np.clip(yy-self.n//2, 0, h-self.n)), int(np.clip(xx-self.m//2, 0, w-self.m))
                break
            i, j = np.random.randint(h-self.n+1), np.random.randint(w-self.m+1)
        x, y = x[:, i:i+self.n, j:j+self.m], y[:, i:i+self.n, j:j+self.m]
        if self.training:
            k = np.random.randint(8)
            if k & 1: x, y = x[:, ::-1], y[:, ::-1]
            if k & 2: x, y = x[:, :, ::-1], y[:, :, ::-1]
            if k & 4: x, y = x.transpose(0, 2, 1), y.transpose(0, 2, 1)
        return np.ascontiguousarray(x), np.ascontiguousarray(y)


def target_weights(image, distances, d0, dw, alpha):
    d = distances[:, :1]
    valid = (image[:, 2:4] > .5).all(1, keepdim=True)
    weights = (torch.exp(-d.square()/dw**2)*(1-alpha)+alpha)*valid
    weights /= weights.mean().clamp_min(1e-12)
    return (d < d0).float(), weights


@torch.no_grad()
def sliding_predict(model, image, crop=256):
    h, w = image.shape[-2:]
    if min(h, w) < crop:
        raise ValueError('Validation tile smaller than crop')
    def starts(size):
        return sorted(set(range(0, size-crop+1, crop//2)) | {size-crop})
    result = torch.zeros((len(image), 1, h, w), device=image.device)
    count = torch.zeros_like(result)
    for i in starts(h):
        for j in starts(w):
            pred = model(image[..., i:i+crop, j:j+crop])[:, :1]
            result[..., i:i+crop, j:j+crop] += pred
            count[..., i:i+crop, j:j+crop] += 1
    return result/count
