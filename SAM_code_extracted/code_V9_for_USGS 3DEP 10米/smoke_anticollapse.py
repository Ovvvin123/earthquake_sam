"""Controlled small-data loss ablations. Never modifies source data or old runs.

Without a reviewed supervision mask, --catalog-background-ablation must be explicit:
such results diagnose optimization only, not geological accuracy.
"""
import argparse
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from sam2_lora import (add_adaptation_arguments, build_model, optimizer_groups,
                       training_mode, gradient_norms, load_run_model, write_run_metadata)
from fault_losses import forward_for_loss, supervised_loss, supervision, binary_fault_logit
from fault_metrics import FaultMetrics
from aio2_online import fixed_validation
from tool import Generator_new

ARMS = {'ft': ('FocalTversky', 1.), 'ft_low_lr': ('FocalTversky', .2),
        'bce': ('BCEWithLogits', 1.), 'mixed': ('BCEFocalTversky', 1.)}


def select_crops(folder, split, per_group, supervision_key):
    keys = (folder/f'key_{split}.lst').read_text().split()
    reader = Generator_new(str(folder/'faults.h5'), keys, training=False,
                           fixed_crop=True, supervision_key=supervision_key)
    buckets = {}
    for i, key in enumerate(keys):
        x, d = reader[i]
        known = (d[3] > .5) if supervision_key else np.ones_like(d[0], dtype=bool)
        if not known.any():
            continue
        group = key.split('__')[0] if '__' in key else ('positive' if ((d[0] < .01) & known).any() else 'empty')
        bucket = buckets.setdefault(group, [])
        if len(bucket) < per_group:
            bucket.append((key, x, d))
        if len(buckets) >= 2 and all(len(v) == per_group for v in buckets.values()):
            break
    items = [item for group in sorted(buckets) for item in buckets[group]]
    if len(items) < 2 or not any(((d[0] < .01) & ((d[3] > .5) if supervision_key else True)).any() for _, x, d in items):
        raise ValueError(f'{split}: insufficient supervised crops containing positives')
    return items


def batch(items, device):
    return (torch.from_numpy(np.stack([x for _, x, _ in items])).to(device),
            torch.from_numpy(np.stack([d for _, _, d in items])).to(device))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataDir', type=Path, required=True)
    ap.add_argument('--outDir', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=100, help='Optimizer updates per arm')
    ap.add_argument('--arms', nargs='+', choices=tuple(ARMS), default=list(ARMS))
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--lr0', type=float, default=1e-4)
    ap.add_argument('--supervisionKey', default='')
    ap.add_argument('--catalog-background-ablation', action='store_true')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--tverskyWeight', type=float, default=.5)
    add_adaptation_arguments(ap)
    ap.set_defaults(samAdaptation='lora')
    args = ap.parse_args()
    if not args.supervisionKey and not args.catalog_background_ablation:
        ap.error('Supply --supervisionKey, or explicitly use --catalog-background-ablation for a numerical diagnostic')
    if args.steps < 1 or args.steps > 1000 or len(set(args.arms)) != len(args.arms):
        ap.error('Use 1..1000 steps and unique arms')
    if args.outDir.exists() and any(args.outDir.iterdir()):
        ap.error('Use an empty output directory')
    if args.samAdaptation != 'lora':
        ap.error('This experiment checks the frozen-base LoRA configuration')
    args.outDir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    train = select_crops(args.dataDir, 'train', 4, args.supervisionKey)
    val = select_crops(args.dataDir, 'val', 2, args.supervisionKey)
    args.k0Only=True; args.swMode='single'; args.isOther=False
    args.d0=.01; args.dw=.08; args.alpha=.1; args.modelType='mars_sam'
    args.accumulationVersion='window_v2'; args.supervisionVersion='known_mask_v2'
    args.collapseGuard=True; args.consistencyLoss='probability'
    args.dataset=str(args.dataDir/'key_train.lst')
    train_batches=[batch(train[i:i+2], args.device) for i in range(0,len(train),2)]
    val_batches=[batch(val[i:i+2], args.device) for i in range(0,len(val),2)]
    report = dict(seed=args.seed, steps=args.steps, train_keys=[k for k,_,_ in train],
                  val_keys=[k for k,_,_ in val], arms={},
                  supervision='reviewed_mask' if args.supervisionKey else 'catalog_background_ablation',
                  note='Small fixed-crop optimization experiment, no augmentation or consistency. '
                       'Pass of numerical checks does not establish geological accuracy or generalization.')
    base_lr, base_lora_lr = args.lr0, args.loraLr
    from torchUnet import FocalTversky
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for arm in args.arms:
        torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
        args.lossFunc, factor = ARMS[arm]
        args.lr0, args.loraLr = base_lr*factor, base_lora_lr*factor
        folder=args.outDir/arm; folder.mkdir(); args.resDir=str(folder)+'/'
        net=build_model(vars(args)).to(args.device)
        frozen={n:p.detach().cpu().clone() for n,p in net.sam.named_parameters() if not p.requires_grad}
        adapters={n:p.detach().cpu().clone() for n,p in net.sam.named_parameters() if p.requires_grad}
        opt=torch.optim.AdamW(optimizer_groups(net,args))
        write_run_metadata(net,args,folder)
        start=time.perf_counter()
        initial=fixed_validation(net,None,val_batches,args.device,args,FocalTversky,.01,.08,.1,return_metrics=True)
        first_x, first_d=val_batches[0]
        net.eval()
        with torch.no_grad(): before=net(first_x).cpu()
        last_norms={}
        for step in range(args.steps):
            training_mode(net); opt.zero_grad(set_to_none=True)
            x,d=train_batches[step % len(train_batches)]
            y,w,_=supervision(x,d,args,.01,.08,.1)
            if not (w > 0).any():
                raise ValueError('Selected batch has no supervision')
            p,z=forward_for_loss(net,x,args)
            loss=supervised_loss(y,p,w,args,FocalTversky,z)
            if not torch.isfinite(loss): raise AssertionError('Nonfinite loss')
            loss.backward(); last_norms=gradient_norms(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(),.1,error_if_nonfinite=True)
            opt.step()
            meter=FaultMetrics(); meter.observe(p,y,w)
            row=dict(step=step+1,loss=float(loss.detach()),metrics=meter.report(),gradient_norms=last_norms)
            if z is not None:
                row['supervised_logit_quantiles_01_50_99']=torch.quantile(
                    binary_fault_logit(z.detach())[w > 0], torch.tensor([.01,.5,.99],device=z.device)).cpu().tolist()
            with (folder/'steps.jsonl').open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
            if (step+1)%20==0: print(arm,step+1,row['loss'],flush=True)
        final=fixed_validation(net,None,val_batches,args.device,args,FocalTversky,.01,.08,.1,return_metrics=True)
        for n,p in net.sam.named_parameters():
            if n in frozen:
                torch.testing.assert_close(p.cpu(),frozen[n],rtol=0,atol=0)
                if p.grad is not None: raise AssertionError('Frozen SAM has gradient')
        changed=sum(not torch.equal(p.detach().cpu(),adapters[n]) for n,p in net.sam.named_parameters() if n in adapters)
        if not changed: raise AssertionError('No LoRA adapter updated')
        net.eval()
        with torch.no_grad(): after=net(first_x).cpu()
        torch.save(net.state_dict(),folder/'model.pth')
        del net, frozen, adapters
        reloaded=load_run_model(folder/'model.pth',args.device)
        with torch.no_grad(): torch.testing.assert_close(after,reloaded(first_x).cpu(),rtol=1e-5,atol=1e-6)
        del reloaded
        fig,axes=plt.subplots(1,4,figsize=(13,3))
        views=[first_x[0,0].cpu(),first_d[0,0].cpu()<.01,before[0,0],after[0,0]]
        for ax,view,title in zip(axes,views,['Terrain','Catalog label','Before','After']):
            limits={} if title=='Terrain' else dict(vmin=0,vmax=1)
            ax.imshow(view,cmap='gray' if title=='Terrain' else 'magma',**limits); ax.set_title(title); ax.axis('off')
        fig.tight_layout();fig.savefig(folder/'comparison.png',dpi=130);plt.close(fig)
        result=dict(initial_val_loss=float(np.mean(initial[0])),final_val_loss=float(np.mean(final[0])),
                    initial_health=initial[2],final_health=final[2],frozen_base_unchanged=True,
                    changed_adapter_tensors=changed,strict_reload_passed=True,
                    elapsed_seconds=time.perf_counter()-start)
        report['arms'][arm]=result
        (args.outDir/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        print(arm,result['final_val_loss'],final[2]['student']['collapse'],flush=True)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    print('Report:',args.outDir/'report.json',flush=True)


if __name__ == '__main__': main()
