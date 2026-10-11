"""Read-only model/data diagnosis; isolated gradient probes do not update weights."""
import csv
import json
from pathlib import Path
import numpy as np
import h5py
import torch
from torchUnet import conv_sam, FocalTversky
from tool import Generator_new

HERE=Path(__file__).resolve().parent
RUN=HERE/'train_greece_nz_stride64_unimatch_d001'
OUT=RUN/'results/collapse_diagnosis.json'
torch.set_num_threads(4)


def log_stats(run):
    rows=list(csv.DictReader((run/'preview/epoch_0001/train_batches.csv').open()))
    result={'batches':len(rows),'empty_label_samples':sum(int(r['empty_label_samples']) for r in rows),
            'empty_label_batches':sum(int(r['original_positive_pixels'])==0 for r in rows),
            'positive_pixel_fraction':sum(int(r['original_positive_pixels']) for r in rows)/sum(int(r['valid_pixels']) for r in rows),
            'last_batch_prediction_mean':float(rows[-1]['prediction_mean'])}
    for level in [.9,.99,.999]:
        found=next((r for r in rows if float(r['prediction_mean'])>=level),None)
        result[f'first_batch_mean_ge_{level}']=int(found['batch']) if found else None
    return result


def data_stats(folder,split):
    keys=(folder/f'key_{split}.lst').read_text().split()
    positive=0;pixels=0;zero=0;fractions=[];weighted=[]
    with h5py.File(folder/'faults.h5','r') as f:
        for key in keys:
            d=f[key]['k0'][:]
            label=d<.01
            zero+=int(not label.any());positive+=int(label.sum());pixels+=label.size
            fractions.append(float(label.mean()))
            sw=np.exp(-d.astype('float64')**2/.08**2)*.9+.1
            weighted.append(float(sw[label].sum()/sw.sum()))
    return {'tiles':len(keys),'zero_label_tiles':zero,'positive_pixel_fraction':positive/pixels,
            'mean_weighted_positive_fraction':float(np.mean(weighted)),
            'tile_positive_fraction_quantiles':np.quantile(fractions,[0,.25,.5,.75,1]).tolist(),
            'scope':'full stored tiles; overlaps counted repeatedly'}


def gradient_probe(logits,target,sw,kind):
    z=logits.detach().clone().requires_grad_(True)
    if kind=='focal_tversky':
        loss=FocalTversky(target,z.softmax(1),sw=sw,pu=0)
    else:
        binary_logit=z[:,:1]-torch.logsumexp(z[:,1:],dim=1,keepdim=True)
        loss=(torch.nn.functional.binary_cross_entropy_with_logits(binary_logit,target,reduction='none')*sw).mean()
    grad=torch.autograd.grad(loss,z)[0][:,:1]
    pos=target.bool();neg=~pos
    return {'loss':float(loss.detach()),'positive_logit_grad_mean_abs':float(grad[pos].abs().mean()),
            'background_logit_grad_mean_abs':float(grad[neg].abs().mean()),
            'background_exact_zero_gradient_fraction':float((grad[neg]==0).float().mean())}


def main():
    report={'current_epoch1':log_stats(RUN),'old_epoch1':log_stats(HERE/'train_v5_unimatch_d001')}
    new=HERE/'earth_dataset_greece_nz_stride64_v2'
    report['datasets']={name:{s:data_stats(folder,s) for s in ['train','val']} for name,folder in
                        [('new',new),('old',HERE/'earth_dataset_v3')]}
    online=[json.loads(p.read_text()) for p in sorted((RUN/'online').glob('epoch_*/summary.json'))]
    report['online']={'epochs':len(online),'candidate_pixels_total':sum(x['totals']['candidate_pixels'] for x in online),
                      'accepted_components_total':sum(x['totals']['accepted_components'] for x in online)}
    report['unimatch_final']=json.loads((RUN/'unimatch/epoch_0200.json').read_text())
    report['learning_rates']={'head_epochs_2_3_5_9':[1e-4,5e-5,2.5e-5,1.25e-5],
                              'note':'same code in both experiments; not an identified new bug'}
    keys=(new/'key_train.lst').read_text().split()
    selected=[next(k for k in keys if k.startswith(r)) for r in ['greece_aegean','new_zealand']]
    reader=Generator_new(str(new/'faults.h5'),selected,n=256,m=256,training=False,fixed_crop=True)
    items=[reader[i] for i in range(2)]
    images=torch.tensor(np.stack([x[0] for x in items]))
    d=torch.tensor(np.stack([x[1][:1] for x in items]))
    target=(d<.01).float();sw=torch.exp(-d.square()/.08**2)*.9+.1;sw=sw/sw.mean()
    assert target.any() and (~target.bool()).any()
    report['uniform_predictions']={str(p):float(FocalTversky(target,torch.full_like(target,p),sw=sw,pu=0))
                                   for p in [.1,.25,.5,.9,.99,1.]}
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    from sam2_lora import load_run_model
    net=load_run_model(RUN/'best_student/model.pth', device)
    net.eval();captured=[]
    handle=net.tail[-2].register_forward_hook(lambda m,i,o:captured.append(o.detach().cpu()))
    with torch.no_grad(): prediction=net(images.to(device)).cpu()
    handle.remove()
    logits=captured[0]
    report['probe']={'keys':selected,'label_positive_fraction':float(target.mean()),
                     'predicted_probability_mean':float(prediction[:,0].mean()),
                     'logit_margin_quantiles':np.quantile((logits[:,:1]-torch.logsumexp(logits[:,1:],1,keepdim=True)).numpy(),[0,.5,.99,1]).tolist(),
                     'saturated_tversky':gradient_probe(logits,target,sw,'focal_tversky'),
                     'unsaturated_tversky':gradient_probe(torch.zeros_like(logits),target,sw,'focal_tversky'),
                     'saturated_bce_with_logits':gradient_probe(logits,target,sw,'bce')}
    report['note']='No optimizer step or model/data edits. BCE is an isolated diagnostic probe, not a validated replacement experiment.'
    OUT.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
