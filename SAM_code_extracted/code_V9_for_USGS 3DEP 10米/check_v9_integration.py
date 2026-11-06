"""Bounded validation/preview/precision/state checks, without optimizer steps."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    output=p.parse_args().output
    output.mkdir(parents=True,exist_ok=False)
    from run_experiment import HERE,profile_arguments,validate_parser,command
    from torchUnet import conv_sam,AttnBlock
    from sam2_lora import training_mode
    from aio2_ema import activate_ema
    from aio2_online import fixed_validation
    from aio2_preview import FixedPreview
    from v7_dataset import FullTileGenerator
    from torch.utils.data import DataLoader
    torch.set_num_threads(4);torch.manual_seed(7)
    config=validate_parser(command(profile_arguments(HERE/'experiments/california_v9_30m_1024_8gb.json')))
    config['resDir']=str(output)
    args=SimpleNamespace(**config)
    model=conv_sam(32,isvit=False,adaptation=config,n=1024,m=1024).cuda()
    report=dict(formal_training_started=False,optimizer_steps=0)
    assert all(p.dtype==torch.float32 for p in model.parameters())
    report['parameter_dtype']='float32'
    for name,p in model.sam.named_parameters():
        assert p.requires_grad==('.lora_' in name)
    report['sam_base_frozen_lora_trainable']=True
    # Attribute-only execution options must not add/change parameter keys.
    with torch.no_grad():
        small=torch.randn(1,7,256,256,device='cuda')
        model.eval()
        base_keys=list(model.state_dict())
        optimized=model(small,return_logits=True)
        saved=dict(model.memory_options)
        model.memory_options.update(computePrecision='fp32',attentionBackend='legacy',poolFirst=False,activationCheckpoint='none')
        for mod in model.modules():
            if isinstance(mod,AttnBlock):mod.attention_backend='legacy'
        baseline=model(small,return_logits=True)
        model.memory_options=saved
        for mod in model.modules():
            if isinstance(mod,AttnBlock):mod.attention_backend=saved['attentionBackend']
        error=(optimized-baseline).abs()
        report['bf16_vs_fp32_logits']=dict(mean_absolute=float(error.mean()),max_absolute=float(error.max()),
            relative_rms=float(error.square().mean().sqrt()/baseline.square().mean().sqrt().clamp_min(1e-8)))
        assert torch.isfinite(optimized).all() and report['bf16_vs_fp32_logits']['relative_rms']<.1
        assert base_keys==list(model.state_dict())
        report['state_dict_keys_preserved']=True
        # Confirm strict fused CUDA dispatch actually executed, not just enabled.
        from torch.profiler import profile,ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
            model(small)
            torch.cuda.synchronize()
        report['flash_events']=[e.key for e in prof.key_averages() if 'flash_attention' in e.key]
        assert report['flash_events']
        del small,optimized,baseline,error
    assert activate_ema(model,None,249,args) is None
    teacher=activate_ema(model,None,250,args)
    assert teacher is not None and all(p.dtype==torch.float32 and not p.requires_grad for p in teacher.model.parameters())
    report['teacher_activation_249_off_250_on']=True
    data=Path(args.dataFile).parent
    rows=json.loads((data/'manifest.json').read_text())['tiles']
    selected={s:max((r for r in rows if r['split']==s),key=lambda r:r['known_pixels'])['key'] for s in ('train','val')}
    ds=FullTileGenerator(args.dataFile,[selected['val']])
    loader=DataLoader(ds,batch_size=1)
    training_mode(model)
    torch.cuda.reset_peak_memory_stats()
    student_losses,teacher_losses,health=fixed_validation(model,teacher,loader,'cuda',args,None,args.d0,args.dw,args.alpha,
                                                       epoch=350,return_metrics=True)
    report['validation']=dict(student_losses=student_losses,teacher_losses=teacher_losses,
                              peak_allocated_bytes=torch.cuda.max_memory_allocated())
    preview=FixedPreview(FullTileGenerator,args.dataFile,[selected['train']],output/'preview',1024,1024,args.d0,
                         count=1,dataset_options={'supervision_key':'label_known'})
    report['preview_report']=str(preview.write(model,teacher,350))
    assert model.training
    assert all(p.grad is None for p in teacher.model.parameters())
    report.update(status='PASS',teacher_gradients_absent=True,parameters_remain_fp32=True)
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
