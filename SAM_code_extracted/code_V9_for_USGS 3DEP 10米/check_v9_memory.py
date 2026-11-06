"""Bounded V7 memory check: ephemeral parameters only, no training checkpoint.

Run each epoch scenario in a fresh process. Uses production loss, LoRA,
EMA, pseudo-target and strong-view backward code without importing trainV10.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace


def load_probe_profile(root, profile_path=None, checkpoint=None):
    from run_experiment import profile_arguments
    path = profile_path or root/'experiments/california_v9_30m_1024_8gb.json'
    path = Path(path)
    if not path.is_absolute():
        path = root/path
    profile = profile_arguments(path)
    if checkpoint is not None:
        profile['activationCheckpoint'] = checkpoint
    return profile


def probe_batch(reader, start, batch_size):
    """Stack independently augmented representative tiles into a real batch."""
    import numpy as np
    if batch_size < 1 or len(reader) < 1:
        raise ValueError('A positive batch size and nonempty reader are required')
    items = [reader[(start+i) % len(reader)] for i in range(batch_size)]
    return np.stack([x for x, _ in items]), np.stack([y for _, y in items])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/home/ovvvin/projects/earthquake-sam-v9'))
    parser.add_argument('--profile', type=Path, help='Use the resolved cloud profile from a successful preflight')
    parser.add_argument('--epoch', type=int, choices=(249,250,350), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', choices=('none','decoder','all'),default=None)
    parser.add_argument('--microbatches',type=int,default=16)
    cli = parser.parse_args()
    if cli.microbatches < 1:
        parser.error('--microbatches must be positive')
    root = cli.root.resolve()
    sys.path.insert(0,str(root))
    import numpy as np
    import torch
    from run_experiment import command, validate_parser
    from training_lr import learning_rate_factors
    from training_sources import training_source_snapshot
    from sam2_lora import optimizer_groups, training_mode, set_learning_rates, gradient_norms
    from torchUnet import conv_sam, AttnBlock
    from v7_dataset import FullTileGenerator
    from fault_losses import supervision, forward_for_loss, supervised_loss, accumulation_window
    from training_numerics import require_finite, clip_finite_gradients
    from aio2_ema import EMATeacher
    from aio2_online import temporary_targets, pseudo_weight
    from unimatch_fault import backward_strong_views, consistency_weight

    cli.output.mkdir(parents=True,exist_ok=False)
    profile = load_probe_profile(root, cli.profile, cli.checkpoint)
    config = validate_parser(command(profile))
    args = SimpleNamespace(**config)
    assert args.fullTile and args.mn == '1024'
    assert args.batchsize >= 1 and args.stepPerMiniBatch >= 1
    assert args.emaStartEpoch == 250 and args.unimatchRampEnd == 350
    torch.set_num_threads(4)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this capacity check')
    device = torch.device(f'cuda:{args.gpuIndex}')
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    report = dict(status='RUNNING', epoch_scenario=cli.epoch, config=config,
                  formal_training_started=False, checkpoints_written=False,
                  precision='FP32 parameters and EMA; BF16 autocast, strict FlashAttention, pool-first',
                  gpu=torch.cuda.get_device_name(device), torch_version=torch.__version__,
                  source_sha256=training_source_snapshot(root),
                  effective_batch_size=args.batchsize*args.stepPerMiniBatch,
                  sample_strategy='Cycle independently augmented representative positive tiles; capacity check only',
                  intended_optimizer_steps=(cli.microbatches+args.stepPerMiniBatch-1)//args.stepPerMiniBatch,
                  completed_optimizer_steps=0,
                  completed_microbatches=0, weak_backwards=0,strong_backward_pairs=0,
                  teacher_updates=0, trace=[])
    report['probe_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    monitor_stream = (cli.output/'gpu_usage.csv').open('w')
    monitor = subprocess.Popen(['nvidia-smi',
        '--query-gpu=timestamp,memory.total,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits','--loop-ms=200'],stdout=monitor_stream,stderr=subprocess.DEVNULL)
    free,total = torch.cuda.mem_get_info(device)
    report.update(initial_free_bytes=free,total_device_bytes=total)
    started = time.perf_counter()
    phase = 'initialization'

    def mark(name):
        nonlocal phase
        phase = name
        item = dict(phase=name, seconds=time.perf_counter()-started,
                    allocated_bytes=torch.cuda.memory_allocated(device),
                    reserved_bytes=torch.cuda.memory_reserved(device),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        report['trace'].append(item)
        print(json.dumps(item),flush=True)

    def attention_hook(name):
        def hook(module, inputs):
            shape = list(inputs[0].shape)
            batch,channels,height,width = shape
            report['last_attention_entered'] = dict(module=name,input_shape=shape,
                dense_score_bytes=batch*module.head_num*(height*width)**2*inputs[0].element_size(),
                allocated_bytes=torch.cuda.memory_allocated(device))
        return hook

    try:
        data_dir = Path(args.dataFile).parent
        manifest = json.loads((data_dir/'manifest.json').read_text())
        rows = [r for r in manifest['tiles'] if r['split']=='train' and r['positive_pixels']>0]
        chosen = [max(rows,key=lambda r:r['known_pixels']),max(rows,key=lambda r:r['unknown_valid_pixels'])]
        keys = list(dict.fromkeys(r['key'] for r in chosen))
        report['samples'] = [{k:r[k] for k in ('key','valid_pixels','known_pixels','positive_pixels','unknown_valid_pixels')} for r in chosen]
        reader = FullTileGenerator(args.dataFile,keys,training=True)
        mark('student_initialization')
        student = conv_sam(32,dropout=args.dropout,isvit=False,isalign=args.isalign,
                           adaptation=config,n=1024,m=1024).to(device)
        opt = torch.optim.AdamW(optimizer_groups(student,args),lr=args.lr0)
        for name,module in student.named_modules():
            if isinstance(module,AttnBlock):
                module.register_forward_pre_hook(attention_hook('student.'+name))
        mark('teacher_initialization' if cli.epoch>=args.emaStartEpoch else 'teacher_disabled')
        teacher = EMATeacher(student,args.emaDecay) if cli.epoch>=args.emaStartEpoch else None
        report['teacher_active'] = teacher is not None
        if teacher is not None:
            # deepcopy also copies debug hooks: replace them so reports name the teacher.
            for name,module in teacher.model.named_modules():
                if isinstance(module,AttnBlock):
                    module._forward_pre_hooks.clear()
                    module.register_forward_pre_hook(attention_hook('teacher.'+name))
        training_mode(student)
        opt.zero_grad(set_to_none=True)
        online_weight = pseudo_weight(cli.epoch,args)
        um_weight = consistency_weight(cli.epoch,args)
        report.update(pseudo_weight=online_weight,consistency_weight=um_weight)
        count = cli.microbatches
        for j in range(count):
            head_factor, sam_factor = learning_rate_factors(config, cli.epoch-1, j, count)
            set_learning_rates(opt, args, head_factor, sam_factor)
            report['attempted_microbatch'] = j+1
            mark(f'batch_{j+1}_input')
            x,y = probe_batch(reader, j*args.batchsize, args.batchsize)
            image = torch.from_numpy(x).to(device)
            distances = torch.from_numpy(y).to(device)
            report['last_input_shape'] = list(image.shape)
            assert image.shape[0] == args.batchsize
            target,sw,valid = supervision(image,distances,args,args.d0,args.dw,args.alpha)
            del distances
            assert int(target.sum())>0, 'Strong views must have genuine positive supervision'
            training_target,training_sw = target,sw
            pseudo_mask = torch.zeros_like(target,dtype=torch.bool)
            divisor,should_step = accumulation_window(j,count,args.stepPerMiniBatch,args.accumulationVersion)
            if teacher is not None:
                mark(f'batch_{j+1}_teacher_forward')
                teacher_probabilities = teacher.predict(image)
                torch.cuda.synchronize(device)
                require_finite(teacher_probabilities,'teacher prediction',cli.output)
                training_target,training_sw,pseudo_mask,candidate_stats = temporary_targets(
                    target,sw,teacher_probabilities,image,online_weight,args)
                report['last_pseudo_pixels'] = int(pseudo_mask.sum())
            mark(f'batch_{j+1}_student_weak_forward')
            predict,logits = forward_for_loss(student,image,args)
            require_finite(predict,'weak prediction',cli.output)
            loss = supervised_loss(training_target,predict,training_sw,args,None,logits,0.)
            require_finite(loss,'weak loss',cli.output)
            report['last_supervised_loss'] = float(loss.detach())
            if teacher is not None:
                del teacher_probabilities
            mark(f'batch_{j+1}_student_weak_backward')
            (loss/divisor).backward()
            torch.cuda.synchronize(device)
            report['weak_backwards'] += 1
            del predict,logits,loss
            mark(f'batch_{j+1}_strong_views_forward_backward')
            stats,sample = backward_strong_views(student,image,target,pseudo_mask,online_weight,args,
                um_weight,divisor,cli.output,cli.epoch,j,capture=(j==0 and teacher is not None))
            torch.cuda.synchronize(device)
            if um_weight>0:
                assert stats['active']==1, 'Consistency branch was unexpectedly skipped'
                report['strong_backward_pairs'] += 1
            report['last_strong_stats'] = stats
            if should_step:
                mark(f'batch_{j+1}_optimizer_step')
                report['gradient_norms'] = gradient_norms(opt)
                assert report['gradient_norms'].get('lora',0)>0, 'LoRA gradient disconnected'
                assert report['gradient_norms'].get('head',0)>0, 'Head gradient disconnected'
                clip_finite_gradients(student,.1,cli.output,epoch=cli.epoch,batch=j)
                opt.step()
                report['completed_optimizer_steps'] += 1
                if teacher is not None:
                    mark(f'batch_{j+1}_teacher_update')
                    teacher.update(student)
                    report['teacher_updates'] = teacher.num_updates
                opt.zero_grad()
                torch.cuda.synchronize(device)
            report['completed_microbatches'] += 1
            # Production retains last weak outputs until the next batch; keep the
            # same references here. Strong branches free their own saved tensors.
        mark('complete')
        report['status'] = 'PASS_BOUNDED_CHECK'
    except torch.cuda.OutOfMemoryError as error:
        report.update(status='OUT_OF_MEMORY',failed_phase=phase,error=str(error),
                      traceback=traceback.format_exc())
    except Exception as error:
        report.update(status='ERROR',failed_phase=phase,error=str(error),traceback=traceback.format_exc())
    finally:
        monitor.terminate()
        monitor.wait(timeout=10)
        monitor_stream.close()
        readings = []
        for line in (cli.output/'gpu_usage.csv').read_text().splitlines():
            parts = line.split(',')
            try:
                readings.append(dict(total_mib=float(parts[1]),used_mib=float(parts[2]),free_mib=float(parts[3])))
            except (IndexError,ValueError):
                pass
        if readings:
            report['nvidia_smi'] = dict(samples=len(readings),total_mib=readings[0]['total_mib'],
                peak_used_mib=max(r['used_mib'] for r in readings),min_free_mib=min(r['free_mib'] for r in readings))
        report.update(elapsed_seconds=time.perf_counter()-started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        report['final_free_bytes'],_ = torch.cuda.mem_get_info(device)
        report['allocator_peak_exceeds_device_capacity'] = report['peak_allocated_bytes'] > total
        report['memory_accounting_note'] = ('PyTorch allocated/reserved bytes describe this process allocator; '
            'nvidia-smi reports device-wide usage including other processes.')
        (cli.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        (cli.output/'memory_summary.txt').write_text(torch.cuda.memory_summary(device=device),encoding='utf-8')
        print(json.dumps({k:report.get(k) for k in ('status','epoch_scenario','failed_phase','error',
            'last_input_shape','effective_batch_size',
            'peak_allocated_bytes','peak_reserved_bytes','completed_microbatches','completed_optimizer_steps',
            'weak_backwards','strong_backward_pairs','teacher_updates')}),flush=True)
    if report['status']!='PASS_BOUNDED_CHECK':
        raise RuntimeError(report.get('error', report['status']))


if __name__=='__main__':
    main()
