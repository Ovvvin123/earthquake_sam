"""Isolated synchronized runtime experiment; never resumes a formal run."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('/home/ovvvin/projects/earthquake-sam-v9'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--epoch', type=int, choices=[249, 350], required=True)
    cli = ap.parse_args()
    root = cli.root.resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    from types import SimpleNamespace
    from torch.utils.data import DataLoader
    from tool import collate_function
    from run_experiment import profile_arguments, command, validate_parser
    from training_sources import training_source_snapshot
    from torchUnet import conv_sam
    from v7_dataset import FullTileGenerator
    from sam2_lora import optimizer_groups, training_mode, set_learning_rates, gradient_norms, checkpoint_identity
    from training_lr import learning_rate_factors
    from fault_losses import supervision, forward_for_loss, supervised_loss, accumulation_window
    from fault_metrics import FaultMetrics
    from training_numerics import require_finite, clip_finite_gradients
    from aio2_ema import EMATeacher, save_ema_checkpoint
    from aio2_online import temporary_targets, pseudo_weight, fixed_validation, OnlineDiagnostics
    from aio2_preview import BatchDiagnostics, FixedPreview
    from unimatch_fault import backward_strong_views, consistency_weight

    cli.output.mkdir(parents=True, exist_ok=False)
    profile = profile_arguments(root / 'experiments/california_v9_30m_1024_8gb_600ep.json')
    profile.update(resDir=str(cli.output), backupDir='')
    config = validate_parser(command(profile))
    args = SimpleNamespace(**config)
    assert args.d0 == .005 and args.batchsize == 1 and args.stepPerMiniBatch == 8
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    phases = defaultdict(list)
    enabled = False

    @contextmanager
    def timed(name):
        if not enabled:
            yield
            return
        torch.cuda.synchronize()
        begin = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            phases[name].append(time.perf_counter() - begin)

    report = dict(status='RUNNING', epoch_scenario=cli.epoch, config=config,
                  formal_training_started=False, previous_run_untouched=True,
                  initialization='fresh base SAM plus seeded head/LoRA; not a trained epoch-350 model',
                  source_sha256=training_source_snapshot(root),
                  probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  methodology='8 warmup + 32 distinct measured tiles; synchronized phase wall times; 16 untimed-boundary control tiles',
                  caveats=['Synchronization can affect overlap; control loop estimates instrumentation overhead.',
                           'Pseudo-label cost depends on predictions. This is a timing simulation, not an actual epoch-350 state.',
                           'Filesystem caches are warm; no OS cache flushing. No Windows backup copy is benchmarked.'])
    monitor_file = (cli.output / 'gpu.csv').open('w')
    monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu',
                                '--format=csv,noheader,nounits', '--loop-ms=1000'], stdout=monitor_file,
                               stderr=subprocess.DEVNULL)
    try:
        begin = time.perf_counter()
        student = conv_sam(32, dropout=args.dropout, isvit=False, isalign=args.isalign,
                           adaptation=config, n=1024, m=1024).to(device)
        opt = torch.optim.AdamW(optimizer_groups(student, args), lr=args.lr0)
        teacher = EMATeacher(student, args.emaDecay) if cli.epoch >= args.emaStartEpoch else None
        training_mode(student)
        torch.cuda.synchronize()
        report['model_setup_seconds'] = time.perf_counter() - begin
        keys = Path(args.dataset).read_text().split()
        chosen = random.Random(args.seed).sample(keys, 40)
        report['warmup_keys'], report['measured_keys'] = chosen[:8], chosen[8:]
        measured = chosen[8:]
        online_weight, um_weight = pseudo_weight(cli.epoch, args), consistency_weight(cli.epoch, args)
        report.update(teacher_active=teacher is not None, pseudo_weight=online_weight, consistency_weight=um_weight)
        meters, diag = FaultMetrics(), BatchDiagnostics(args.previewThreshold)
        online = OnlineDiagnostics(cli.epoch, online_weight, cli.output, capture=False)
        active = 0
        candidates = 0
        steps = 0

        def run_block(block_keys, phase_timing):
            nonlocal enabled, active, candidates, steps, meters, diag
            enabled = phase_timing
            meters, diag = FaultMetrics(), BatchDiagnostics(args.previewThreshold)
            reader = FullTileGenerator(args.dataFile, block_keys, training=True)
            loader = DataLoader(reader, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_function)
            iterator = iter(loader)
            training_mode(student)
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            has_signal = False
            for j in range(len(block_keys)):
                if j % 8 == 0:
                    has_signal = False
                divisor, should_step = accumulation_window(j, len(block_keys), 8, args.accumulationVersion)
                set_learning_rates(opt, args, *learning_rate_factors(args, cli.epoch-1, j, len(block_keys)))
                with timed('data_read_and_cpu_preprocess'):
                    x, y = next(iterator)
                with timed('host_to_gpu'):
                    image, distances = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
                with timed('supervision_and_input_checks'):
                    require_finite(image, 'input', cli.output)
                    require_finite(distances, 'distances', cli.output)
                    target, sw, valid = supervision(image, distances, args, args.d0, args.dw, args.alpha)
                    del distances
                    training_target, training_sw = target, sw
                    pseudo_mask = torch.zeros_like(target, dtype=torch.bool)
                if teacher is not None:
                    with timed('teacher_forward'):
                        teacher_prob = teacher.predict(image)
                        require_finite(teacher_prob, 'teacher', cli.output)
                    with timed('pseudo_label_cpu_filter_and_targets'):
                        training_target, training_sw, pseudo_mask, candidate_stats = temporary_targets(
                            target, sw, teacher_prob, image, online_weight, args)
                        candidates += int(pseudo_mask.sum())
                with timed('student_weak_forward'):
                    predict, logits = forward_for_loss(student, image, args)
                    require_finite(predict, 'prediction', cli.output)
                with timed('supervised_loss'):
                    require_finite(training_sw, 'weights', cli.output)
                    loss = supervised_loss(training_target, predict, training_sw, args, None, logits, 0.)
                    require_finite(loss, 'loss', cli.output)
                    has_signal |= bool((training_sw > 0).any())
                with timed('metrics_and_cpu_preview_statistics'):
                    valid_prob = predict[:, 0].detach()[image[:, 2:4].amin(1) > .5]
                    float(valid_prob.sum()); int((valid_prob >= .5).sum())
                    meters.observe(predict, target, sw)
                    diag.observe(j, target, predict, image, loss.item())
                    if teacher is not None:
                        online.observe(j, target, pseudo_mask, teacher_prob, image, candidate_stats, loss.item())
                        del teacher_prob
                with timed('student_weak_backward'):
                    (loss / divisor).backward()
                del predict, logits, loss, valid_prob
                if teacher is not None:
                    with timed('unimatch_two_strong_forward_backward'):
                        stats, _ = backward_strong_views(student, image, target, pseudo_mask, online_weight,
                            args, um_weight, divisor, cli.output, cli.epoch, j, capture=False)
                        active += stats['active']
                        has_signal |= bool(stats['active'])
                if should_step:
                    with timed('optimizer_clip_and_step'):
                        if has_signal:
                            gradient_norms(opt)
                            clip_finite_gradients(student, .1, cli.output, epoch=cli.epoch, batch=j)
                            opt.step()
                            steps += 1
                        opt.zero_grad(set_to_none=True)
                    if teacher is not None and has_signal:
                        with timed('ema_parameter_update'):
                            teacher.update(student)
                if (j+1) % 8 == 0:
                    print(json.dumps(dict(epoch=cli.epoch, timing=phase_timing, batches=j+1,
                                          total=len(block_keys))), flush=True)
            torch.cuda.synchronize()
            return time.perf_counter() - started

        report['warmup_seconds'] = run_block(chosen[:8], False)
        active = candidates = steps = 0
        torch.cuda.reset_peak_memory_stats()
        report['measured_wall_seconds'] = run_block(measured, True)
        report.update(measured_microbatches=32, active_strong_pairs=active,
                      pseudo_pixels_total=candidates, optimizer_steps=steps,
                      supervised_samples=meters.samples,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        report['phase_seconds'] = {k: dict(total=sum(v), calls=len(v), mean=statistics.mean(v),
                                           median=statistics.median(v)) for k, v in phases.items()}
        report['control_16_wall_seconds'] = run_block(measured[:16], False)
        report['instrumented_to_control_per_batch_ratio'] = (
            report['measured_wall_seconds']/32 / (report['control_16_wall_seconds']/16))
        enabled = False
        valkeys = (Path(args.dataFile).parent / 'key_val.lst').read_text().split()
        valreader = FullTileGenerator(args.dataFile, valkeys, training=False)
        valloader = DataLoader(valreader, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_function)
        torch.cuda.synchronize(); begin = time.perf_counter()
        fixed_validation(student, teacher, valloader, device, args, None, args.d0, args.dw, args.alpha,
                         epoch=cli.epoch, return_metrics=True)
        torch.cuda.synchronize()
        report['validation_40_seconds'] = time.perf_counter()-begin
        print(json.dumps(dict(validation_seconds=report['validation_40_seconds'])), flush=True)
        begin = time.perf_counter()
        checkpoint_identity(config)
        report['one_time_identity_hash_seconds'] = time.perf_counter()-begin
        begin = time.perf_counter()
        scratch = cli.output / 'timing_only_checkpoint.pth'
        save_ema_checkpoint(scratch, student, opt, teacher, cli.epoch, config, global_step=steps)
        report['checkpoint_save_seconds'] = time.perf_counter()-begin
        report['checkpoint_bytes'] = scratch.stat().st_size
        # Only this experiment-owned scratch file is removed, never an existing training checkpoint.
        scratch.unlink()
        begin = time.perf_counter()
        preview = FixedPreview(FullTileGenerator, args.dataFile, keys, cli.output/'preview', 1024, 1024,
                               args.d0, count=6, dataset_options=dict(supervision_key='label_known'))
        report['one_time_preview_setup_seconds'] = time.perf_counter()-begin
        torch.cuda.synchronize(); begin = time.perf_counter()
        preview.write(student, teacher, cli.epoch)
        torch.cuda.synchronize()
        report['six_image_preview_write_seconds'] = time.perf_counter()-begin
        report['estimated_epoch_seconds_excluding_windows_backup'] = (
            report['control_16_wall_seconds']/16*174 + report['validation_40_seconds'] +
            report['checkpoint_save_seconds'] + report['six_image_preview_write_seconds']/5)
        report['status'] = 'PASS'
    except BaseException:
        import traceback
        report.update(status='ERROR', traceback=traceback.format_exc())
        raise
    finally:
        monitor.terminate(); monitor.wait(timeout=10); monitor_file.close()
        readings = []
        for line in (cli.output/'gpu.csv').read_text().splitlines():
            try:
                _, memory, utilization, temperature = line.split(',')
                readings.append((float(memory), float(utilization), float(temperature)))
            except ValueError:
                pass
        if readings:
            report['gpu_samples'] = dict(peak_memory_mib=max(r[0] for r in readings),
                                         peak_temperature_c=max(r[2] for r in readings), count=len(readings))
        (cli.output/'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(dict(status=report['status'], output=str(cli.output))), flush=True)


if __name__ == '__main__':
    main()
