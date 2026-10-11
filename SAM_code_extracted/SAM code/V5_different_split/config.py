import argparse
import numpy as np
parser = argparse.ArgumentParser(description='transformer inversion')
parser.add_argument('--job', default='train', type=str, help='batch size')
parser.add_argument('--dataMode', default='both', type=str, help='batch size')
parser.add_argument('--gpuIndex', default=0, type=int, help='batch size')
parser.add_argument('--batchsize', default=32, type=int, help='batch size')
parser.add_argument('--transN', default=1, type=int, help='batch size')
parser.add_argument('--lr0', default=1e-4, type=float, help='batch size')
parser.add_argument('--resDir', default='train/', type=str, help='batch size')
parser.add_argument('--mn', default='256', type=str, help='batch size')
parser.add_argument('--isAtten', default=False, help='batch size', action='store_true')
parser.add_argument('--alpha', default=0.1, type=float, help='batch size')
parser.add_argument('--dw', default=0.15, type=float, help='batch size')
parser.add_argument('--d0', default=0.05, type=float, help='batch size')
parser.add_argument('--modelType', default='mars_sam', type=str, help='batch size')
parser.add_argument('--stepPerMiniBatch', default=8, type=int, help='batch size')
parser.add_argument('--dropout', default=0, type=float, help='batch size')
parser.add_argument('--threshold', default=0.5, type=float, help='batch size')
parser.add_argument('--dataset', default='key.lst', type=str, help='batch size')
parser.add_argument('--datasetAdd', default='', type=str, help='batch size')
parser.add_argument('--faultFile', default='data/20240311_f/tectonics_Compression.shp', type=str, help='batch size')
parser.add_argument('--valueFile', default='/H5/all_eqA_296_predict_feature.h5', type=str, help='batch size')
parser.add_argument('--value', default='feature', type=str, help='batch size')
parser.add_argument('--isalign', default=False, help='batch size', action='store_true')
parser.add_argument('--isOther', default=False, help='batch size', action='store_true')
parser.add_argument('--swMode', default='single', type=str, help='batch size')
parser.add_argument('--lossFunc', default='Dice', type=str, help='Dice | IoU | FocalTversky')
parser.add_argument('--pu', default=0.0, type=float, help='PU discount strength 0..1 for unlabeled-but-confident pixels (FocalTversky only, 0=off)')
parser.add_argument('--puWarmup', default=20, type=int, help='epochs of plain training before PU discount starts')
parser.add_argument('--puRamp', default=20, type=int, help='epochs to linearly ramp PU discount to full strength')
parser.add_argument('--k0Only', default=False, help='train loss on k0 (any-fault) channel only', action='store_true')
parser.add_argument('--epochs', default=160, type=int, help='training epochs')
parser.add_argument('--emaTeacher', action='store_true', help='enable EMA teacher; correction is separately opt-in')
parser.add_argument('--emaDecay', default=0.99, type=float, help='EMA decay per optimizer update, in [0, 1)')
parser.add_argument('--emaResume', default='', type=str, help='resume an EMA run from ema_checkpoint.pth; requires --emaTeacher')

parser.add_argument('--emaPreview', action='store_true', help='fixed-crop previews and batch diagnostics; no label correction')
parser.add_argument('--previewCount', type=int, default=6, help='number of fixed training crops')
parser.add_argument('--previewEvery', type=int, default=1, help='write preview images every N epochs')
parser.add_argument('--previewThreshold', type=float, default=0.7, help='candidate probability threshold, for preview only')
parser.add_argument('--previewBuffer', type=int, default=2, help='exclude pixels within this buffer of original labels')
parser.add_argument('--previewMinPixels', type=int, default=8, help='minimum 8-connected candidate area in pixels')

# Online correction is opt-in; the PU implementation/default remains unchanged.
parser.add_argument('--onlineCorrection', action='store_true', help='temporary online k0 labels (fixed trigger, not full ACT/O2C)')
parser.add_argument('--onlineStart', type=int, default=41)
parser.add_argument('--onlineHoldEnd', type=int, default=50)
parser.add_argument('--onlineRampEnd', type=int, default=80)
parser.add_argument('--onlineWeightStart', type=float, default=0.05)
parser.add_argument('--onlineWeightMax', type=float, default=0.2)
parser.add_argument('--onlineThreshold', type=float, default=0.9, help='high-confidence seed threshold')
parser.add_argument('--onlineSupport', type=float, default=0.5, help='component connectivity threshold')
parser.add_argument('--onlineBuffer', type=int, default=3)
parser.add_argument('--onlineMinPixels', type=int, default=16)
parser.add_argument('--onlineMaxFraction', type=float, default=0.05, help='maximum candidate area / valid crop area')
parser.add_argument('--fixedVal', action='store_true', help='fixed center crop per validation key; per-sample loss and best checkpoints')
parser.add_argument('--checkpointEvery', type=int, default=20, help='full stage checkpoints with --fixedVal; 0 disables')
parser.add_argument('--emaStageTransition', action='store_true', help='explicitly fork a warm-up checkpoint into a new fixed-validation run')

# Fault-specific UniMatch V2 adaptation; old commands remain opt-out.
parser.add_argument('--unimatch', action='store_true', help='SAM-2 dual strong views with positive-only consistency; requires online correction')
parser.add_argument('--unimatchStart', type=int, default=41, help='first consistency epoch, one-based')
parser.add_argument('--unimatchRampEnd', type=int, default=80, help='epoch at which consistency reaches maximum weight')
parser.add_argument('--unimatchWeight', type=float, default=0.1, help='maximum auxiliary consistency loss coefficient')
parser.add_argument('--unimatchNoDropout', action='store_true', help='ablation: dual strong images without complementary feature dropout')
parser.add_argument('--unimatchContrast', type=float, default=0.2, help='shared display-channel contrast range around 1')
parser.add_argument('--unimatchBrightness', type=float, default=0.1, help='display-channel brightness shift in normalized units')
parser.add_argument('--unimatchNoise', type=float, default=0.03, help='shared display-channel Gaussian noise standard deviation')

parser.add_argument('--maxSteps', type=int, default=3200)
parser.add_argument('--warmupSteps', type=int, default=20)
parser.add_argument('--onlineStartStep', type=int, default=800)
parser.add_argument('--onlineHoldStep', type=int, default=1000)
parser.add_argument('--onlineRampStep', type=int, default=1600)
parser.add_argument('--unimatchStartStep', type=int, default=800)
parser.add_argument('--unimatchRampStep', type=int, default=1600)
parser.add_argument('--puWarmupSteps', type=int, default=400)
parser.add_argument('--puRampSteps', type=int, default=400)
parser.add_argument('--evalEverySteps', type=int, default=100)
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--device', default='auto')
parser.add_argument('--resume', default='', help='grid_checkpoint.pth only; resumes optimizer and schedule')
from sam2_lora import add_adaptation_arguments, adaptation_config
add_adaptation_arguments(parser)
args = parser.parse_args()
try:
    adaptation_config(args)
except ValueError as error:
    parser.error(str(error))
if args.samAdaptation != 'full':
    if args.modelType != 'mars_sam' or args.isalign:
        parser.error('Frozen/LoRA requires mars_sam without isalign')
    args.fixedVal = True

if args.maxSteps < 1 or args.evalEverySteps < 1 or args.batchsize < 1 or args.stepPerMiniBatch < 1:
    parser.error('Step budget, evaluation interval, batch size and accumulation must be positive')
if not (0 <= args.onlineStartStep <= args.onlineHoldStep < args.onlineRampStep):
    parser.error('Require 0 <= onlineStartStep <= onlineHoldStep < onlineRampStep')
if not (args.onlineStartStep <= args.unimatchStartStep <= args.unimatchRampStep):
    parser.error('Require onlineStartStep <= unimatchStartStep <= unimatchRampStep')
if args.warmupSteps < 0 or args.puWarmupSteps < 0 or args.puRampSteps < 1:
    parser.error('Invalid warmup/ramp steps')

if args.unimatch:
    from pathlib import Path
    import os
    args.resDir = os.path.join(args.resDir, '')
    if not (args.emaResume or args.resume) and any((Path(args.resDir) / name).exists() for name in
                                  ('ema_checkpoint.pth', 'model.pth', 'best_metrics.json')):
        parser.error('UniMatch output already contains a run; use --emaResume or a new --resDir')
    if not (args.onlineCorrection and args.emaTeacher and args.k0Only
            and args.lossFunc == 'FocalTversky' and args.job == 'train'):
        parser.error('--unimatch requires --onlineCorrection, --emaTeacher, --k0Only and FocalTversky training')
    if args.modelType != 'mars_sam' or args.isalign:
        parser.error('This UniMatch adaptation supports --modelType mars_sam without --isalign')
    if not (args.onlineStart <= args.unimatchStart <= args.unimatchRampEnd):
        parser.error('Require onlineStart <= unimatchStart <= unimatchRampEnd')
    if not (0 <= args.unimatchWeight <= 1):
        parser.error('--unimatchWeight must be finite and in [0, 1]')
    if not all(0 <= value <= 1 for value in
               (args.unimatchContrast, args.unimatchBrightness, args.unimatchNoise)):
        parser.error('UniMatch augmentation strengths must be finite and in [0, 1]')
if args.onlineCorrection:
    if not (args.emaTeacher and args.k0Only and args.lossFunc == 'FocalTversky' and args.job == 'train'):
        parser.error('--onlineCorrection requires training, --emaTeacher, --k0Only and FocalTversky')
    if args.pu != 0:
        parser.error('This online correction experiment requires --pu 0; PU remains available when online correction is off')
    args.fixedVal = True
if not (1 <= args.onlineStart <= args.onlineHoldEnd < args.onlineRampEnd):
    parser.error('Require 1 <= onlineStart <= onlineHoldEnd < onlineRampEnd (one-based epochs)')
if not (0 <= args.onlineWeightStart <= args.onlineWeightMax <= 1):
    parser.error('Online pixel weights must satisfy 0 <= start <= max <= 1')
if not (0 < args.onlineSupport <= args.onlineThreshold < 1):
    parser.error('Require 0 < onlineSupport <= onlineThreshold < 1')
if args.onlineBuffer < 0 or args.onlineMinPixels < 1 or not (0 < args.onlineMaxFraction <= 1):
    parser.error('Invalid online buffer, minimum area, or area fraction')
if args.checkpointEvery < 0:
    parser.error('--checkpointEvery must be nonnegative')
if args.emaStageTransition and not (args.emaResume and args.fixedVal):
    parser.error('--emaStageTransition requires --emaResume and --fixedVal (implied by online correction)')

if not 0.0 <= args.emaDecay < 1.0:
    parser.error('--emaDecay must be in [0, 1)')
if args.emaResume and (not args.emaTeacher or args.job != 'train'):
    parser.error('--emaResume requires --emaTeacher and --job train')

if args.emaPreview and (not args.emaTeacher or args.job != 'train'):
    parser.error('--emaPreview requires --emaTeacher and --job train')
if args.previewCount < 1 or args.previewEvery < 1 or args.previewBuffer < 0 or args.previewMinPixels < 1:
    parser.error('Preview count/every/minimum area must be positive; buffer must be nonnegative')
if not 0.0 < args.previewThreshold < 1.0:
    parser.error('--previewThreshold must be in (0, 1)')

def get_config(args):
    config = {
        'dataMode': args.dataMode,
        'dataset': args.dataset,
        'gpuIndex': args.gpuIndex,
        'batchsize': args.batchsize,
        'lr0': args.lr0,
        'stepPerMiniBatch': args.stepPerMiniBatch,
        'resDir': args.resDir,
        'mn': args.mn,
        'job': args.job,
        'isAtten': args.isAtten,
        'alpha': args.alpha,
        'dw': args.dw,
        'd0': args.d0,
        'modelType': args.modelType,
        'dropout': args.dropout}
    return config
def write_config(args):
    config = get_config(args)
    keys = ['dataMode', 'dataset', 'gpuIndex', 'batchsize', 'lr0','stepPerMiniBatch', 'resDir', 'mn', 'job', 'isAtten', 'alpha', 'dw', 'd0','dropout', 'modelType']
    with open(config['resDir'] + 'config.txt', 'w') as f:
        for key in keys:
            f.write(key + ': ' + str(config[key]) + '\n')
        
        
