import argparse
import os
import numpy as np
from project_paths import DATA_DIR, OUTPUT_DIR, QFAULTS_FILE

parser = argparse.ArgumentParser(
    description='Train or analyse the Earthquake-SAM model',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--job', default='train', type=str, help='operation to run')
parser.add_argument('--dataMode', default='both', type=str, help='input data mode')
parser.add_argument('--gpuIndex', default=0, type=int, help='CUDA device index')
parser.add_argument('--batchsize', default=8, type=int, help='training batch size')
parser.add_argument('--transN', default=1, type=int, help='legacy transform count')
parser.add_argument('--lr0', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--resDir', default=str(OUTPUT_DIR / 'train_default'), type=str, help='output directory')
parser.add_argument('--mn', default='256', type=str, help='crop size, e.g. 256 or 256,256')
parser.add_argument('--isAtten', default=False, help='enable attention', action='store_true')
parser.add_argument('--alpha', default=0.1, type=float, help='minimum sample weight')
parser.add_argument('--dw', default=0.15, type=float, help='distance weighting scale')
parser.add_argument('--d0', default=0.05, type=float, help='positive-label distance threshold')
parser.add_argument('--modelType', default='mars_sam', type=str, help='model architecture')
parser.add_argument('--stepPerMiniBatch', default=8, type=int, help='gradient accumulation steps')
parser.add_argument('--dropout', default=0, type=float, help='dropout probability')
parser.add_argument('--threshold', default=0.5, type=float, help='prediction threshold')
parser.add_argument('--dataset', default=str(DATA_DIR / 'key_train.lst'), type=str, help='training key list')
parser.add_argument('--datasetAdd', default='', type=str, help='optional additional key list')
parser.add_argument('--faultFile', default=str(QFAULTS_FILE), type=str, help='fault shapefile for legacy analysis jobs')
parser.add_argument('--valueFile', default='', type=str, help='feature HDF5 for legacy analysis jobs')
parser.add_argument('--predictFile', default='', type=str, help='prediction HDF5 for legacy analysis jobs')
parser.add_argument('--value', default='feature', type=str, help='feature name')
parser.add_argument('--isalign', default=False, help='enable spatial alignment', action='store_true')
parser.add_argument('--isOther', default=False, help='enable other-landform label', action='store_true')
parser.add_argument('--swMode', default='single', type=str, help='sample-weight mode')
parser.add_argument('--lossFunc', default='Dice', type=str, help='Dice | IoU | FocalTversky')
parser.add_argument('--pu', default=0.0, type=float, help='PU discount strength 0..1 for unlabeled-but-confident pixels (FocalTversky only, 0=off)')
parser.add_argument('--puWarmup', default=20, type=int, help='epochs of plain training before PU discount starts')
parser.add_argument('--puRamp', default=20, type=int, help='epochs to linearly ramp PU discount to full strength')
parser.add_argument('--k0Only', default=False, help='train loss on k0 (any-fault) channel only', action='store_true')
parser.add_argument('--epochs', default=160, type=int, help='training epochs')

args = parser.parse_args()
args.resDir = os.path.join(args.resDir, '')

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
