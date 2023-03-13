'''
Title: Evaluation of models (Classification)
Author: Osvaldo M Velarde
Project: Feedback connections in visual system

-------------------------------------------------------------
Descripion/Steps:
1. Dataset: Load CIFAR10 dataset
2. Model: R-CNN
3. Loss function
4. Training
5. Evaluation

Details in 'Velarde_et_al_2023'
'''

# ===========================================================
# ============== MODULES & PATHS ============================
import os

import numpy as np
import pandas as pd

import argparse
from argsmod import parse_none, parse_coords, parse_boolean

from torchvision.datasets import CIFAR100, CIFAR10
import torchvision.transforms as tt
from torch.utils.data import DataLoader
cifardata = {10:CIFAR10, 100:CIFAR100}
statsdata = {10:((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 100:((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))}

import torch
import torch.nn as nn
from load_save_models import initialize_model, initialize_optimizer

from utils import progress_bar

from noise_functions import pixel_noise,blurr_noise,contr_noise,occlu_noise, sharp_noise
OPTS_NOISES = { 'pxl':pixel_noise,
                'blurr':blurr_noise,
                'contr':contr_noise,
                'occlu':occlu_noise}

# ===========================================================
# ============== IO & MODEL FILES ===========================
parser = argparse.ArgumentParser(description='Description')
parser.add_argument('--PATH', required=True)
parser.add_argument('--title', required=True)
parser.add_argument('--idx_eval', required=True)

parser.add_argument('--pretraining-model', default=None, type=parse_none)

parser.add_argument('--test-batch-size', default=100, type=int)

parser.add_argument('--device-type', default="cpu", choices=['gpu', 'cpu'])
parser.add_argument('--gpu-number', default=0, type=int)

parser.add_argument('--datapath', required=True)
parser.add_argument('--num-class', default=10, type=int, choices=[10,100])

parser.add_argument('--num-filters', default=64, type=int)
parser.add_argument('--growth-factor', default=2, type=int)
parser.add_argument('--block-per-layer-list', default=[2, 2, 2, 2],  type=int, nargs="+")
parser.add_argument('--block-strides-list', default=[1, 2, 2, 2], type=int, nargs="+")
parser.add_argument('--frozen-BN', default=False, type=parse_boolean)

parser.add_argument('--feedback-connections', default=[], type=parse_coords, nargs="*")
parser.add_argument('--order', default="before", choices=['before', 'after'])
parser.add_argument('--rzs-chs', default="linear", choices=['linear', 'conv'])
parser.add_argument('--type-agg', default="II", choices=['I', 'II', 'III'])
parser.add_argument('--func-agg', default="sum", choices=['sum', 'prod'])

parser.add_argument('--rnn-cell', default="time_decay", choices=['time_decay', 'recip_gated'])
parser.add_argument('--idxs-cell', default=[], type=int, nargs="*")
parser.add_argument('--time-steps', default=1, type=int)

parser.add_argument('--type-noise', default='pxl', type=str)
parser.add_argument('--level-noise', default=0, type=float)

args = parser.parse_args()

type_computation = True
if args.time_steps == 1:
    type_computation = False

if args.feedback_connections == []:
    type_computation = False


# ===========================================================
# ===================== DATASET =============================
num_channels = 3
#num_classes = args.database
cifar=cifardata[args.num_class]
stats = statsdata[args.num_class]

test_data  = cifar(root=args.datapath,train=False,transform=tt.ToTensor())
test_dataloader  = DataLoader(test_data, batch_size = args.test_batch_size, shuffle=False, num_workers=2)

normalization = tt.Normalize(*stats)

# ===========================================================
# ================= NETWORK MODEL  ==========================
model_parameters = {
    "device_type": args.device_type,
    "gpu_number": args.gpu_number,
    "model_path": args.pretraining_model,
    "task":'classification',
    "cfg":{"input_channels": num_channels,
            "num_filters":args.num_filters,
            "first_layer_kernel_size":3,
            "first_layer_conv_stride":1,
            "first_layer_padding":1,
            "first_pool_size":3,
            "first_pool_stride":2,
            "first_pool_padding":1,
            "blocks_per_layer_list":args.block_per_layer_list,
            "block_strides_list":args.block_strides_list,
            "block_fn":'V2',
            "growth_factor":args.growth_factor,
            "frozenBN":args.frozen_BN,
            "feedback_connections":args.feedback_connections,
            "cfg_agg":{'order':args.order,
                        'rsz': args.rzs_chs, 
                        'type': args.type_agg,
                        'function':args.func_agg},
            "rnn_cell":args.rnn_cell,
            "idxs_cell":args.idxs_cell,
            "time_steps":args.time_steps,
            "num_classes":args.num_class,
            "loss_function": nn.CrossEntropyLoss(),
            "bio_computation":type_computation,
            "lr_alg":'BP',
            "typeRBP":'o',
            "truncate_iter":50, 
            "contractor":False}}

model, device, initial_epoch, best_acc = initialize_model(model_parameters)

# ==========================================================
# ===================== OUTPUTS ============================
evaluation_folder = args.PATH + 'results/classification/evaluations/' + args.title + '/'

if not os.path.isdir(evaluation_folder): os.makedirs(evaluation_folder)

# ===========================================================
# ==================== EVALUATION ===========================

# Online Evaluation -----------------------------------------
model.eval()
test_loss = 0.0
correct = 0
total = 0

# Predictions
predictions = []
function_sft = nn.Softmax(dim=1)

# Noise -----------------------------------------------------
level = args.level_noise
noise_function = OPTS_NOISES[args.type_noise] 

# -----------------------------------------------------------

with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(test_dataloader):
        inputs = normalization(inputs)
        inputs = noise_function(inputs,level)

        inputs, targets = inputs.to(device), targets.to(device) # Inputs to device
        outputs, test_loss_batch, _ = model(inputs,targets,batch_idx)   

        test_loss += test_loss_batch.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        batch_results = torch.stack((targets,predicted),1)
        batch_results = torch.hstack((batch_results,function_sft(outputs)))
        predictions.append(batch_results)

        progress_bar(batch_idx, len(test_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    eval_loss_epoch = test_loss/(batch_idx+1)
    eval_acc_epoch = 100.*correct/total

df = pd.DataFrame(torch.cat(predictions).cpu().numpy(),columns=['label','prediction'] + ['class_'+ str(ii) for ii in range(args.num_class)])

print('Loss Eval:',eval_loss_epoch,'Acc Eval:',eval_acc_epoch,'correct:',correct,'total:',total)

# -------------------------------------------------------------------

# Save results
df.to_csv(evaluation_folder + args.idx_eval +'.csv',index=False)