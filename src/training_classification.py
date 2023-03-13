'''
Title: Training of models (Classification)
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

import numpy as np
import pandas as pd
import os

import argparse
from argsmod import parse_none, parse_coords, parse_boolean

import torch
import torch.nn as nn

from torchvision.datasets import CIFAR100, CIFAR10
import torchvision.transforms as tt
from torch.utils.data import DataLoader

from load_save_models import initialize_model, initialize_optimizer
from training_functions import update_grad_rbp, update_grad_bp

from utils import progress_bar

cifardata = {10:CIFAR10, 100:CIFAR100}
statsdata = {10:((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 100:((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))}

OPTS_UPDATE_GRADS = {'BP':update_grad_bp,'RBP':update_grad_rbp}

# ===========================================================
# ============== IO & MODEL FILES ===========================

parser = argparse.ArgumentParser(description='Description')
parser.add_argument('--PATH', required=True)
parser.add_argument('--title', required=True)

parser.add_argument('--pretraining-model', default=None, type=parse_none)

parser.add_argument('--num-train-epochs', default=200, type=int)
parser.add_argument('--train-batch-size', default=128, type=int)
parser.add_argument('--lr-rate', default=0.1, type=float, help='learning rate')
parser.add_argument('--lr-alg', default="BP", choices=['BP', 'RBP'])
parser.add_argument('--typeRBP', default='o', choices=['o', 'n', 'f'])      #o: original, n: neumman, f:feedbacks
parser.add_argument('--truncate-iter', default=50, type=int)
parser.add_argument('--contractor', default=False, type=parse_boolean)

parser.add_argument('--test-batch-size', default=100, type=int)

parser.add_argument('--device-type', default="cpu", choices=['gpu', 'cpu'])
parser.add_argument('--gpu-number', type=int, default=0)

parser.add_argument('--datapath', required=True)
parser.add_argument('--num-class', default=10, type=int, choices=[10,100])

parser.add_argument('--num-filters', default=64, type=int)
parser.add_argument('--growth-factor', default=2, type=int)
parser.add_argument('--block-per-layer-list', default=[2, 2, 2, 2], nargs="+", type=int)
parser.add_argument('--block-strides-list', default=[1, 2, 2, 2], nargs="+", type=int)
parser.add_argument('--frozen-BN', default=False, type=parse_boolean)

parser.add_argument('--feedback-connections', default=[], type=parse_coords, nargs="*")
parser.add_argument('--order', default="before", choices=['before', 'after'])
parser.add_argument('--rzs-chs', default="linear", choices=['linear', 'conv'])
parser.add_argument('--type-agg', default="II", choices=['I', 'II', 'III'])
parser.add_argument('--func-agg', default="sum", choices=['sum', 'prod'])

parser.add_argument('--rnn-cell', default="time_decay", choices=['time_decay', 'recip_gated'])
parser.add_argument('--idxs-cell', default=[], nargs="*", type=int)
parser.add_argument('--time-steps', default=1, type=int)

args = parser.parse_args()

type_computation = True
if args.time_steps == 1:
    args.lr_alg = 'BP'
    type_computation = False

if args.feedback_connections == []:
    type_computation = False

# ===========================================================
# ===================== DATASET =============================
num_channels=3
cifar=cifardata[args.num_class]
stats = statsdata[args.num_class]

transform_train = tt.Compose([
    tt.RandomCrop(32, padding=4),tt.RandomHorizontalFlip(),
    tt.ToTensor(), tt.Normalize(*stats)])

transform_test = tt.Compose([tt.ToTensor(),tt.Normalize(*stats)])

train_data = cifar(root=args.datapath, transform=transform_train)
test_data  = cifar(root=args.datapath, train=False,transform=transform_test)

train_dataloader = DataLoader(train_data, batch_size = args.train_batch_size, shuffle=True, num_workers=2)
test_dataloader  = DataLoader(test_data, batch_size = args.test_batch_size, shuffle=False, num_workers=2)

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
            "time_steps":args.time_steps,
            "rnn_cell":args.rnn_cell,
            "idxs_cell":args.idxs_cell,
            "num_classes":args.num_class,
            "loss_function": nn.CrossEntropyLoss(),
            "bio_computation":type_computation,
            "lr_alg":args.lr_alg,
            "typeRBP":args.typeRBP,
            "truncate_iter":args.truncate_iter, 
            "contractor":args.contractor}}

model, device, initial_epoch, best_acc = initialize_model(model_parameters)
#TO-DO: FIX FEEDFORWARD CONNECTION IN R-CNNS WITH PRE-TRAINED MODEL.

# ==========================================================
# ===================== OUTPUTS ============================
# Checkpoints and results folder ---------------------------
checkpoint_folder = args.PATH + 'results/classification/checkpoints/'
evolution_folder = args.PATH + 'results/classification/training-evolution/'
prediction_folder = args.PATH + 'results/classification/predictions/'

if not os.path.isdir(checkpoint_folder): os.mkdir(checkpoint_folder)
if not os.path.isdir(evolution_folder): os.mkdir(evolution_folder)
if not os.path.isdir(prediction_folder): os.mkdir(prediction_folder)

# Results folder -------------------------------------------
# Track of losses ------------------------------------------
train_loss_history = []
eval_loss_history = []
train_acc_history = []
eval_acc_history = []

# ==========================================================
# ==================== TRAINING ============================

num_epochs = args.num_train_epochs
last_epoch = initial_epoch + num_epochs

optimizer_parameters = {
    "name": 'SGD',
    "cfg": {"lr":args.lr_rate,"momentum":0.9,"weight_decay":5e-4}, #momentum = 0
    "optimizer_path": None}

optimizer,scheduler = initialize_optimizer(model,optimizer_parameters,task='classification')

num_stag = 0
index_epoch = 0

function_sft = nn.Softmax(dim=1)

update_grad = OPTS_UPDATE_GRADS[args.lr_alg]

# Training process -----------------------------------------
for epoch in range(initial_epoch, last_epoch):
    
    # Training ------------------------------------------------
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    index_epoch = epoch

    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        optimizer.zero_grad()                                               # Clear-the-gradients

        inputs, targets = inputs.to(device), targets.to(device)             # Inputs to device
        outputs, train_loss_batch, grad = model(inputs,targets)             # Prediction

        update_grad(model,train_loss_batch,grad)

        optimizer.step()                                        # parameter-update
        train_loss += train_loss_batch.item()                 # update training loss

        if torch.isnan(train_loss_batch):
            print('Error-Loss NAN')
            exit()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    train_loss_epoch = train_loss/(batch_idx+1)
    train_acc_epoch = 100.*correct/total
    train_loss_history.append(train_loss_epoch)
    train_acc_history.append(train_acc_epoch)

    print('Epoch:',epoch,'Loss Train:',train_loss_epoch,'Acc Train:',train_acc_epoch,'correct:',correct,'total:',total)
    # Online Evaluation ------------------------------------------------
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    # Predictions
    predictions = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dataloader):
            inputs, targets = inputs.to(device), targets.to(device) # Inputs to device
            outputs, test_loss_batch, _ = model(inputs,targets)   

            test_loss += test_loss_batch.item()
            if torch.isnan(train_loss_batch):
                print('Error-Loss NAN')
                exit()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if epoch == last_epoch-1:
                batch_results = torch.stack((targets,predicted),1)
                batch_results = torch.hstack((batch_results,function_sft(outputs)))
                predictions.append(batch_results)

            progress_bar(batch_idx, len(test_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        eval_loss_epoch = test_loss/(batch_idx+1)
        eval_acc_epoch = 100.*correct/total
        eval_loss_history.append(eval_loss_epoch)
        eval_acc_history.append(eval_acc_epoch)

    if epoch == last_epoch-1:
        df = pd.DataFrame(torch.cat(predictions).cpu().numpy(),columns=['label','prediction'] + ['class_'+ str(ii) for ii in range(args.num_class)])

    print('Epoch:',epoch,'Loss Eval:',eval_loss_epoch,'Acc Eval:',eval_acc_epoch,'correct:',correct,'total:',total)

    # Save checkpoint. -------------------------------------------------
    acc = 100.*correct/total
    if acc > best_acc:
        print('Checkpoint -- ')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch}
        
        torch.save(state, checkpoint_folder +'ckpt_'+ args.title + '.pth')
        best_acc = acc
        num_stag = 0

    scheduler.step()

# Save results
np.savetxt(evolution_folder + args.title +'.dat', np.array([np.arange(initial_epoch,index_epoch+1),train_loss_history, train_acc_history, eval_loss_history, eval_acc_history]), fmt="%.5f")
df.to_csv(prediction_folder + args.title +'.csv',index=False)