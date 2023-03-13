'''
Title: Training of models (Object Detection)
Author: Osvaldo M Velarde
Project: Feedback connections in visual system

-------------------------------------------------------------
Descripion/Steps:
1. Dataset: Load COCO dataset
2. Model: R-CNN
3. Loss function
4. Training
5. Evaluation

Details in 'Velarde_et_al_2023'
'''

# ===========================================================
# ============== MODULES & PATHS ============================

import argparse
from argsmod import parse_none, parse_coords, parse_boolean

import os
import datetime
import time
import numpy as np

from torch import isnan, save
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
import coco_utils
from group_by_aspect_ratio import create_aspect_ratio_groups, GroupedBatchSampler
import transforms as tt

from load_save_models import initialize_model, initialize_optimizer
from load_save_models import upload_original_fasterrcnn 
from load_save_models import warmup_lr_scheduler
from training_functions import update_grad_rbp, update_grad_bp

from utils import collate_fn, MetricLogger, SmoothedValue
from engine import evaluate

OPTS_UPDATE_GRADS = {'BP':update_grad_bp,'RBP':update_grad_rbp}

# ===========================================================
# ============== IO & MODEL FILES ===========================

parser = argparse.ArgumentParser(description='Description')
parser.add_argument('--PATH', required=True)
parser.add_argument('--title', required=True)

parser.add_argument('--pretraining-model', default=None, type=parse_none)

parser.add_argument('--num-train-epochs', default=26, type=int)
parser.add_argument('--train-batch-size', default=2, type=int)
parser.add_argument('--lr-rate', default=0.1, type=float, help='learning rate')
parser.add_argument('--lr-alg', default="BP", choices=['BP', 'RBP'])
parser.add_argument('--typeRBP', default='o', choices=['o', 'n', 'f'])      #o: original, n: neumman, f:feedbacks
parser.add_argument('--truncate-iter', default=50, type=int)
parser.add_argument('--contractor', default=False, type=parse_boolean)

parser.add_argument('--test-batch-size', default=1, type=int)

parser.add_argument('--device-type', default="cpu", choices=['gpu', 'cpu'])
parser.add_argument('--gpu-number', type=int, default=0)

parser.add_argument('--datapath', required=True)
parser.add_argument('--num-class', default=91, type=int, choices=[91])

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

transform_train = tt.Compose([coco_utils.ConvertCocoPolysToMask(),tt.RandomHorizontalFlip(),tt.ToTensor()])
transform_test = tt.Compose([coco_utils.ConvertCocoPolysToMask(),tt.ToTensor()])

train_data = coco_utils.CocoDetection(args.datapath + "train2017/", args.datapath + "annotations/instances_train2017.json", transforms=transform_train)
train_data = coco_utils._coco_remove_images_without_annotations(train_data)
test_data = coco_utils.CocoDetection(args.datapath + "val2017/", args.datapath + "annotations/instances_val2017.json", transforms=transform_test)

train_sampler = RandomSampler(train_data)
test_sampler = SequentialSampler(test_data)
group_ids = create_aspect_ratio_groups(train_data, k=3)
train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.train_batch_size)

train_dataloader = DataLoader(train_data, batch_sampler=train_batch_sampler, num_workers=4, collate_fn=collate_fn)
test_dataloader = DataLoader(test_data, batch_size=1, sampler=test_sampler, num_workers=4, collate_fn=collate_fn)

# ===========================================================
# ================= NETWORK MODEL  ==========================
model_parameters = {
    "device_type": args.device_type,
    "gpu_number": args.gpu_number,
    "model_path": args.pretraining_model,
    "task":'detection',
    "cfg":{"input_channels": num_channels,
            "num_filters":args.num_filters,
            "first_layer_kernel_size":7,
            "first_layer_conv_stride":2,
            "first_layer_padding":3,
            "first_pool_size":3,
            "first_pool_stride":2,
            "first_pool_padding":1,
            "blocks_per_layer_list":args.block_per_layer_list,
            "block_strides_list":args.block_strides_list,
            "block_fn":'Bottleneck',
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
            "num_classes":arg.num_class,
            "loss_function": None,
            "bio_computation":type_computation,
            "lr_alg":args.lr_alg,
            "typeRBP":args.typeRBP,
            "truncate_iter":args.truncate_iter, 
            "contractor":args.contractor}}

# Initialization of model: Create architecture and upload pre-trained weights (Resnet-50)
model, device, initial_epoch, best_acc = initialize_model(model_parameters)
path_models_pretrained = args.PATH + 'models-pretrained/'
upload_original_fasterrcnn(model.module.segmenter,device,path_models_pretrained)

# ==========================================================
# ===================== OUTPUTS ============================
# Checkpoints and results folder ---------------------------
checkpoint_folder = args.PATH + 'results/detection/checkpoints/'
evolution_folder = args.PATH + 'results/detection/training-evolution/'
prediction_folder = args.PATH + 'results/detection/predictions/' + args.title + '/'

if not os.path.isdir(checkpoint_folder): os.mkdir(checkpoint_folder)
if not os.path.isdir(evolution_folder): os.mkdir(evolution_folder)
if not os.path.isdir(prediction_folder): os.mkdir(prediction_folder)

path_preds = None

# Results folder -------------------------------------------
# Track of losses ------------------------------------------
train_loss_history = []

# ==========================================================
# ==================== TRAINING ============================

num_epochs = args.num_train_epochs
last_epoch = initial_epoch + num_epochs

optimizer_parameters = {
    "name": 'SGD',
    "cfg": {"lr":args.lr_rate,"momentum":0.9,"weight_decay":5e-4}, #lr=0.02/8 -- weight_decay=1e-4
    "optimizer_path": None}

optimizer,scheduler = initialize_optimizer(model,optimizer_parameters,task='detection')
update_grad = OPTS_UPDATE_GRADS[args.lr_alg]

index_epoch = 0

# Training process -----------------------------------------
print("Start training")
start_time = time.time()

for epoch in range(initial_epoch, last_epoch):
    
    # ---------------------------------------------------------------------------------------------
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = {'loss_classifier': 0.0, 'loss_box_reg': 0.0, 'loss_objectness': 0.0, 'loss_rpn_box_reg': 0.0,'total':0.0}
    index_epoch = epoch

    scheduler_soft_init = None
    if epoch == 0: scheduler_soft_init = warmup_lr_scheduler(optimizer, warmup_factor=1./1000, warmup_iters=min(1000, len(train_dataloader)-1))

    # ---------------------------------------------------------------------------------------------
    batch_idx = 0
    for images, targets in metric_logger.log_every(train_dataloader, 100, header):
        optimizer.zero_grad()  
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        train_loss_batch_dict = model(images, targets)
        train_loss_batch = sum(loss for loss in train_loss_batch_dict.values())

        if isnan(train_loss_batch):
            print('Error-Loss NAN')
            exit()

        update_grad(model,train_loss_batch,None)
        optimizer.step()

        if scheduler_soft_init is not None: scheduler_soft_init.step()

        train_loss['total'] += train_loss_batch.item()

        metric_logger.update(loss=train_loss_batch, **train_loss_batch_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        batch_idx += 1
    # ---------------------------------------------------------------------------------------------

    train_loss_epoch = train_loss['total']/(batch_idx+1)
    train_loss_history.append(train_loss_epoch)
    print('Epoch:',epoch,'Loss Train:',train_loss_epoch)

    # Save checkpoint. ----------------------------------------------------------------------------
    print('Saving Checkpoint -- ')
    state = {
            'model': model.state_dict(),
            'epoch': epoch}
    optimizer_state = {
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': scheduler.state_dict()}
    
    save(state, checkpoint_folder +'ckpt_'+ args.title + '.pth')
    save(optimizer_state, checkpoint_folder +'opt_ckpt_'+ args.title + '.pth')

    # ---------------------------------------------------------------------------------------------
    if epoch == last_epoch-1: path_preds = prediction_folder
    evaluate(model, test_dataloader, device=device,path_preds=path_preds)

    # ----------------------------------------------------------------------------------------------

    scheduler.step()

# -------------------------------------------------------------------------------------------------
total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print(f"Training time {total_time_str}")

# Save results
np.savetxt(evolution_folder + args.title +'.dat', np.array([np.arange(initial_epoch,index_epoch+1),train_loss_history]), fmt="%.5f")