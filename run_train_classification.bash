#!/bin/bash

TITLE='exp_'$1
filename1='cfgfiles/struct_'$TITLE'.cfg'
filename2='cfgfiles/train_'$TITLE'.cfg' 

source $filename1
source $filename2

DATABASE_PATH='/media/osvaldo/Seagate Basic/classification_datasets/'

export PYTHONPATH=$(pwd):$PYTHONPATH

echo 'Training Stage - Classifier' $TITLE

python3.7 src/training_classification.py \
    --PATH $_PATH \
    --title  $TITLE\
    --pretraining-model $PRETRAINED_MODEL \
    --datapath '/media/osvaldo/Seagate Basic/classification_datasets/' \
    --num-class $NUM_CLASS \
    --num-train-epochs $NUM_EPOCHS \
    --train-batch-size $TRAIN_BATCH_SIZE \
    --lr-rate $LR_RATE \
    --lr-alg $LR_ALG \
    --typeRBP $TYPE_RBP \
    --truncate-iter $TRUNC_ITER \
    --contractor $CONTRACTOR \
    --test-batch-size $TEST_BATCH_SIZE \
    --device-type $DEVICE_TYPE \
    --gpu-number $GPU_NUMBER \
    --num-filters $NUM_FILTERS \
    --growth-factor $GROWTH_FACTOR\
    --frozen-BN $FROZEN_BN\
    --block-per-layer-list ${BLOCKS_PER_LAYER[@]}\
    --block-strides-list ${STRIDES_PER_LAYER[@]} \
    --rnn-cell $RNN_CELL\
    --time-steps $TIME_STEPS \
    --idxs-cell ${IDXS_CELL[@]} \
    --feedback-connections ${FEEDBKS_CONN[@]} \
    --order $ORDER \
    --rzs-chs $RZS_CHS \
    --type-agg $TYPE_AGG \
    --func-agg $FUNC_AGG