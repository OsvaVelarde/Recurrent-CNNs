#!/bin/bash

export PYTHONPATH=$(pwd):$PYTHONPATH

EXP=$1
filename1='cfgfiles/struct_exp_'$EXP'.cfg'
source $filename1

_PATH_MODEL=$_PATH'results/classification/checkpoints/'
MODEL="${_PATH_MODEL}ckpt_exp_${EXP}.pth"
DATABASE_PATH='/media/osvaldo/Seagate Basic/classification_datasets/'

LIST_TYPE_NOISES=('pxl' 'blurr' 'occlu')
LEVEL=0
index=0

for tt in "${!LIST_TYPE_NOISES[@]}"; do

    str_id=$(printf "%02g\n" $index)
    TITLE="exp_${EXP}/${LIST_TYPE_NOISES[tt]}_noise"
    IDX_EVAL="ev_"${str_id}
    echo "Running ${TITLE}" "${LEVEL}"

    python3.7 src/evaluation_classification.py \
        --PATH $_PATH \
        --title  $TITLE\
        --idx_eval $IDX_EVAL \
        --pretraining-model $MODEL \
        --datapath '/media/osvaldo/Seagate Basic/classification_datasets/' \
        --num-class $NUM_CLASS \
        --test-batch-size $TEST_BATCH_SIZE \
        --device-type $DEVICE_TYPE \
        --gpu-number $GPU_NUMBER \
        --num-filters $NUM_FILTERS \
        --growth-factor $GROWTH_FACTOR\
        --block-per-layer-list ${BLOCKS_PER_LAYER[@]}\
        --block-strides-list ${STRIDES_PER_LAYER[@]} \
        --rnn-cell $RNN_CELL\
        --time-steps $TIME_STEPS \
        --idxs-cell ${IDXS_CELL[@]} \
        --feedback-connections ${FEEDBKS_CONN[@]} \
        --order $ORDER \
        --rzs-chs $RZS_CHS \
        --type-agg $TYPE_AGG \
        --func-agg $FUNC_AGG \
        --type-noise 'pxl'\
        --level-noise $LEVEL

done