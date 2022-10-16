#!/usr/bin/env bash

set -x
set -o pipefail

LR=${LR:-2e-4}
FT_LR=${FT_LR:-2e-4}
BACKBONE_LR=${BACKBONE_LR:-2e-05}

DS_DIR=exps/hm_datasets
mkdir -p ${DS_DIR}

EXP_DIR=exps/OWDETR_t1
PY_ARGS=${@:1}
mkdir -p ${EXP_DIR}

# Build the datasets
python -u main_open_world.py \
    --output_dir ${DS_DIR} --dataset owod --num_queries 100 --eval_every 5 \
    --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 19 --data_root './data/OWDETR' --train_set 't1_train' --test_set 't1_train' --num_classes 81 \
    --unmatched_boxes --epochs 45 --lr $LR --lr_backbone $BACKBONE_LR --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    --resume 'exps/OWDETR_t1/checkpoint0044.pth' \
    --build_hard_mode_datasets --ft_samples 100 \
    ${PY_ARGS}  2>&1 | tee -a ${EXP_DIR}/run_log.txt

test $? -ne 0 && exit 1


if [[ ! -f "${EXP_DIR}/checkpoint0044.pth" ]]; then
python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 5 \
    --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 19 --data_root './data/OWDETR' --train_set 't1_train' --test_set 'test' --num_classes 81 \
    --unmatched_boxes --epochs 45 --lr $LR --lr_backbone $BACKBONE_LR --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    ${PY_ARGS}  2>&1 | tee -a ${EXP_DIR}/run_log.txt
fi
