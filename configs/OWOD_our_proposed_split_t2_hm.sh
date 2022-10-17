#!/usr/bin/env bash

set -x
set -o pipefail

LR=${LR:-2e-5}
FT_LR=${FT_LR:-2e-4}
BACKBONE_LR=${BACKBONE_LR:-2e-05}

DS_DIR=exps/hm_datasets
EXP_DIR=exps/OWDETR_t2
PY_ARGS=${@:1}
mkdir -p ${EXP_DIR}

# Build the datasets
if [[ ! -f exps/hm_datasets/t2_ft.json ]]; then
python -u main_open_world.py \
    --output_dir ${DS_DIR} --dataset owod --num_queries 100 --eval_every 5 \
    --PREV_INTRODUCED_CLS 19 --CUR_INTRODUCED_CLS 21 --data_root './data/OWDETR' --train_set 't2_train' --test_set 't2_train' --num_classes 81 \
    --unmatched_boxes --epochs 45 --lr $LR --lr_backbone $BACKBONE_LR --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    --resume 'exps/OWDETR_t1/checkpoint0044.pth' \
    --build_hard_mode_datasets --ft_samples 100 \
    ${PY_ARGS}  2>&1 | tee -a ${EXP_DIR}/run_log.txt
fi

if [[ ! -f exps/hm_datasets/t2_ft.json ]]; then
   exit 1
fi

if [[ ! -f "exps/OWDETR_t2/checkpoint0049.pth" ]]; then
python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owdetr --num_queries 100 --eval_every 5 \
    --PREV_INTRODUCED_CLS 19 --CUR_INTRODUCED_CLS 21 --data_root './data/OWDETR' --json_data_root="${DS_DIR}" --train_set 't2_train_filtered' --test_set 'test' --num_classes 81 \
    --unmatched_boxes --epochs 50 --lr $LR --lr_backbone $BACKBONE_LR --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    --pretrain 'exps/OWDETR_t1/checkpoint0044.pth' \
    ${PY_ARGS}  2>&1 | tee -a ${EXP_DIR}/run_log.txt

test $? -ne 0 && exit 1
fi

EXP_DIR=exps/OWDETR_t2_ft
PY_ARGS=${@:1}
mkdir -p ${EXP_DIR}

python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owdetr --num_queries 100 --eval_every 5 \
    --PREV_INTRODUCED_CLS 19 --CUR_INTRODUCED_CLS 21 --data_root './data/OWDETR' --json_data_root="${DS_DIR}" --train_set 't2_ft' --test_set 'test' --num_classes 81 \
    --unmatched_boxes --epochs 100 --lr $FT_LR --lr_backbone $BACKBONE_LR --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    --pretrain 'exps/OWDETR_t2/checkpoint0049.pth' \
    ${PY_ARGS}  2>&1 | tee -a ${EXP_DIR}/run_log.txt

