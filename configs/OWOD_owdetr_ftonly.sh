#!/usr/bin/env bash

set -x

#EXP_DIR=exps/OWDETR_t1
#PY_ARGS=${@:1}
#mkdir -p ${EXP_DIR}

#python -u main_open_world.py \
#    --output_dir ${EXP_DIR} --dataset owdetr --num_queries 100 --eval_every 5 \
#    --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 19 --data_root './data/OWDETR' --train_set 'owdetr_eval_ftonly_t1' --test_set 'owdetr_eval_test_t1' --num_classes 81 \
#    --unmatched_boxes --epochs 500 --lr 4e-5 --lr_backbone 4e-6 --lr_drop 400 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
#    --backbone 'dino_resnet50' \
#    --batch_size 3 \
#    ${PY_ARGS}  2>&1 | tee -a ${EXP_DIR}/run_log.txt


EXP_DIR=exps/OWDETR_t2
PY_ARGS=${@:1}
mkdir -p ${EXP_DIR}

python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owdetr --num_queries 100 --eval_every 5 \
    --PREV_INTRODUCED_CLS 19 --CUR_INTRODUCED_CLS 21 --data_root './data/OWDETR' --train_set 'owdetr_eval_ftonly_t2' --test_set 'owdetr_eval_test_t2' --num_classes 81 \
    --unmatched_boxes --epochs 300 --lr 4e-5 --lr_backbone 4e-6 --lr_drop 250 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    ${PY_ARGS}  2>&1 | tee -a ${EXP_DIR}/run_log.txt

EXP_DIR=exps/OWDETR_t3
PY_ARGS=${@:1}
mkdir -p ${EXP_DIR}

python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owdetr --num_queries 100 --eval_every 5 \
    --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 --data_root './data/OWDETR' --train_set 'owdetr_eval_ftonly_t3' --test_set 'owdetr_eval_test_t3' --num_classes 81 \
    --unmatched_boxes --epochs 300 --lr 4e-5 --lr_backbone 4e-6 --lr_drop 250 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    ${PY_ARGS}  2>&1 | tee -a ${EXP_DIR}/run_log.txt

EXP_DIR=exps/OWDETR_t4
PY_ARGS=${@:1}
mkdir -p ${EXP_DIR}

python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owdetr --num_queries 100 --eval_every 5 \
    --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 --data_root './data/OWDETR' --train_set 'owdetr_eval_ftonly_t4' --test_set 'owdetr_eval_test_t4' --num_classes 81 \
    --unmatched_boxes --epochs 300 --lr 4e-5 --lr_backbone 4e-6 --lr_drop 250 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    ${PY_ARGS}  2>&1 | tee -a ${EXP_DIR}/run_log.txt

