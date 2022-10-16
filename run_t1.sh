#!/usr/bin/env bash
export LR=0.7e-04
export FT_LR=0.7e-5
export BACKBONE_LR=0.7e-05

GPUS_PER_NODE=2 ./tools/run_dist_launch.sh 2 configs/OWOD_our_proposed_split_t1_hm.sh
