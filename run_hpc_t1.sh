#!/bin/bash -l
#PBS -N og_hm_t1
#PBS -l ncpus=24,mem=128GB,ngpus=4,gputype=P100,walltime=144:59:00

export LR=1.4e-04
export FT_LR=1.4e-05
export BACKBONE_LR=1.4e-05

cd $PBS_O_WORKDIR
conda activate owdetr || exit 1
GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 configs/OWOD_our_proposed_split_t1_hm.sh
