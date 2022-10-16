#!/bin/bash -l
#PBS -N og_t4
#PBS -l ncpus=32,mem=512GB,ngpus=8,gputype=A100,walltime=23:59:00
cd $PBS_O_WORKDIR
conda activate owdetr
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 2 configs/OWOD_our_proposed_split_t4.sh
