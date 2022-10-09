#!/bin/bash -l
#PBS -N og_full
#PBS -l ncpus=32,mem=256GB,ngpus=8,gputype=A100,walltime=192:00:00
cd $PBS_O_WORKDIR
conda activate owdetr || exit 1
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 configs/OWOD_our_proposed_split.sh
