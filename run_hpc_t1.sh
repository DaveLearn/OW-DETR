#!/bin/bash -l
#PBS -N og_t1
#PBS -l ncpus=24,mem=128GB,ngpus=4,gputype=P100,walltime=23:59:00
cd $PBS_O_WORKDIR
conda activate owdetr || exit 1
GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 configs/OWOD_our_proposed_split_t1.sh
