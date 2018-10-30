#!/bin/bash
OMP_NUM_THREADS=1 yhrun -n $SLURM_NPROCS -p TH_SR1 python -u run.py --id $1 --step_per_iter $((100*$SLURM_NPROCS)) --layer_norm --difficulty=1 --reward=10 --fix_target --policy=res --net 256 256 256 256 256 256
