#!/bin/bash
OMP_NUM_THREADS=1 yhrun -n $SLURM_NPROCS -p TH_SR1 python run.py --id $1 --step_per_iter $((100*$SLURM_NPROCS))
