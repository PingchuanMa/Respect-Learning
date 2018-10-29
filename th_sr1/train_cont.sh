#!/bin/bash
OMP_NUM_THREADS=1 yhrun -n $SLURM_NPROCS -p TH_SR1 python tools/run.py --id ClimberWalkerEnv-v$1 --timesteps_per_actorbatch $2  --cont --iter $3
