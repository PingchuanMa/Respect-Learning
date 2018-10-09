#!/usr/bin/env bash

export LD_LIBRARY_PATH=/mnt/lustre/share/cuda-9.0/lib64/:$LD_LIBRARY_PATH
if [ ! -d "../log" ]; then
  mkdir "../log"
fi

python=/mnt/lustre/DATAshare2/mapingchuan/anaconda3/envs/opensim-rl/bin/python

identifier=origin

num_cpus=8
num_gpus=0
machines=$1

policy=mpc

jobname=TEST
step_per_iter=512
log_dir=../log/${identifier}-$( { date +"%H:%M:%S-%h_%d_%y"; } 2>&1 ).log

OMP_NUM_THREADS=1 srun -p Bigvideo4 -w ${machines} --gres=gpu:${num_gpus} -n${num_cpus} --cpu_bind=threads --ntasks-per-node=${num_cpus} --cpus-per-task=1 --job-name=${jobname} --mpi=pmi2 --kill-on-bad-exit=1 \
${python} -u ../run.py --step_per_iter=${step_per_iter} --id=${identifier} --layer_norm --difficulty=1 --reward=8 --fix_target --policy=${policy} \
       2>&1|tee ${log_dir} &\
