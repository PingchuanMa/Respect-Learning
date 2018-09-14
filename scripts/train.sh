#!/usr/bin/env bash

# Script for training videos in TensorFlow

export LD_LIBRARY_PATH=/mnt/lustre/share/cuda-9.0/lib64/:$LD_LIBRARY_PATH
if [ ! -d "../log" ]; then
  mkdir "../log"
fi

python=/mnt/lustre/DATAshare2/mapingchuan/anaconda3/envs/opensim-rl/bin/python

if [ $1 ]; then
  identifier=$1
else
  identifier=origin
fi

if [ $2 ]; then
  num_cpus=$2
else
  num_cpus=8
fi

if [ $3 ]; then
  num_gpus=$3
else
  num_gpus=8
fi

if [ $4 ]; then
  machines=$4
else
  machines=TITANXP
fi

jobname=TEST
step_per_iter=512
log_dir=../log/${identifier}-$( { date +"%H:%M:%S-%h_%d_%y"; } 2>&1 ).log

srun -p ${machines} --gres=gpu:${num_gpus} -n${num_cpus} --job-name=${jobname} --mpi=pmi2 --kill-on-bad-exit=1 \
${python} -u ../run.py --step_per_iter=${step_per_iter} --id=${identifier} \
       2>&1|tee ${log_dir} &\
