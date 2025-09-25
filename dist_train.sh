#!/usr/bin/env bash

GPUS=$1
CONFIG=$2
PORT=$3

# usage
if [ $# -lt 3 ] ;then
    echo "usage:"
    echo "./scripts/dist_train.sh [number of gpu] [path to option file] [port]"
    exit
fi

PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    recurrent_basicsr_train.py -opt $CONFIG --launcher pytorch ${@:4}
