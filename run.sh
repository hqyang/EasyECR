#!/usr/bin/env bash
start_time=`date +%Y%m%d%H%M%S`
echo "start ${start_time}--------------------------------------------------"

gpu_card=$1
shift
export CUDA_VISIBLE_DEVICES=${gpu_card}

CUR_DIR=`pwd`

ROOT=${CUR_DIR}

export PYTHONPATH=${ROOT}:${PYTHONPATH}

/home/nobody/.virtualenvs/EasyECR/bin/python $@

end_time=`date +%Y%m%d%H%M%S`
echo ${end_time}
