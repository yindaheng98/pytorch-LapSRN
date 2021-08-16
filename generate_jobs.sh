#!/bin/bash -x
N=$1
while true; do
    for GPU in $(lsload | grep gpu | grep ok | awk '{print $1}'); do
        echo "bsub -q gpu_v100 -m \"$GPU\" -gpu \"num=1\" python main_frogsrn.py --cuda --depth $N"
        ((N--))
        if ((N <= 0)); then
            break 2
        fi
    done
done
