#!/bin/bash
#BSUB -J dataset_gen #任务名称
#BSUB -q gpu_v100  #队列名称，可用bqueues查看
#BSUB -m $GPU      #指定节点
#BSUB -gpu "num=1" #GPU数
#BSUB -o $HERE/log/dataset_gen.out
#BSUB -e $HERE/log/dataset_gen.err
export TERM=xterm
cd $HERE #进入作业工作目录
mkdir -p log
mkdir -p frames/540p
mkdir -p frames/1080p
mkdir -p frames/4K
../ffmpeg -i ../4K.webm -g 30 -ss 00:00:00 -t 00:00:30 -s 960x540 "frames/540p/frame%3d.png"
../ffmpeg -i ../4K.webm -g 30 -ss 00:00:00 -t 00:00:30 -s 1920x1080 "frames/1080p/frame%3d.png"
../ffmpeg -i ../4K.webm -g 30 -ss 00:00:00 -t 00:00:30 "frames/4K/frame%3d.png"