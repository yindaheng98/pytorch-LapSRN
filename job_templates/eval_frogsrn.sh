#!/bin/bash
#BSUB -J train_frogsrn-$N #任务名称
#BSUB -q gpu_v100  #队列名称，可用bqueues查看
#BSUB -m $GPU      #指定节点
#BSUB -gpu "num=1" #GPU数
#BSUB -o $HERE/log/eval_frogsrn-$N.out
#BSUB -e $HERE/log/eval_frogsrn-$N.err
export TERM=xterm
module load anaconda3
module load cuda-11.02
conda activate /seu_share/home/dongfang/df_yindh/frogsrn
cd $HERE #进入作业工作目录
mkdir -p log
mkdir -p result
python eval.py --cuda \
--model 'model/frogsrn_model_'$N'_epoch_100.pth' \
--result 'result/frogsrn_model_'$N'_epoch_100.json'