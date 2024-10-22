#!/bin/bash

#SBATCH --job-name=yolo
#SBATCH --output="./logs/yolov1"
#SBATCH --nodelist=nv178
#SBATCH --gpus=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G

##### Number of total processes
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "


echo "Run started at:- "
date

# ex) srun python -m mnist_resnet50.train

srun python -m yolov1.train
#srun python -m cifar10_resnet.test -model resnet20

#cnt=1
#while [ 1 == 1 ]
#do
#  if [ $cnt -eq 5 ]; then
#    break
#  fi
#  srun python -m cifar10_resnet.train -model resnet56
#  srun python -m cifar10_resnet.test -model resnet56
#  let cnt++
#done
