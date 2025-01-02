#!/bin/bash

#SBATCH --job-name=two_step_derain
#SBATCH --ntasks=1
#SBATCH --partition=gpuq
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=64

srun --ntasks=1 --nodes=1 --cpus-per-task=64 --gpus-per-task=1 python train.py -opt=/home/prrgpp000/image-restoration-sde/codes/config/deraining/options/train/two-step.yml