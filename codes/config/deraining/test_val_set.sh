#!/bin/bash

#SBATCH --job-name=test_derain
#SBATCH --ntasks=1
#SBATCH --partition=gpuq
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=64

srun --ntasks=1 --nodes=1 --cpus-per-task=64 --gpus-per-task=1 python test.py -opt=/home/prrgpp000/image-restoration-sde/codes/config/deraining/options/test/val_set.yml