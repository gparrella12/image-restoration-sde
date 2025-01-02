#!/bin/bash

#SBATCH --job-name=ir_test
#SBATCH --ntasks=1
#SBATCH --partition=gpuq
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4

srun --ntasks=1 --nodes=1 --cpus-per-task=4 --gpus-per-task=1 python3 train.py -opt=/home/prrgpp000/image-restoration-sde/codes/config/sisr/options/train/my_refusion_test.yml