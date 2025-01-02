#!/bin/bash

#SBATCH --job-name=sr_rog
#SBATCH --ntasks=1
#SBATCH --partition=gpuq
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=20

srun --ntasks=1 --exclusive --nodes=1 --cpus-per-task=20 --gpus-per-task=1 python3 train.py -opt=/home/prrgpp000/image-restoration-sde/codes/config/sisr/options/train/my_refusion_rog_final.yml