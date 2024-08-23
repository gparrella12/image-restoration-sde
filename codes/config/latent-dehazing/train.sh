#!/bin/bash

#SBATCH --job-name=dehaze
#SBATCH --ntasks=1
#SBATCH --partition=gpuq
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4

srun --ntasks=1 --nodes=1 --cpus-per-task=4 --gpus-per-task=1 python3 train.py -opt=/home/prrgpp000/image-restoration-sde/codes/config/latent-dehazing/options/dehazing/train/my_nasde.yml