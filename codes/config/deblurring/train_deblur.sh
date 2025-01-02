#!/bin/bash

#SBATCH --job-name=two-step-deblur
#SBATCH --ntasks=1
#SBATCH --partition=gpuq
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8

srun --ntasks=1 --nodes=1 --cpus-per-task=8 --gpus-per-task=1 python3 train.py -opt=/home/prrgpp000/image-restoration-sde/codes/config/deblurring/options/train/my-refusion-deblur.yml