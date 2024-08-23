#!/bin/bash

#SBATCH --job-name=test
#SBATCH --ntasks=1
#SBATCH --partition=gpuq
#SBATCH --gpus-per-task=1

srun --partition=gpuq --ntasks=1 --nodes=1 python test.py -opt /home/prrgpp000/image-restoration-sde/codes/config/latent-dehazing/options/dehazing/test/my_nasde.yml