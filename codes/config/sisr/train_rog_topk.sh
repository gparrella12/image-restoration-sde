#!/bin/bash

#SBATCH --job-name=rog_topk    # Job name
#SBATCH --ntasks=2                 # Number of tasks
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --partition=gpuq           # Partition name
#SBATCH --gres=gpu:2                  # Total number of GPUs
#SBATCH --cpus-per-task=20         # Number of CPUs per task

# Esegui entrambi i torchrun in parallelo
torchrun --nproc_per_node=2 --master_port=4231 train.py -opt=/home/prrgpp000/image-restoration-sde/codes/config/sisr/options/train/my_refusion_rog_topk.yml --launcher pytorch