#!/bin/bash

#SBATCH --job-name=ir_sde_train    # Job name
#SBATCH --ntasks=2                 # Number of tasks
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --partition=gpuq           # Partition name
#SBATCH --gres=gpu:2                  # Total number of GPUs
#SBATCH --cpus-per-task=20         # Number of CPUs per task


CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=4231 train.py -opt=/home/prrgpp000/image-restoration-sde/codes/config/sisr/options/train/my_refusion_rog_distributed.yml --launcher pytorch