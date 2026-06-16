#!/bin/bash
#SBATCH -t 48:00:10
#SBATCH -J "BOOKS"
#SBATCH --gres=gpu:a100:2

module load system/CUDA/12.6.0
python train_muse_books.py