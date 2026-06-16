#!/bin/bash
#SBATCH -t 48:00:10
#SBATCH -J "AUTOMU"
#SBATCH --gres=gpu:a100:2

module load system/CUDA/12.6.0
python train_muse_news.py