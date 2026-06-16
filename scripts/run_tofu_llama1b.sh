#!/bin/bash
#SBATCH -t 48:00:10
#SBATCH -J "AUTOMU"
#SBATCH --gres=gpu:a100:2

module load system/CUDA/12.6.0
python train.py --model_name open-unlearning/tofu_Llama-3.2-1B-Instruct_full   --data_dir tofu_data   --forget_split forget10_train.jsonl   --retain_split retain90.jsonl --no_retain