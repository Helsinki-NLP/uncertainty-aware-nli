#!/bin/bash
#SBATCH -J MNLI
#SBATCH -o out_%J.txt
#SBATCH -e err_%J.txt
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH -t 10:00:00
#SBATCH --mem-per-cpu=64000
#SBATCH --gres=gpu:v100:1
#SBATCH --account=project_2001194
# run command

module purge
module load pytorch/1.6
export MODEL=${1}

srun python main.py \
    --model $MODEL \
    --learning_rate 2e-5 \
    --batch_size 8 \
    --epochs 3 \
    --gpu 0 \
    --data_path data \
    --output_path output/$MODEL
