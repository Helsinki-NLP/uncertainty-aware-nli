#!/bin/bash
#SBATCH -J SWA-NLI
#SBATCH -o out_%J.txt
#SBATCH -e err_%J.txt
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH -t 72:00:00
#SBATCH --mem-per-cpu=64000
#SBATCH --gres=gpu:v100:1
#SBATCH --account=project_2001194
# run command

module purge
module load pytorch/1.11
export DATASET=${1}
export OPTIMIZER=${2}
export METHOD=${3}

srun python main.py \
    --model roberta \
    --batch_size 32 \
    --epochs 20 \
    --early_stopping 3 \
    --optimizer ${OPTIMIZER} \
    --gpu 0 \
    --method ${METHOD} \
    --dataset ${DATASET}