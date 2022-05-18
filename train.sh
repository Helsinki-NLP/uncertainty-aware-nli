#!/bin/bash
#SBATCH -J MNLI
#SBATCH -o out_%J.txt
#SBATCH -e err_%J.txt
#SBATCH -p gpusmall
#SBATCH -n 1
#SBATCH -t 72:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --account=project_2001194
# run command

module purge
module load pytorch/1.6
export DATASET=${1}
export OPTIMIZER=${2}
export METHOD=${3}

srun python main.py \
    --model roberta \
    --batch_size 32 \
    --epochs 5 \
    --gpu 0 \
    --method ${METHOD} \
    --dataset ${DATASET} \
    --output_path output/${DATASET}
