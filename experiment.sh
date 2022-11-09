#!/bin/bash

for dataset in mnli-m mnli-mm snli mnli-snli snli-mnli-m snli-mnli-mm snli-sick mnli-sick; do
    for optimizer in AdamW; do
        for seed in 1011 2022 3033 4044 5055; do
            sbatch train.sh $dataset $optimizer $seed
        done
    done
done
