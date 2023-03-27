#!/bin/bash

for dataset in mnli-m mnli-mm snli mnli-snli snli-mnli-m snli-mnli-mm snli-sick mnli-sick; do
    for method in swa; do
        sbatch train.sh $dataset $method
    done
done
