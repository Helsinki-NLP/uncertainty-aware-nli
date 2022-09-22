#!/bin/bash

for dataset in  mnli-m mnli-mm snli mnli-snli snli-mnli-m snli-mnli-mm snli-sick mnli-sick; do
    for optimizer in Adam AdamW SGD; do
        for method in swa swag no-avg; do
            sbatch train.sh $dataset $optimizer $method
        done
    done
done
