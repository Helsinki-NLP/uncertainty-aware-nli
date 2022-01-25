for model in bert roberta
do
    sbatch train.sh $model
done
