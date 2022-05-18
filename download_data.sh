#!/bin/bash

mkdir -p data/
cd data/

mkdir -p mnli-m
mkdir -p mnli-mm
mkdir -p snli
mkdir -p mnli-snli
mkdir -p snli-mnli-m
mkdir -p snli-mnli-mm
mkdir -p snli-sick
mkdir -p mnli-sick

wget https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip
wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
wget https://zenodo.org/record/2787612/files/SICK.zip

unzip multinli_1.0.zip
unzip snli_1.0.zip
unzip SICK.zip

cp multinli_1.0/multinli_1.0_train.jsonl mnli-m/train.jsonl
cp multinli_1.0/multinli_1.0_dev_matched.jsonl mnli-m/dev.jsonl
cp multinli_1.0/multinli_1.0_dev_matched.jsonl mnli-m/test.jsonl

cp multinli_1.0/multinli_1.0_train.jsonl mnli-mm/train.jsonl
cp multinli_1.0/multinli_1.0_dev_mismatched.jsonl mnli-mm/dev.jsonl
cp multinli_1.0/multinli_1.0_dev_mismatched.jsonl mnli-mm/test.jsonl

cp snli_1.0/snli_1.0_train.jsonl snli/train.jsonl
cp snli_1.0/snli_1.0_dev.jsonl snli/dev.jsonl
cp snli_1.0/snli_1.0_test.jsonl snli/test.jsonl

cp snli_1.0/snli_1.0_train.jsonl snli-mnli-m/train.jsonl
cp snli_1.0/snli_1.0_dev.jsonl snli-mnli-m/dev.jsonl
cp multinli_1.0/multinli_1.0_dev_matched.jsonl snli-mnli-m/test.jsonl

cp snli_1.0/snli_1.0_train.jsonl snli-mnli-mm/train.jsonl
cp snli_1.0/snli_1.0_dev.jsonl snli-mnli-mm/dev.jsonl
cp multinli_1.0/multinli_1.0_dev_mismatched.jsonl snli-mnli-mm/test.jsonl

cp multinli_1.0/multinli_1.0_train.jsonl mnli-snli/train.jsonl
cp multinli_1.0/multinli_1.0_dev_mismatched.jsonl mnli-snli/dev.jsonl
cp snli_1.0/snli_1.0_test.jsonl snli-mnli-m/test.jsonl

python3 prepare_sick_data.py

cp multinli_1.0/multinli_1.0_train.jsonl mnli-sick/train.jsonl
cp multinli_1.0/multinli_1.0_dev_mismatched.jsonl mnli-sick/dev.jsonl
cp sick.jsonl mnli-sick/test.jsonl

cp snli_1.0/snli_1.0_train.jsonl snli-sick/train.jsonl
cp snli_1.0/snli_1.0_dev.jsonl snli-sick/dev.jsonl
cp sick.jsonl snli-sick/test.jsonl

rm sick.jsonl
rm SICK.txt
rm SICK.zip
rm readme.txt

rm multinli_1.0.zip
rm snli_1.0.zip
rm -r multinli_1.0
rm -r snli_1.0
cd ..