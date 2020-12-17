# NLI with Transformers
Fine-tune [transformers](https://huggingface.co/transformers/) with NLI data. 

Support the following models:
* [`BartForSequenceClassification`](https://huggingface.co/transformers/model_doc/bart.html#bartforsequenceclassification)
* [`BertForSequenceClassification`](https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification)
* [`DistilBertForSequenceClassification`](https://huggingface.co/transformers/model_doc/distilbert.html#distilbertforsequenceclassification)
* [`GPT2ForSequenceClassification`](https://huggingface.co/transformers/model_doc/gpt2.html#gpt2forsequenceclassification)
* [`RobertaForSequenceClassification`](https://huggingface.co/transformers/model_doc/roberta.html#robertaforsequenceclassification)
* [`XLNetForSequenceClassification`](https://huggingface.co/transformers/model_doc/xlnet.html#xlnetforsequenceclassification)

## Requirements

Install requirements by running:
```console
pip3 install -r requirements.txt
```

## Train & Evaluate

To fine-tune the DistilBERT model, run the following:

```console
python3 main.py \
    --model distilbert \
    --dataset multi_nli \
    --batch_size 32 \
    --epochs 4 \
    --log_every 50 \
    --learning_rate 0.00005 \
    --gpu 0 \
    --seed 1234
```
