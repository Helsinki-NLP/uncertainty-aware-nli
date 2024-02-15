import torch.nn as nn
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    DebertaV2Tokenizer,
    DebertaV2ForSequenceClassification
)

#+++HANDE

MODEL_SPECS = {
    'bert': {'model_cls': BertForSequenceClassification, 'tokenizer_cls': BertTokenizer,
            'model_subtype': 'bert-base-multilingual-cased', 'tokenizer_subtype': 'bert-base-cased'},
    'deberta_v2': {'model_cls': DebertaV2ForSequenceClassification,  'tokenizer_cls': DebertaV2Tokenizer,
               'model_subtype': 'microsoft/deberta-v2-xlarge', 'tokenizer_subtype': 'microsoft/deberta-v2-xlarge'},
    'roberta': {'model_cls': RobertaForSequenceClassification,  'tokenizer_cls': RobertaTokenizer,
               'model_subtype': 'roberta-base', 'tokenizer_subtype': 'roberta-base'}
}

#---HANDE

def get_model(config):
    if config.model == "bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-multilingual-cased", num_labels=3
        )
    elif config.model == "deberta_v2":
        tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v2-xlarge") # hf-internal-testing/tiny-random-deberta-v2")
        model = DebertaV2ForSequenceClassification.from_pretrained("microsoft/deberta-v2-xlarge", num_labels=3) # "hf-internal-testing/tiny-random-deberta-v2")
    else:
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base", num_labels=3
        )
    return tokenizer, model


#+++HANDE

class LangModel(nn.Module):
    def __init__(self, num_labels=3, model_cls=BertForSequenceClassification, tokenizer_cls=BertTokenizer,
                 model_subtype='bert-base-multilingual-cased', tokenizer_subtype='bert-base-cased'):
        super(LangModel, self).__init__()

        self.lm = model_cls.from_pretrained(model_subtype, num_labels=num_labels)
        self.tokenizer = tokenizer_cls.from_pretrained(tokenizer_subtype)


    def get_model(self):
        return self.lm

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, *args, **kwargs):
        return self.lm(*args, **kwargs)


def get_model_specs(model_type):
    return MODEL_SPECS[model_type]

#---HANDE
