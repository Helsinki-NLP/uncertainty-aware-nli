import torch.nn as nn
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    RobertaTokenizer,
    RobertaForSequenceClassification,
)

#+++HANDE

MODEL_SPECS = {
    'bert': {'model_cls': BertForSequenceClassification, 'tokenizer_cls': BertTokenizer, 
            'model_subtype': 'bert-base-multilingual-cased', 'tokenizer_subtype': 'bert-base-cased'},
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
    else:
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base", num_labels=3
        )
    return tokenizer, model


#+++HANDE

class LangModel(nn.Module):
    def __init__(self, num_labels=3, model_cls = BertForSequenceClassification, tokenizer_cls = BertTokenizer, 
                                   model_subtype='bert-base-multilingual-cased', tokenizer_subtype='bert-base-cased'):
        super(LangModel, self).__init__()

        self.lm = model_cls.from_pretrained(model_subtype, num_labels=num_labels)
        self.tokenizer = tokenizer_cls.from_pretrained(tokenizer_subtype)


    def get_model(self):
        return self.lm

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, **kwargs):
        return self.lm(**kwargs)       


def get_model_specs(model_type):
    return MODEL_SPECS[model_type]

#---HANDE