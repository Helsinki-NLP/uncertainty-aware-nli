import datasets
import logging
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader


def _get_data(file_path: str):
    data = [json.loads(line) for line in open(file_path, "r")]
    dataset = pd.DataFrame(data)
    dataset = dataset[dataset.gold_label != "-"]
    dataset["gold_label"].replace(to_replace="entailment", value=0, inplace=True)
    dataset["gold_label"].replace(to_replace="neutral", value=1, inplace=True)
    dataset["gold_label"].replace(to_replace="contradiction", value=2, inplace=True)
    return (
        dataset["sentence1"].tolist(),
        dataset["sentence2"].tolist(),
        dataset["gold_label"].tolist(),
    )


class NLIDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        # return len(self.labels)
        return len(self.encodings.input_ids)


def get_nli_dataset(config, tokenizer):

    if config.dataset:
        logging.info(f"Experiment dataset: {config.dataset}")
        
        train_premises, train_hypotheses, train_labels = _get_data(
            "data/" + config.dataset + "/train.jsonl"
        )
        logging.info(
            f"First training example: {train_premises[0]} --> {train_hypotheses[0]} ({train_labels[0]})"
        )
        dev_premises, dev_hypotheses, dev_labels = _get_data(
            "data/" + config.dataset + "/dev.jsonl"
        )
        logging.info(
            f"First dev example: {dev_premises[0]} --> {dev_hypotheses[0]} ({dev_labels[0]})"
        )
        test_premises, test_hypotheses, test_labels = _get_data(
            "data/" + config.dataset + "/test.jsonl"
        )
        logging.info(
            f"First test example: {test_premises[0]} --> {test_hypotheses[0]} ({test_labels[0]})"
        )
    else:
        train_dataset = datasets.load_dataset(
            "snli", config.train_language, split="train"
        ).filter(lambda sample: sample['label'] >= 0)
        train_premises = train_dataset["premise"]
        train_hypotheses = train_dataset["hypothesis"]
        train_labels = train_dataset["label"]

        dev_dataset = datasets.load_dataset(
            "snli", config.test_language, split="validation"
        ).filter(lambda sample: sample['label'] >= 0)
        dev_premises = dev_dataset["premise"]
        dev_hypotheses = dev_dataset["hypothesis"]
        dev_labels = dev_dataset["label"]

        test_dataset = datasets.load_dataset("snli", config.test_language, split="test").filter(lambda sample: sample['label'] >= 0)
        test_premises = test_dataset["premise"]
        test_hypotheses = test_dataset["hypothesis"]
        test_labels = test_dataset["label"]

    train_encodings = tokenizer(
        train_premises,
        train_hypotheses,
        truncation=True,
        padding=True,
    )
    dev_encodings = tokenizer(
        dev_premises, dev_hypotheses, truncation=True, padding=True
    )
    test_encodings = tokenizer(
        test_premises,
        test_hypotheses,
        truncation=True,
        padding=True,
    )

    train_dataset = NLIDataset(train_encodings, train_labels)
    dev_dataset = NLIDataset(dev_encodings, dev_labels)
    test_dataset = NLIDataset(test_encodings, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    return train_loader, dev_loader, test_loader
