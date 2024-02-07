import sys
import copy
import datasets
import itertools
import logging
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader


def _get_data(file_path: str, limit: int = None):
    with open(file_path, "r") as fobj:
        data = [json.loads(line) for line in itertools.islice(fobj, limit)]
    dataset = pd.DataFrame(data)
    dataset = dataset[dataset.gold_label != "-"]
    dataset["gold_label"].replace(to_replace="entailment", value=0, inplace=True)
    dataset["gold_label"].replace(to_replace="neutral", value=1, inplace=True)
    dataset["gold_label"].replace(to_replace="contradiction", value=2, inplace=True)


    if "annotator_labels" not in dataset.keys() or "train" in file_path:
        #Exceptional handling for the SICK dataset, which does not have 5 different annotation labels
        dataset["annotator_labels"] = dataset.apply(lambda x: [-1, -1, -1, -1, -1], axis = 1)
        annotator_labels = dataset['annotator_labels'].tolist()

    else:
        annotator_labels = dataset['annotator_labels'].tolist()
        #FIXME: Would be more elegant to handle in dataframe. Key error at 66 in dev.
        #for aid in range(len(dataset["annotator_labels"])):
        #    print("aid", aid)
        #print(dataset["annotator_labels"])
        #    print(file_path, dataset.at[aid,"annotator_labels"])
        #    dataset.at[aid,"annotator_labels"] = [i if i!="entailment" else 0 for i in dataset.at[aid,"annotator_labels"]]
        #    dataset.at[aid,"annotator_labels"] = [i if i!="neutral" else 1 for i in dataset.at[aid,"annotator_labels"]]
        #    dataset.at[aid,"annotator_labels"] = [i if i!="contradiction" else 2 for i in dataset.at[aid,"annotator_labels"]]

        for aid in range(len(dataset["annotator_labels"])):
            annotator_labels[aid] = [i if i!="entailment" else 0 for i in annotator_labels[aid]]
            annotator_labels[aid] = [i if i!="neutral" else 1 for i in annotator_labels[aid]]
            annotator_labels[aid] = [i if i!="contradiction" else 2 for i in annotator_labels[aid]]

            annotator_labels[aid] += [-1] * (5 - len(annotator_labels[aid]))

    return (
            dataset["sentence1"].tolist(),
            dataset["sentence2"].tolist(),
            dataset["gold_label"].tolist(),
            annotator_labels
    )


class NLIDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, annotations):
        self.encodings = encodings
        self.labels = labels
        self.annotations = annotations

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        item["annotations"] = torch.tensor(self.annotations[idx])
        return item

    def __len__(self):
        # return len(self.labels)
        return len(self.encodings.input_ids)


def get_nli_dataset(config, tokenizer):

    if config.dataset:
        logging.info(f"Experiment dataset: {config.dataset}")

        train_premises, train_hypotheses, train_labels, train_annotations = _get_data(
            "data/" + config.dataset + "/train.jsonl", limit=config.train_limit
        )
        logging.info(
            f"First training example: {train_premises[0]} --> {train_hypotheses[0]} ({train_labels[0]})"
        )
        dev_premises, dev_hypotheses, dev_labels, dev_annotations = _get_data(
            "data/" + config.dataset + "/dev.jsonl", limit=config.dev_limit
        )
        logging.info(
            f"First dev example: {dev_premises[0]} --> {dev_hypotheses[0]} ({dev_labels[0]})"
        )
        test_premises, test_hypotheses, test_labels, test_annotations = _get_data(
            "data/" + config.dataset + "/test.jsonl", limit=config.test_limit
        )
        logging.info(
            f"First test example: {test_premises[0]} --> {test_hypotheses[0]} ({test_labels[0]})"
        )
    else:
        raise Exception("Expecting local dataset identifier provided as the --dataset argument (choose from 'mnli-mm', 'mnli-m', 'snli', 'mnli-snli', 'snli-mnli-m', 'snli-mnli-mm', 'snli-sick', 'mnli-sick')")

        #train_dataset = datasets.load_dataset(
        #    "snli", config.train_language, split="train"
        #).filter(lambda sample: sample['label'] >= 0)
        #train_premises = train_dataset["premise"]
        #train_hypotheses = train_dataset["hypothesis"]
        #train_labels = train_dataset["label"]

        #dev_dataset = datasets.load_dataset(
        #    "snli", config.test_language, split="validation"
        #).filter(lambda sample: sample['label'] >= 0)
        #dev_premises = dev_dataset["premise"]
        #dev_hypotheses = dev_dataset["hypothesis"]
        #dev_labels = dev_dataset["label"]

        #test_dataset = datasets.load_dataset("snli", config.test_language, split="test").filter(lambda sample: sample['label'] >= 0)
        #test_premises = test_dataset["premise"]
        #test_hypotheses = test_dataset["hypothesis"]
        #test_labels = test_dataset["label"]

    logging.info("Tokenizing train")
    train_encodings = tokenizer(
        train_premises,
        train_hypotheses,
        truncation=True,
        padding=True,
    )
    logging.info("Tokenizing dev")
    dev_encodings = tokenizer(
        dev_premises, dev_hypotheses, truncation=True, padding=True
    )
    logging.info("Tokenizing test")
    test_encodings = tokenizer(
        test_premises,
        test_hypotheses,
        truncation=True,
        padding=True,
    )

    logging.info("Initializing loaders")
    train_dataset = NLIDataset(train_encodings, train_labels, train_annotations)
    dev_dataset = NLIDataset(dev_encodings, dev_labels, dev_annotations)
    test_dataset = NLIDataset(test_encodings, test_labels, test_annotations)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, dev_loader, test_loader
