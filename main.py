from argparse import ArgumentParser
from data import get_nli_dataset
import logging
import models
import numpy as np
from pathlib import Path
import random
import torch
from tqdm import tqdm
from transformers import AdamW
from torch.optim import Adam, SGD
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
import time


parser = ArgumentParser(description="NLI with Transformers")

parser.add_argument("--train_language", type=str, default=None)
parser.add_argument("--test_language", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--early_stopping", type=int, default=3)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--log_every", type=int, default=100)
parser.add_argument("--method", type=str, choices=["swa", "no-avg"], default="swa")
parser.add_argument("--gpu", type=int, default=None)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument(
    "--dataset",
    type=str,
    choices=[
        "mnli-mm",
        "mnli-m",
        "snli",
        "mnli-snli",
        "snli-mnli-m",
        "snli-mnli-mm",
        "snli-sick",
        "mnli-sick",
    ],
    default="mnli-mm",
)
parser.add_argument("--optimizer", type=str, default="AdamW")
parser.add_argument(
    "--model",
    type=str,
    choices=["bert", "roberta"],
    default="roberta",
)

logging.basicConfig(level=logging.INFO)


def train(config, train_loader, model, optim, device, epoch):
    logging.info("Starting training...")
    model.train()
    logging.info(f"Epoch: {epoch + 1}/{config.epochs}")
    for i, batch in enumerate(train_loader):
        optim.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()
        if i == 0 or i % config.log_every == 0 or i + 1 == len(train_loader):
            logging.info(
                "Epoch: {} - Progress: {:3.0f}% - Batch: {:>4.0f}/{:<4.0f} - Loss: {:<.4f}".format(
                    epoch + 1,
                    100.0 * (1 + i) / len(train_loader),
                    i + 1,
                    len(train_loader),
                    loss.item(),
                )
            )


def evaluate(model, dataloader, device):
    logging.info("Starting evaluation...")
    model.eval()
    with torch.no_grad():
        eval_preds = []
        eval_labels = []

        for batch in tqdm(dataloader, total=len(dataloader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            preds = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = preds[0]
            preds = preds[1].argmax(dim=-1)
            eval_preds.append(preds.cpu().numpy())
            eval_labels.append(batch["labels"].cpu().numpy())

    logging.info("Done evaluation")
    return np.concatenate(eval_labels), np.concatenate(eval_preds), loss.item()


def main():

    config = parser.parse_args()

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    device = (
        torch.device(f"cuda:{config.gpu}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    logging.info(f"Training on {device}.")

    tokenizer, model = models.get_model(config)

    train_loader, dev_loader, test_loader = get_nli_dataset(config, tokenizer)

    logging.info(f"Optimizer {config.optimizer}")
    if config.optimizer == "Adam":
        optim = Adam(model.parameters(), lr=0.00002)
    elif config.optimizer == "SGD":
        optim = SGD(model.parameters(), lr=0.01, momentum=0.9)
    else:
        optim = AdamW(model.parameters(), lr=0.00002)

    model.to(device)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_dir = f"output/{config.dataset}/{timestr}"

    if config.method == "swa":
        logging.info("SWA training")
        swa_model = AveragedModel(model)
        scheduler = CosineAnnealingLR(optim, T_max=5)
        swa_start = 1
        swa_scheduler = SWALR(optim, swa_lr=0.05)
        output_dir = f"{output_dir}-swa"
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    start = time.time()

    early_stopping = config.early_stopping
    best_loss = 999
    stopped_after = config.early_stopping

    for epoch in range(config.epochs):
        train(config, train_loader, model, optim, device, epoch)

        if config.method == "swa":
            if epoch > swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step()

        dev_labels, dev_preds, dev_loss = evaluate(model, dev_loader, device)

        dev_accuracy = (dev_labels == dev_preds).mean()

        logging.info(f"Dev loss after epoch {epoch+1}: {dev_loss:<.4f}")
        logging.info(f"Previous best: {best_loss:<.4f}")

        snapshot_path = f"{output_dir}/{config.model}-{config.dataset}_snapshot_epoch_{epoch+1}_devacc_{round(dev_accuracy, 3)}.pt"
        torch.save(model, snapshot_path)

        if dev_loss < best_loss:
            best_loss = dev_loss
            early_stopping = config.early_stopping
        else: 
            early_stopping = early_stopping - 1

        if early_stopping == 0:
            logging.info(f"Stopping early after {epoch+1}/{config.epochs} epochs.")
            stopped_after = epoch+1
            break

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)

    if config.method == "swa":
        torch.optim.swa_utils.update_bn(train_loader, swa_model)
        test_labels, test_preds = evaluate(swa_model, test_loader, device)
    else:
        test_labels, test_preds, test_loss = evaluate(model, test_loader, device)

    test_accuracy = (test_labels == test_preds).mean()

    with open(
        f"{output_dir}/predictions.tsv",
        "w",
    ) as predictions_file:
        predictions_file.write(f"prediction\tlabel")
        for pred, labl in zip(test_preds, test_labels):
            predictions_file.write(f"{pred}\t{labl}")

    logging.info(f"=== SUMMARY ===")
    logging.info(f"Model: {model.__class__.__name__}")
    logging.info(f"Optimizer {config.optimizer}")
    logging.info(f"Method: {config.method}")
    logging.info(f"Epochs: {stopped_after}/{config.epochs}")
    logging.info(f"Batch size: {config.batch_size}")
    logging.info(f"Training time: {int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}")
    logging.info(f"Test accuracy: {test_accuracy}")

    with open(
        f"{output_dir}/results.txt",
        "w",
    ) as resultfile:
        resultfile.write(f"Dataset: {config.dataset}\n")
        resultfile.write(f"Model: {model.__class__.__name__}\n")
        resultfile.write(f"Optimizer {config.optimizer}\n")
        resultfile.write(f"Method: {config.method}\n")
        resultfile.write(f"Epochs:  {stopped_after}/{config.epochs}\n")
        resultfile.write(f"Batch size: {config.batch_size}\n")
        resultfile.write(
            f"Training time: {int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}\n"
        )
        resultfile.write(f"Test accuracy: {test_accuracy}\n\n")

    with open("output/result_summary.tsv", "a") as summary_results:
        summary_results.write(
            f"{config.dataset}\t{model.__class__.__name__}\t{config.optimizer}\t{config.method}\t{stopped_after}\t{config.batch_size}\t{int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}\t{test_accuracy}\n"
        )

    final_snapshot_path = f"{output_dir}/{config.model}-{config.dataset}_final_snapshot_epochs_{stopped_after}_devacc_{round(dev_accuracy, 3)}.pt"
    torch.save(model, final_snapshot_path)


if __name__ == "__main__":
    main()
