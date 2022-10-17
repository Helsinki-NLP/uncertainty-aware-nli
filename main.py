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

import swag_utils
from swa_gaussian.swag.posteriors.swag import SWAG

parser = ArgumentParser(description="NLI with Transformers")

parser.add_argument("--train_language", type=str, default=None)
parser.add_argument("--test_language", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--early_stopping", type=int, default=3)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--log_every", type=int, default=100)
parser.add_argument("--method", type=str, choices=["swa", "swag", "no-avg"], default="no-swa")
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
    ]
)
parser.add_argument("--optimizer", type=str, default="AdamW")
parser.add_argument(
    "--model",
    type=str,
    choices=["bert", "roberta"],
    default="roberta",
)

#+++HANDE
parser.add_argument("--swa_start", type=int, default=1)
parser.add_argument("--num_labels", type=int, default=3)
parser.add_argument("--cov_mat", action="store_true", help="save sample covariance")

parser.add_argument(
    "--max_num_models",
    type=int,
    default=20,
    help="maximum number of SWAG models to save",
)

#---HANDE

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

    #+++HANDE
    if config.cov_mat:
        config.no_cov_mat = False
    else:
        config.no_cov_mat = True
    
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{config.gpu}")
        use_cuda = True

    else:
        device = torch.device("cpu")
        use_cuda = False

    #---HANDE


    logging.info(f"Training on {device}.")


    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_dir = f"output/{config.dataset}/{timestr}"

    #+++HANDE
    if config.method in ["no-avg", "swa"]:
        tokenizer, model = models.get_model(config)

    elif config.method == "swag":
        model_specs = models.get_model_specs(config.model)
        model = models.LangModel(num_labels=config.num_labels,
            model_cls=model_specs['model_cls'],
            model_subtype=model_specs['model_subtype'],
            tokenizer_cls=model_specs['tokenizer_cls'],
            tokenizer_subtype=model_specs['tokenizer_subtype'])
        tokenizer = model.get_tokenizer()

    train_loader, dev_loader, test_loader = get_nli_dataset(config, tokenizer)
    
    logging.info(f"Optimizer {config.optimizer}")

    if config.optimizer == "Adam":
        optim = Adam(model.parameters(), lr=0.00002)
    elif config.optimizer == "SGD":
        optim = SGD(model.parameters(), lr=0.01, momentum=0.9)
    else:
        optim = AdamW(model.parameters(), lr=0.00002)

    model.to(device)
    

    #---HANDE

    if config.method == "swa":
        logging.info("SWA training")
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optim, swa_lr=0.05)
 
    #+++HANDE
    elif config.method == "swag":
        # initialize SWAG
        logging.info("SWAG training")
        model_specs = models.get_model_specs(config.model)
        swag_model = SWAG(
            models.LangModel,
            no_cov_mat=config.no_cov_mat,
            max_num_models=config.max_num_models,
            num_labels=config.num_labels, 
            model_cls=model_specs['model_cls'],
            model_subtype=model_specs['model_subtype'],
            tokenizer_cls=model_specs['tokenizer_cls'],
            tokenizer_subtype=model_specs['tokenizer_subtype']
        )
        swag_model.to(device) 

    else: #config.method == "no-avg"
        # raise not implemented error
        pass

    output_dir = f"{output_dir}-{config.method}"
    #---HANDE
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    start = time.time()

    early_stopping = config.early_stopping
    best_loss = 999
    stopped_after = config.early_stopping

    
    for epoch in range(config.epochs):
        logging.info(f"Epoch {epoch}")

        if config.method == "swa":
            train(config, train_loader, model, optim, device, epoch)
            if epoch > swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step()

        #+++HANDE
        elif config.method == "swag":
            swag_utils.train_epoch(train_loader, model, optim, cuda=use_cuda)
            if epoch > swa_start:
                swag_model.collect_model(model)
                if (
                    epoch == 0
                    or epoch % args.eval_freq == args.eval_freq - 1
                    or epoch == args.epochs - 1
                ):
                    swag_model.sample(0.0)
                    swag_utils.bn_update(train_loader, swag_model)
                    swag_res = utils.eval(dev_loader, swag_model)
                else:
                    swag_res = {"loss": None, "accuracy": None}

            else:
                #scheduler.step()
                pass

        if config.method == "swa":
            dev_labels, dev_preds, dev_loss = evaluate(model, dev_loader, device)
            dev_accuracy = (dev_labels == dev_preds).mean()
        elif config.method == "swag":
            dev_res = utils.eval(dev_loader, model, cuda=use_cuda)
            dev_loss = dev_res["loss"]
            dev_accuracy = dev_res["accuracy"]

        #---HANDE    

        logging.info(f"Dev accuracy after epoch {epoch+1}: {dev_accuracy}")
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
        test_labels, test_preds, test_loss = evaluate(swa_model, test_loader, device)
        test_accuracy = (test_labels == test_preds).mean()
    #+++HANDE
    elif config.method == "swag":
        swag_utils.bn_update(train_loader, swag_model)
        test_res = utils.eval(test_loader, model, cuda=use_cuda)
        test_loss = test_res["loss"]
        test_accuracy = test_res["accuracy"]
    #---HANDE
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

    with open(f"output/result_summary.csv", "a") as summary_results:
        summary_results.write(
            f"{config.dataset}\t{model.__class__.__name__},{config.optimizer},{config.method},{stopped_after},{config.batch_size},{int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f},{test_accuracy}\n"
        )

    final_snapshot_path = f"{output_dir}/{config.model}-{config.dataset}_final_snapshot_epochs_{stopped_after}_devacc_{round(dev_accuracy, 3)}.pt"
    torch.save(model, final_snapshot_path)


if __name__ == "__main__":
    main()
