import itertools
import torch
import os
import copy
from datetime import datetime
import math
import numpy as np
import tqdm

import torch.nn.functional as F

from swa_gaussian.swag.utils import *


def train_epoch(
    loader,
    model,
    optimizer,
    cuda=True,
    regression=False,
    verbose=False,
    subset=None,
):
    loss_sum = 0.0
    correct = 0.0
    verb_stage = 0

    num_objects_current = 0
    num_batches = len(loader)

    model.train()

    if subset is not None:
        num_batches = int(num_batches * subset)
        loader = itertools.islice(loader, num_batches)

    if verbose:
        loader = tqdm.tqdm(loader, total=num_batches)

    for i, batch in enumerate(loader):
        if cuda:
            for key in batch.keys():
                batch[key] = batch[key].cuda()
        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        loss_sum += loss.data.item() * batch['input_ids'].size(0)
        if not regression:
            pred = outputs.logits.argmax(1, keepdim=True)
            correct += pred.eq(batch['labels'].view_as(pred)).sum().item()
        num_objects_current += batch['input_ids'].size(0)

        if verbose and 10 * (i + 1) / num_batches >= verb_stage + 1:
            print(
                "Stage %d/10. Loss: %12.4f. Acc: %6.2f"
                % (
                    verb_stage + 1,
                    loss_sum / num_objects_current,
                    correct / num_objects_current * 100.0,
                )
            )
            verb_stage += 1

    return {
        "loss": loss_sum / num_objects_current,
        "accuracy": None if regression else correct / num_objects_current * 100.0,
    }


def eval(loader, model, cuda=True, regression=False, verbose=False):
    loss_sum = 0.0
    correct = 0.0
    num_objects_total = len(loader.dataset)

    model.eval()

    with torch.no_grad():
        if verbose:
            loader = tqdm.tqdm(loader)
        for i, batch in enumerate(loader):
            if cuda:
                for key in batch.keys():
                    batch[key] = batch[key].cuda()
            outputs = model(**batch)
            loss = outputs.loss
    
            loss_sum += loss.data.item() * batch['input_ids'].size(0)
            if not regression:
                pred = outputs.logits.argmax(1, keepdim=True)
                correct += pred.eq(batch['labels'].view_as(pred)).sum().item()


    return {
        "loss": loss_sum / num_objects_total,
        "accuracy": None if regression else correct / num_objects_total * 100.0,
    }


def predict(loader, model, verbose=False):
    predictions = list()
    targets = list()

    model.eval()

    if verbose:
        loader = tqdm.tqdm(loader)

    offset = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if cuda:
                for key in batch.keys():
                    batch[key] = batch[key].cuda()
            outputs = model(**batch)
            loss = outputs.loss

            batch_size = batch['input_ids'].size(0)
            predictions.append(F.softmax(outputs.logits, dim=1).cpu().numpy())
            targets.append(batch['labels'].numpy())
            offset += batch_size

    return {"predictions": np.vstack(predictions), "targets": np.concatenate(targets)}
