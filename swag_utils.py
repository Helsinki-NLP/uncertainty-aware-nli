import itertools

import numpy as np
import tqdm
import torch
import torch.nn.functional as F

from swag.utils import *


EPSILON = 1e-12


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

        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
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

        #break #DEBUG

    return {
        "loss": loss_sum / num_objects_current,
        "accuracy": None if regression else correct / num_objects_current * 100.0,
    }


def eval(test_loader, train_loader, swag_model, num_samples, is_cov_mat, scale, is_blockwise, cuda=True):
    swag_predictions_accum = np.zeros((len(test_loader.dataset), len(set(test_loader.dataset.labels))))
    swag_predictions_history = np.zeros((num_samples, len(test_loader.dataset), len(set(test_loader.dataset.labels))))
    for i in range(num_samples):
        swag_model.sample(scale, cov=is_cov_mat, block=is_blockwise)   #and (not args.use_diag_bma))

        print("SWAG Sample %d/%d. BN update" % (i + 1, num_samples))
        bn_update(train_loader, swag_model, verbose=True, subset=0.1)
        print("SWAG Sample %d/%d. EVAL" % (i + 1, num_samples))
        res = predict(test_loader, swag_model, cuda=cuda, verbose=True)
        predictions = res["predictions"]
        labels = res["labels"]
        annotations = res["annotations"]
        ids = res["ids"]

        accuracy = np.mean(np.argmax(predictions, axis=1) == labels)
        nll = -np.mean(np.log(predictions[np.arange(predictions.shape[0]), labels] + EPSILON))
        print(
            "SWAG Sample %d/%d. Accuracy: %.2f%% NLL: %.4f"
            % (i + 1, num_samples, accuracy * 100, nll)
        )

        swag_predictions_accum += predictions
        swag_predictions_history[i,:,:] = predictions

        ens_accuracy = np.mean(np.argmax(swag_predictions_accum, axis=1) == labels)
        ens_nll = -np.mean(
            np.log(
                swag_predictions_accum[np.arange(swag_predictions_accum.shape[0]), labels] / (i + 1)
                + EPSILON
            )
        )
        print(
            "Ensemble %d/%d. Accuracy: %.2f%% NLL: %.4f"
            % (i + 1, num_samples, ens_accuracy * 100, ens_nll)
        )

    swag_predictions_accum /= num_samples

    swag_accuracy = np.mean(np.argmax(swag_predictions_accum, axis=1) == labels)
    swag_confidences = np.argmax(swag_predictions_accum, axis=1)
    swag_nll = -np.mean(
        np.log(swag_predictions_accum[np.arange(swag_predictions_accum.shape[0]), labels] + EPSILON)
    )
    swag_entropies = -np.sum(np.log(swag_predictions_accum + EPSILON) * swag_predictions_accum, axis=1)

    return {
        "loss": swag_nll,
        "accuracy": swag_accuracy * 100,
        "confidences": swag_confidences,
        "nll": swag_nll,
        "entropies": swag_entropies,
        "predictions": swag_predictions_history,
        "labels": labels,
        "annotations": annotations,
        "ids": ids
    }


def predict(loader, model, cuda=True, verbose=False):
    predictions = list()
    labels = list()
    annotations = list()
    ids = list()

    model.eval()

    if verbose:
        loader = tqdm.tqdm(loader)

    offset = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if cuda:
                for key in batch.keys():
                    batch[key] = batch[key].cuda()

            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            loss = outputs.loss

            batch_size = batch["input_ids"].size(0)

            predictions.append(F.softmax(outputs.logits, dim=1).cpu().numpy())
            labels.append(batch["labels"].cpu().numpy())
            annotations.append(batch["annotations"].cpu().numpy())
            ids.append(batch["input_ids"].cpu().numpy())

            offset += batch_size

    return {"predictions": np.concatenate(predictions), "labels": np.concatenate(labels),
            "annotations": np.concatenate(annotations), "ids": np.concatenate(ids)}
