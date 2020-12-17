import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader
import datasets
from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser(description='NLi Transformers')

parser.add_argument('--batch_size',
                    type=int,
                    default=32)
parser.add_argument('--epochs',
                    type=int,
                    default=2)
parser.add_argument('--log_every',
                    type=int,
                    default=10)
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.00005)
parser.add_argument('--weight_decay',
                    type=float,
                    default=0)
parser.add_argument('--gpu',
                    type=int,
                    default=None)
parser.add_argument('--fraction_of_train_data',
                    type=float,
                    default=1
                    )
parser.add_argument('--seed',
                    type=int,
                    default=1234)
parser.add_argument('--dataset',
                    type=str,
                    choices=['multi_nli',
                             'snli'],
                    default='multi_nli')


class NLIDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        #return len(self.labels)
        return len(self.encodings.input_ids)


def get_nli_dataset(config, tokenizer):
    
    nli_data = datasets.load_dataset(config.dataset)

    # For testing purposes get a slammer slice of the training data
    num_examples = int(round(len(nli_data['train']['label']) * config.fraction_of_train_data))
    
    train_dataset = nli_data['train'][:num_examples]

    if config.dataset == "multi_nli":
        validation_data = 'validation_matched'
        test_data = 'validation_matched'
    else:
        validation_data = 'validation'
        test_data = 'test'

    dev_dataset = nli_data[validation_data]
    test_dataset = nli_data[test_data]

    train_labels = train_dataset['label']

    val_labels = dev_dataset['label']
    test_labels = test_dataset['label']

    train_encodings = tokenizer(train_dataset['premise'], train_dataset['hypothesis'], truncation=True, padding=True)
    val_encodings = tokenizer(dev_dataset['premise'], dev_dataset['hypothesis'], truncation=True, padding=True)
    test_encodings = tokenizer(test_dataset['premise'], test_dataset['hypothesis'], truncation=True, padding=True)

    train_dataset = NLIDataset(train_encodings, train_labels)
    val_dataset = NLIDataset(val_encodings, val_labels)
    test_dataset = NLIDataset(test_encodings, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    return train_loader, dev_loader, test_loader


def train(train_loader, model, optim, device, epochs=3):
    print("Starting training...")
    model.train()
    for epoch in range(epochs):
        print("Epoch: {}/{}".format(epoch+1, epochs))
        for i, batch in enumerate(train_loader):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()
            if i == 0 or i % 10 == 0 or i+1 == len(train_loader):
                print("Progress: {:3.0f}% - Batch: {:>4.0f}/{:<4.0f} - Loss: {:<.4f}".format(
                    100. * (1+i) / len(train_loader), # Progress
                    i+1, len(train_loader), # Batch
                    loss.item())) # Loss
        snapshot_path = 'model_snapshot_epoch_{}.pt'.format(epoch)
        torch.save(model, snapshot_path)

    print("Done training!")


def evaluate(model, dataloader, device):
    print("\nStarting evaluation...")
    model.eval()
    with torch.no_grad():
        eval_preds = []
        eval_labels = []

        for batch in tqdm(dataloader, total=len(dataloader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            preds = model(input_ids, attention_mask=attention_mask, labels=labels)
            preds = preds[1].argmax(dim=-1)
            eval_preds.append(preds.cpu().numpy())
            eval_labels.append(batch['labels'].cpu().numpy())

    print("Done evaluation")
    return np.concatenate(eval_labels), np.concatenate(eval_preds)


def main():

    config = parser.parse_args()

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    if config.gpu is not None:
        torch.cuda.set_device(config.gpu)
        device = torch.device('cuda:{}'.format(config.gpu))
        torch.cuda.manual_seed(config.seed)
        print("\nTraining on GPU[{}] (torch.device({})).".format(config.gpu, device))
    else:
        device = torch.device('cpu')
        print("GPU not available so training on CPU (torch.device({})).".format(device))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_loader, dev_loader, test_loader = get_nli_dataset(config, tokenizer)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    optim = AdamW(model.parameters(), lr=5e-5)
    model.to(device)

    train(train_loader, model, optim, device, epochs=config.epochs)
    labels, preds = evaluate(model, dev_loader, device)
    accuracy = (labels == preds).mean()

    final_snapshot_path = 'model_final_snapshot_acc_{}_epochs_{}.pt'.format(accuracy, epoch)
    torch.save(model, snapshot_path)

    print("\nAccuracy: {}".format(accuracy))


if __name__ == '__main__':
    main()