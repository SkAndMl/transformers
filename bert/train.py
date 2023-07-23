import torch
from torch import nn
import numpy as np
import pandas as pd

from typing import Tuple, List
import json

import sys
sys.path.insert(0, '..')

from data_setup import IMDBMaskedBertDataset
from model import BERTMaskedLM

from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

with open("../config/base_config.json", "r") as f:
    config = json.load(f)

PAD_TOKEN = 0
CLS_TOKEN = 1
SEP_TOKEN = 2
MASK_TOKEN = 3
UNK_TOKEN = 4

def train_masked_lm(bert: BERTMaskedLM,
                    train_data_loader: DataLoader,
                    val_data_loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device="cpu") -> Tuple[List[int], List[int], List[int], List[int]]:

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    bert = bert.to(device)
    for epoch in range(1, config["train_iters"]+1):
        train_loss = 0
        train_acc = 0
        bert.train()
        for batch, (sentence, masked_token, masked_token_idx) in enumerate(train_data_loader):
            sentence = sentence.to(device)
            masked_token = masked_token.to(device)
            masked_token_idx = masked_token_idx.to(device)

            logits, loss = bert(sentence, masked_token, masked_token_idx)
            train_loss += loss.item()
            train_acc += (logits.argmax(dim=-1).squeeze()==masked_token.squeeze()).sum()/config["batch_size"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        test_loss, test_acc = 0, 0
        bert.eval()
        with torch.inference_mode():
            for batch, (sentence, masked_token, masked_token_idx) in enumerate(val_data_loader):
                sentence = sentence.to(device)
                masked_token = masked_token.to(device)
                masked_token_idx = masked_token_idx.to(device)

                logits, loss = bert(sentence, masked_token, masked_token_idx)
                test_loss += loss.item()
                test_acc += (logits.argmax(dim=-1).squeeze()==masked_token.squeeze()).sum()/config["batch_size"]

        

        train_loss /= len(train_data_loader)
        train_acc /= len(train_data_loader)
        test_loss /= len(val_data_loader)
        test_acc /= len(val_data_loader)
        print(f"epoch {epoch}: train_loss: {train_loss:.4f} acc: {round(train_acc.item()*100, 2)}% ",end="")
        print(f"test_loss: {test_loss: .4f} test_acc: {round(test_acc.item()*100, 2)}%")
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    return train_losses, train_accs, test_losses, test_accs

print("Loading data")

train_masked_ds = IMDBMaskedBertDataset(path="./data/train.csv")
test_masked_ds = IMDBMaskedBertDataset(path="./data/test.csv")
vocab_size = max(len(train_masked_ds.vocab), len(test_masked_ds.vocab))

train_data_loader = DataLoader(dataset=train_masked_ds,
                                batch_size=config["batch_size"],
                                shuffle=True)
test_data_loader = DataLoader(dataset=test_masked_ds,
                              batch_size=config["batch_size"],
                              shuffle=True)

bert = BERTMaskedLM(config=config,
                    vocab_size=vocab_size)
optimizer = torch.optim.AdamW(params=bert.parameters(),
                             lr=config["learning_rate"])


print("Training model")

train_losses, train_accs, test_losses, test_accs = train_masked_lm(bert=bert,
                                                                   train_data_loader=train_data_loader,
                                                                   val_data_loader=test_data_loader,
                                                                   optimizer=optimizer,
                                                                   device=device)


torch.save(obj=bert.state_dict(),
           f="../weights/bert_masked_lm.pt")

