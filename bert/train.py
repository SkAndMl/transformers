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

URL = "https://github.com/SK7here/Movie-Review-Sentiment-Analysis/raw/master/IMDB-Dataset.csv"

with open("../config/base_config.json", "r") as f:
    config = json.load(f)

PAD_TOKEN = 0
CLS_TOKEN = 1
SEP_TOKEN = 2
MASK_TOKEN = 3
UNK_TOKEN = 4

def train_masked_lm(bert: BERTMaskedLM,
                    data_loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device="cpu") -> Tuple[List[int], List[int]]:

    losses = []
    accs = []
    bert = bert.to(device)
    bert.train()
    for epoch in range(1, config["train_iters"]+1):
        epoch_loss = 0
        epoch_acc = 0
        for batch, (sentence, masked_token, masked_token_idx) in enumerate(data_loader):
            sentence = sentence.to(device)
            masked_token = masked_token.to(device)
            masked_token_idx = masked_token_idx.to(device)

            logits, loss = bert(sentence, masked_token, masked_token_idx)
            epoch_loss += loss.item()
            epoch_acc += (logits.argmax(dim=-1).squeeze()==masked_token.squeeze()).sum()/config["batch_size"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        epoch_loss /= len(data_loader)
        epoch_acc /= len(data_loader)
        print(f"epoch {epoch}: loss: {epoch_loss:.4f} acc: {round(epoch_acc.item()*100, 2)}%")
        losses.append(epoch_loss)
        accs.append(epoch_acc)

    return losses, accs

print("Loading data")

masked_ds = IMDBMaskedBertDataset(path=URL)
vocab_size = len(masked_ds.vocab)

data_loader = DataLoader(dataset=masked_ds,
                         batch_size=config["batch_size"],
                         shuffle=True)
bert = BERTMaskedLM(config=config,
                    vocab_size=vocab_size)
optimizer = torch.optim.AdamW(params=bert.parameters(),
                             lr=config["learning_rate"])


print("Training model")

losses, accs = train_masked_lm(bert=bert,
                               data_loader=data_loader,
                               optimizer=optimizer)


torch.save(obj=bert.state_dict(),
           f="../weights/bert_masked_lm.pt")

