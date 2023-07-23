import torch
from torch import nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

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

SPECIALS = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]

tokenizer = get_tokenizer(tokenizer="spacy",
                          language="en_core_web_sm")

def build_vocab(data_iter):
    for sent in data_iter:
        yield tokenizer(sent)

def prepare_vocab():
    df = pd.read_csv("./data/imdb.csv")
    sentences = []
    for i in range(df.shape[0]):
        for sent in df.iloc[i, 0].split('. '):
            sentences.append(sent)
    vocab = build_vocab_from_iterator(build_vocab(sentences),
                                      min_freq=2,
                                      specials=SPECIALS,
                                      special_first=True)
    vocab.set_default_index(UNK_TOKEN)
    return vocab

vocab = prepare_vocab()


def train_step(bert: BERTMaskedLM,
               data_loader: DataLoader,
               optimizer: torch.optim.Optimizer,
               device: torch.device="cpu") -> Tuple[float ,float]:
    
    loss, acc = 0, 0
    bert.train()
    for batch, (sentence, masked_token, masked_token_idx) in enumerate(data_loader):
        sentence = sentence.to(device)
        masked_token = masked_token.to(device)
        masked_token_idx = masked_token_idx.to(device)

        logits, loss_ = bert(sentence, masked_token, masked_token_idx)
        loss += loss_.item()
        acc += (logits.argmax(dim=-1).squeeze()==masked_token.squeeze()).sum()/config["batch_size"]

        optimizer.zero_grad(set_to_none=True)
        loss_.backward()
        optimizer.step()
    
    loss /= len(data_loader)
    acc /= len(data_loader)
    return loss, acc.item()


def test_step(bert: BERTMaskedLM,
              data_loader: DataLoader,
              device: torch.device="cpu") -> Tuple[float, float]:
    
    loss, acc = 0, 0
    bert.eval()
    with torch.inference_mode():
        for batch, (sentence, masked_token, masked_token_idx) in enumerate(data_loader):
            sentence = sentence.to(device)
            masked_token = masked_token.to(device)
            masked_token_idx = masked_token_idx.to(device)

            logits, loss_ = bert(sentence, masked_token, masked_token_idx)
            loss += loss_.item()
            acc += (logits.argmax(dim=-1).squeeze()==masked_token.squeeze()).sum()/config["batch_size"]
    
    loss /= len(data_loader)
    new_acc = acc.item()/len(data_loader)
    return loss, new_acc


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
        train_loss, train_acc = train_step(bert, train_data_loader, optimizer, device)
        test_loss, test_acc = test_step(bert, val_data_loader, device)
        print(f"epoch {epoch}: train_loss: {train_loss:.4f} acc: {round(train_acc*100, 2)}% ",end="")
        print(f"test_loss: {test_loss: .4f} test_acc: {round(test_acc*100, 2)}%")
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    return train_losses, train_accs, test_losses, test_accs

print("Loading data")

train_masked_ds = IMDBMaskedBertDataset(path="./data/train.csv",
                                        vocab=vocab,
                                        tokenizer=tokenizer)
test_masked_ds = IMDBMaskedBertDataset(path="./data/test.csv",
                                       vocab=vocab,
                                       tokenizer=tokenizer)
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")
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


