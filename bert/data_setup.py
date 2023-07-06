import torch
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import pandas as pd
import random
from typing import Tuple


class IMDBBERTDataset(Dataset):

    NSP_PERCENTAGE = 0.50

    CLS_TOKEN = 0
    SEP_TOKEN = 1
    MASK_TOKEN = 2
    PAD_TOKEN = 3 
    UNK_TOKEN = 4

    def __init__(self,
                 path: str,
                 max_sent_len: int=50) -> None:
        super().__init__()
        self.df = pd.read_csv(path)
        self.tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
        self.vocab = build_vocab_from_iterator(self._build_vocab(self.df["review"].to_list()),
                                               min_freq=2,
                                               specials=["[CLS]", "[SEP]", "[MASK]", "[PAD]", "<UNK>"],
                                               special_first=True)
        self.vocab.set_default_index(self.UNK_TOKEN)
        self.max_sent_len = max_sent_len
        self.token_ids = []
        self.masked_token = []
        self.masked_idx = []
        self.is_next = []
        self.segment_tokens = []
        self._prepare_data()
        

    def _prepare_data(self) -> None:
        for i in range(self.df.shape[0]):
            try:
                sentences = self.df.iloc[i, 0].split(". ")
                if random.random() <= self.NSP_PERCENTAGE:
                    rand_idx = random.randint(0, len(sentences)-2)
                    sentences = sentences[rand_idx:rand_idx+2]
                    is_next = 0
                else:
                    rand_idx = random.randint(1, len(sentences)-1)
                    sentences = [sentences[rand_idx], sentences[rand_idx-1]]
                    is_next = 1
                
                sentences = ["[CLS]"] + self.tokenizer(sentences[0]) + ["[SEP]"] + self.tokenizer(sentences[1])
                if len(sentences) < self.max_sent_len:
                    while len(sentences) < self.max_sent_len:
                        sentences += ["[PAD]"]
                else:
                    sentences = sentences[:self.max_sent_len]
                
                sep_idx = sentences.index("[SEP]")
                segment_token = [0]*(sep_idx+1) + [1]*(len(sentences)-1-sep_idx)

                # assert len(segment_token) == len(sentences), f"Length not equal, sep_idx: {sep_idx} "

                token_ids = self.vocab(sentences)
                mask_token, mask_idx = self.SEP_TOKEN, -1
                while mask_token == self.SEP_TOKEN:
                    mask_idx = random.randint(1, len(token_ids)-1)
                    mask_token = token_ids[mask_idx]
                token_ids[mask_idx] = self.MASK_TOKEN
                self.token_ids.append(token_ids)
                self.masked_token.append(mask_token)
                self.masked_idx.append(mask_idx)
                self.segment_tokens.append(segment_token)
                self.is_next.append(is_next)
            except:
                pass
 
        self.bert_df = pd.DataFrame(data={
            "token_ids" : self.token_ids,
            "segment_tokens" : self.segment_tokens,
            "masked_token" : self.masked_token,
            "masked_idx" : self.masked_idx,
            "is_next" : self.is_next
        })
        
    def _build_vocab(self, data_iter):
        for sentence in data_iter:
            yield self.tokenizer(sentence)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        token_ids = self.token_ids[index]
        segment_tokens = self.segment_tokens[index]
        masked_token = self.masked_token[index]
        masked_idx = self.masked_idx[index]
        is_next = self.is_next[index]
        return torch.tensor(token_ids), torch.tensor(segment_tokens), torch.tensor(masked_token), torch.tensor(masked_idx), torch.tensor(is_next)
    
    def __len__(self) -> int:
        return self.bert_df.shape[0]

if __name__ == "__main__":
    ds = IMDBBERTDataset(path="../data/IMDB.csv")
    print(f"Shape: {ds.bert_df.shape}")
    print(ds.bert_df.head(10))
    print(ds[0])
