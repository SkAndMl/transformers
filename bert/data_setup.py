import torch
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import pandas as pd
import random
from typing import Tuple

URL = "https://github.com/SK7here/Movie-Review-Sentiment-Analysis/raw/master/IMDB-Dataset.csv"
    
class IMDBMaskedBertDataset(Dataset):

    PAD_TOKEN = 0
    CLS_TOKEN = 1
    SEP_TOKEN = 2
    MASK_TOKEN = 3
    UNK_TOKEN = 4

    SPECIALS = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]

    def __init__(self,
                 path: str,
                 max_len: int=10) -> None:
        super().__init__()
        self.max_len = max_len
        self.df = pd.read_csv(path)

        self.tokenizer = get_tokenizer(tokenizer="spacy",
                                       language="en_core_web_sm")

        self.masked_tokens = []
        self.masked_token_idxs = []
        self.sentences = []

        self._prepare_data()
        self.sentences = torch.tensor(self.sentences)
        self.masked_token_idxs = torch.tensor(self.masked_token_idxs)
        self.masked_tokens = torch.tensor(self.masked_tokens)


    def __len__(self) -> int:
        return len(self.sentences)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sentences[index], self.masked_tokens[index], self.masked_token_idxs[index]
    
    def _build_vocab(self, data_iter):
        for sent in data_iter:
            yield self.tokenizer(sent)


    def _prepare_data(self):

        for i in range(self.df.shape[0]):
            for sent in self.df.iloc[i, 0].split('. '):
                if 1 < len(self.tokenizer(sent)) <= self.max_len:
                    self.sentences.append(sent)
        
        self.vocab = build_vocab_from_iterator(self._build_vocab(self.sentences),
                                               min_freq=2,
                                               special_first=True,
                                               specials=self.SPECIALS)
        self.vocab.set_default_index(self.UNK_TOKEN)
        
        self._mask_data()

        for i in range(len(self.sentences)):
            self.sentences[i] = self.vocab(self.sentences[i])
            self.masked_tokens[i] = self.vocab(self.masked_tokens[i])

    
    def _mask_data(self):

        for i in range(len(self.sentences)):
            sentence = self.tokenizer(self.sentences[i])
            mask_idx = random.randint(0, len(sentence)-1)
            self.masked_token_idxs.append(mask_idx+1)
            self.masked_tokens.append([sentence[mask_idx]])
            sentence[mask_idx] = "[MASK]"
            sentence = ["[CLS]"] + sentence + ["[SEP]"]
            while len(sentence)<self.max_len+2:
                sentence.append("[PAD]")
            self.sentences[i] = sentence

if __name__ == "__main__":
    ds = IMDBMaskedBertDataset(path=URL)
    sentence, masked_token, masked_token_idx = ds[random.randint(0, len(ds)-1)]
    print(f"Sentence: {sentence}")
    print(f"Masked token: {masked_token}")
    print(f"Masked token idx posn: {masked_token_idx}")