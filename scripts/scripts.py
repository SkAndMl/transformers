from torch import nn
import torch
from torch.nn import functional as F
import math
from typing import Tuple


class Embedding(nn.Module):

    def __init__(self,
                 d_model: int=512,
                 vocab_size: int=1000,
                 max_seq_len: int=10,
                 dropout: float=0.2):
        """
            Embedding generates learnable representation of an input sequence which encodes
            contextual, semantic meaning for each word.
            Params:
                d_model(int): specifies the embedding dimension for each token/word
                vocab_size(int): number of embeddings that would be needed. # of unique words
                max_seq_len(int): the maximum sequence length of an input sequence. Used for generation positional encoding
                dropout(float): probability of dropout applied on the final embedding output
        """
        
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size,
                                                  embedding_dim=d_model)
        self.position_embedding_table = nn.Embedding(num_embeddings=max_seq_len,
                                                     embedding_dim=d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x => [B, S]
        B, S = x.shape
        token_emb = self.token_embedding_table(x) # [B, S, D]
        pos_emb = self.position_embedding_table(torch.arange(S, dtype=torch.long)) # [S, D]
        out = self.dropout(token_emb+pos_emb)
        return self.dropout(out)

    

class BERTEmbedding(nn.Module):

    def __init__(self,
                 d_model: int=512,
                 vocab_size: int=1000,
                 max_seq_len: int=100,
                 dropout: float=0.1) -> None:

        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=d_model)
        pe = torch.zeros(size=(max_seq_len, d_model),
                         requires_grad=False)

        for pos in range(max_seq_len):
            for dim in range(d_model):
                if pos%2==0:
                    pe[pos, dim] = math.sin(pos//(10000**(2*dim//d_model)))
                else:
                    pe[pos, dim] = math.cos(pos//(10000**(2*dim//d_model)))
        self.register_buffer('pe', pe)

        self.segment_embedding = nn.Embedding(num_embeddings=2,
                                              embedding_dim=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def __repr__(self) -> str:
        return f"BERTEmbedding(d_model={self.d_model}, vocab_size={self.vocab_size})"

    def __str__(self) -> str:
        return f"BERTEmbedding(d_model={self.d_model}, vocab_size={self.vocab_size})"

    def forward(self,
                x: torch.Tensor,
                segment_tokens: torch.Tensor) -> torch.Tensor:
        # x -> [batch_size, max_seq_len]
        token_embeddings = self.token_embedding(x)
        position_encoding = self.pe[:x.shape[1], :].unsqueeze(0) # positional_encoding -> [1, max_seq_len, d_model]
        segment_embedding = self.segment_embedding(segment_tokens)
        return self.dropout(token_embeddings + position_encoding + segment_embedding)
    


class AttentionHead(nn.Module):

    def __init__(self,
                 config) -> None:
        
        super().__init__()

        self.d_model = config["d_model"]
        self.head_dim = config["head_dim"]

        self.query = nn.Linear(self.d_model, self.head_dim)
        self.key = nn.Linear(self.d_model, self.head_dim)
        self.value = nn.Linear(self.d_model, self.head_dim)
        self.dropout = nn.Dropout(p=config["dropout"])

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask=None) -> torch.Tensor:
        
        # query => [B, Q, D]
        # key => [B, K, D]
        # value => [B, K, D]

        q = self.query(query) # B, Q, HEAD_DIM 
        k = self.key(key) # B, K, HEAD_DIM
        v = self.value(value) # B, K, HEAD_DIM

        weights = q @ k.transpose(1, 2) # B, Q, K
        if mask:
            weights = weights.masked_fill(mask==0, value=float("-inf"))
        weights = F.softmax(weights/math.sqrt(self.head_dim))
        out = weights @ v # [B, Q, K] x [B, K, HEAD_DIM] => [B, Q, HEAD_DIM]
        return self.dropout(out)



class MultiHeadAttention(nn.Module):

    def __init__(self,
                 config) -> None:
         
         super().__init__()
         self.sa_heads = nn.ModuleList([AttentionHead(config) for _ in range(config["n_heads"])])
         self.proj = nn.Linear(config["d_model"], config["d_model"])
         self.dropout = nn.Dropout(p=config["dropout"])
        
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask=None) -> torch.Tensor:
        
        out = torch.cat([h(query, key, value) for h in self.sa_heads], dim=-1)
        out = self.proj(out)
        return self.dropout(out)


class FeedForward(nn.Module):

    def __init__(self,
                 config):
        
        super().__init__()
        d_model = config["d_model"]
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4, d_model),
            nn.Dropout(p=config["dropout"])
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.net(x)
        return x


class EncoderBlock(nn.Module):

    def __init__(self,
                 config) -> None:
        
        super().__init__()
        self.mha = MultiHeadAttention(config)
        self.ln1 = nn.LayerNorm(normalized_shape=config["d_model"])
        self.ff = FeedForward(config)
        self.ln2 = nn.LayerNorm(normalized_shape=config["d_model"])
    
    def forward(self,
                x: torch.Tensor,
                mask=None) -> torch.Tensor:
        
        x = x + self.mha(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class Encoder(nn.Module):

    def __init__(self,
                 config) -> None:
        
        super().__init__()
        self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config["n_encoders"])])
    
    def forward(self,
                x: torch.Tensor,
                mask=None):
        
        for block in self.blocks:
            x = block(x, mask)
        return x


def create_padding_mask(batch: torch.Tensor,
                        padding_idx: int=0) -> torch.Tensor:
    # batch -> [batch_size, max_seq_len]
    mask = batch != padding_idx
    return mask.unsqueeze(1).unsqueeze(2) # mask -> [batch_size, 1, 1, max_seq_len] 
