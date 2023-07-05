import torch
from torch import nn
import math

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
        self.pe = torch.zeros(size=(max_seq_len, d_model),
                              requires_grad=False)
        
        for pos in range(max_seq_len):
            for dim in range(d_model):
                if pos%2==0:
                    self.pe[pos, dim] = math.sin(pos//(10000**(2*dim//d_model)))
                else:
                    self.pe[pos, dim] = math.cos(pos//(10000**(2*dim//d_model)))

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
