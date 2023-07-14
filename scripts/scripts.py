from torch import nn
import torch
import math

from typing import Tuple


class TransformerEmbedding(nn.Module):

    def __init__(self,
                 d_model: int=512,
                 vocab_size: int=1000,
                 max_seq_len: int=10,
                 dropout: float=0.1):
        
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

        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=d_model)
        self.pe = torch.zeros(max_seq_len, d_model, requires_grad=False)
        
        for i in range(max_seq_len):
            for j in range(d_model):
                if j%2==0:
                    self.pe[i, j] = math.sin(i/(10000**(2*j/d_model)))
                else:
                    self.pe[i, j] = math.cos(i/(10000**(2*j/d_model)))
        
        self.pe = self.pe.unsqueeze(0)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> [batch_size, seq_len]
        embed_out = self.embedding(x) # embed_out -> [batch_size, seq_len, d_model]
        return self.dropout(self.pe[:, embed_out.size(1), :] + embed_out)
    

class BERTEmbedding(nn.Module):

    def __init__(self,
                 d_model: int=512,
                 vocab_size: int=1000,
                 max_seq_len: int=100,
                 dropout: float=0.1,
                 device="cpu") -> None:

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
    


class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, 
                 d_model: int=512,
                 num_heads: int=8):
        
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model%num_heads == 0, f"Dimensionality of the model {d_model} should be divisible by the number of heads {num_heads}"

        self.W_v = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    @staticmethod
    def scaled_dot_product_attention(query: torch.Tensor,
                                     key: torch.Tensor,
                                     value: torch.Tensor,
                                     mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # query = key = value -> [batch, query_len/key_len/value_len, num_heads, d_k]
        d_k = query.size(-1)
        compatability = torch.einsum("bqhd,bkhd->bhqk", [query, key]) # compatability -> [batch, num_heads, query_len, key_len]
        if mask is not None:
            compatability = compatability.masked_fill(mask==0, -float("inf"))
        compatability = torch.softmax(compatability, dim=-1)/math.sqrt(d_k)
        x = torch.einsum("bhqk,bkhd->bqhd", [compatability, value]) # x -> [batch, query_len, num_heads, d_k]
        return x, compatability
    
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask=None):
        
        b, query_len, key_len, d_k = query.shape[0], query.shape[1], key.shape[1], self.d_model//self.num_heads 
        query, key, value = self.W_q(query), self.W_k(key), self.W_v(value)
        query = query.view(b, query_len, self.num_heads, d_k)
        key = key.view(b, key_len, self.num_heads, d_k)
        value = value.view(b, key_len, self.num_heads, d_k)
        x, _ = MultiHeadAttentionLayer.scaled_dot_product_attention(query, key, value, mask)
        return self.W_o(x.reshape(b, query_len, self.d_model))
    

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self,
                 d_model: int=512,
                 num_heads: int=8,
                 dropout: float=0.1) -> None:
        super().__init__()
        self.multihead_attention = MultiHeadAttentionLayer(d_model, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None) -> torch.Tensor:
        return self.layer_norm(query + self.dropout(self.multihead_attention(query, key, value, mask)))
    

class FeedForwardBlock(nn.Module):

    def __init__(self,
                 d_model: int=512,
                 d_ff: int=2048,
                 dropout: float=0.1) -> None:
        
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(x + self.dropout(self.fc(x)))
    

class TransformerEncoderLayer(nn.Module):

    def __init__(self,
                 d_model: int=512,
                 num_heads: int=8,
                 d_ff: int=2048,
                 attn_dropout: float=0.1,
                 ff_dropout: float=0.1) -> None:
        
        super().__init__()
        self.multihead_attention_block = MultiHeadAttentionBlock(d_model, num_heads, attn_dropout)
        self.ff_block = FeedForwardBlock(d_model, d_ff, ff_dropout)
    
    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:

        return self.ff_block(self.multihead_attention_block(x, x, x, mask))


class TransformerEncoder(nn.Module):

    def __init__(self,
                 num_encoders: int=6,
                 d_model: int=512,
                 num_heads: int=8,
                 d_ff: int=2048,
                 attn_dropout: float=0.1,
                 ff_dropout: float=0.1) -> None:
        
        super().__init__()
        self.num_encoders = num_encoders
        self.encoders = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, d_ff, attn_dropout, ff_dropout) for _ in range(num_encoders)])
    
    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:

        for i in range(self.num_encoders):
            x = self.encoders[i](x, mask)
        return x
    
class TransformerDecoderLayer(nn.Module):

    def __init__(self,
                 d_model: int=512,
                 num_heads: int=8,
                 d_ff: int=2048,
                 attn_dropout: float=0.1,
                 ff_dropout: float=0.1) -> None:
        
        super().__init__()
        self.multihead_attention_block = MultiHeadAttentionBlock(d_model, num_heads, attn_dropout)
        self.cross_attention_block = MultiHeadAttentionBlock(d_model, num_heads, attn_dropout)
        self.ff_block = FeedForwardBlock(d_model, d_ff, ff_dropout)
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask=None, tgt_mask=None) -> torch.Tensor:
        x = self.multihead_attention_block(x, x, x, tgt_mask)
        x = self.cross_attention_block(x, encoder_output, encoder_output, src_mask)
        return self.ff_block(x)


def create_padding_mask(batch: torch.Tensor,
                        padding_idx: int=0) -> torch.Tensor:
    # batch -> [batch_size, max_seq_len]
    mask = batch != padding_idx
    return mask.unsqueeze(1).unsqueeze(2) # mask -> [batch_size, 1, 1, max_seq_len] 
