import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from typing import Tuple, List, Dict
from torchinfo import summary


class Embedding(nn.Module):

    def __init__(self,
                 d_model: int=512,
                 vocab_size: int=1000,
                 max_seq_len: int=10,
                 dropout: float=0.1):
        
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
        embed_out = self.embedding(x) # embed_out -> [batch_size, seq_len, ]
        return self.dropout(self.pe[:, embed_out.size(1), :] + embed_out)


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


class TransformerDecoder(nn.Module):

    def __init__(self,
                 num_decoders: int=6,
                 d_model: int=512,
                 num_heads: int=8,
                 d_ff: int=2048,
                 attn_dropout: float=0.1,
                 ff_dropout: float=0.1) -> None:
        
        super().__init__()
        self.num_decoders = num_decoders
        self.decoders = nn.ModuleList([TransformerDecoderLayer(d_model, num_heads, d_ff, attn_dropout, ff_dropout) for _ in range(num_decoders)])
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask=None, tgt_mask=None) -> torch.Tensor:

        for i in range(self.num_decoders):
            x = self.decoders[i](x, encoder_output, src_mask, tgt_mask)
        return x
    
class Transformer(nn.Module):

    def __init__(self,
                 num_encoders: int=6,
                 num_decoders: int=6,
                 output_size: int=1000,
                 d_model: int=512,
                 num_heads: int=8,
                 d_ff: int=2048,
                 attn_dropout: float=0.1,
                 ff_dropout: float=0.1) -> None:
        
        super().__init__()
        self.encoder = TransformerEncoder(num_encoders, d_model, num_heads, d_ff, attn_dropout, ff_dropout)
        self.decoder = TransformerDecoder(num_decoders, d_model, num_heads, d_ff, attn_dropout, ff_dropout)
        self.projection = nn.Linear(d_model, output_size)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask=None, tgt_mask=None) -> torch.Tensor:
        encoder_output = self.encoder(src, src_mask)
        return self.projection(self.decoder(tgt, encoder_output, src_mask, tgt_mask))



if __name__ == '__main__':
    t = Transformer()
    summary(model=t,
            input_size=[(3, 3, 512), (3, 2, 512)])